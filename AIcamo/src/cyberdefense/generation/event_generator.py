"""Host-based event generation using trained VAE-GAN model"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np
import logging

from ..models.vae_gan import VAEGANModel
from ..preprocessing.event_tokenizer import EventTokenizer

logger = logging.getLogger(__name__)


class EventGenerator:
    """
    Generates realistic host-based system events using trained VAE-GAN model.

    Supports:
    - Random generation from latent space
    - Context-aware generation (conditioning on recent events)
    - Temperature-controlled sampling
    - Top-k and nucleus (top-p) sampling
    """

    def __init__(
        self,
        model: VAEGANModel,
        tokenizer: EventTokenizer,
        device: str = "cuda",
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """
        Initialize event generator.

        Args:
            model: Trained VAE-GAN model
            tokenizer: Event tokenizer
            device: Device to run on
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
        """
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    @torch.no_grad()
    def generate(
        self,
        num_events: int = 100,
        batch_size: int = 1,
        context_events: Optional[List[Dict[str, Any]]] = None,
        max_length: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate event sequences.

        Args:
            num_events: Number of events to generate
            batch_size: Batch size for generation (number of parallel sequences)
            context_events: Optional context events to condition on

        Returns:
            List of generated events
        """
        generated_events = []
        logger.info(f"Generating {num_events} events...")

        # We generate events in batches until we have enough
        while len(generated_events) < num_events:
            # 1. Sample Latent Vector Z
            if context_events:
                z_context = self._encode_context(context_events)
                z_mean = torch.mean(z_context, dim=0, keepdim=True)
                noise = torch.randn(batch_size, self.model.latent_dim).to(self.device)
                z = z_mean + (noise * 0.5)
            else:
                z = torch.randn(batch_size, self.model.latent_dim).to(self.device)

            # 2. Generate Full Sequence from Z
            # The model's generator runs the autoregressive loop internally
            # Returns logits: [batch, seq_len, vocab_size]
            # Returns logits: [batch, seq_len, vocab_size]
            logits = self.model.generator(z, max_length=max_length)

            # 3. Sample Tokens
            # We sample from the logits. Note: The model used greedy decoding internally
            # to choose the path, so these logits correspond to that path.
            # We can apply temperature here to add slight noise to the final tokens,
            # but the structure is largely determined by Z.
            sampled_ids = self._sample_tokens(logits) # [batch, seq_len]

            # 4. Decode Sequences to Events
            for i in range(batch_size):
                seq_ids = sampled_ids[i]
                events = self._decode_sequence(seq_ids)
                
                # Attach latent vector to each event for downstream use
                current_z = z[i].detach().cpu()
                for evt in events:
                    evt["_host_latent"] = current_z
                
                generated_events.extend(events)
                
                if len(generated_events) >= num_events:
                    break
            
        return generated_events[:num_events]

    def generate_with_scenario(
        self, scenario_config: Dict[str, Any], duration: int = 3600
    ) -> List[Dict[str, Any]]:
        """
        Generate events for a specific scenario.

        Args:
            scenario_config: Scenario configuration
            duration: Scenario duration in seconds

        Returns:
            List of generated events
        """
        logger.info(f"Generating scenario: {scenario_config.get('description', 'Unknown')}")

        # Get seed events
        seed_events = scenario_config.get("seed_events", [])

        # Estimate number of events (approximate 1 event per second)
        num_events = duration

        # Generate events conditioned on seed
        generated = self.generate(num_events=num_events, context_events=seed_events)

        # Filter by target processes if specified
        target_processes = scenario_config.get("target_processes", [])
        if target_processes:
            filtered = []
            for event in generated:
                if any(proc.lower() in event.get("process_name", "").lower() for proc in target_processes):
                    filtered.append(event)
            generated = filtered

        logger.info(f"Generated {len(generated)} events for scenario")
        return generated

    def generate_autoregressive(
        self,
        context_events: List[Dict[str, Any]],
        num_steps: int = 50,
        context_window: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Generate events autoregressively using sliding context window.

        Args:
            context_events: Initial context events
            num_steps: Number of generation steps
            context_window: Size of context window

        Returns:
            List of generated events
        """
        generated_events = list(context_events)

        for step in range(num_steps):
            # Use last N events as context
            context = generated_events[-context_window:]

            # Generate next event(s)
            next_events = self.generate(num_events=1, context_events=context)

            generated_events.extend(next_events)

        return generated_events[len(context_events) :]

    def _encode_context(self, context_events: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode context events to latent code"""
        # Tokenize context events
        tokenized = []
        for event in context_events:
            tokens = self.tokenizer.tokenize_event(event)
            tokenized.append(tokens)

        # Pad to same length
        max_len = max(len(t) for t in tokenized) if tokenized else 1
        max_len = min(max_len, 20)

        padded = []
        for tokens in tokenized:
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            tokens += [0] * (max_len - len(tokens))
            padded.append(tokens)

        # Convert to tensor
        input_ids = torch.tensor([padded], dtype=torch.long).to(self.device)

        # Create attention mask
        attention_mask = (input_ids != 0).float()

        # Encode to latent
        z, _, _ = self.model.encode(input_ids, attention_mask)

        return z

    def _sample_tokens(self, logits: torch.Tensor, previous_token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample tokens from logits with temperature, filtering, and repetition penalty.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            previous_token_ids: Previously generated token IDs [batch, seq_len]

        Returns:
            Sampled token IDs [batch, seq_len]
        """
        # Apply repetition penalty (Frequency Penalty)
        if previous_token_ids is not None:
            # previous_token_ids: [batch, history_len]
            # logits: [batch, seq_len, vocab_size]
            
            # 1. Create a count of each token in the history for each batch element
            # We want a tensor of shape [batch, vocab_size]
            vocab_size = logits.size(-1)
            batch_size = logits.size(0)
            
            # One-hot encode history to count
            # [batch, history_len, vocab_size]
            history_one_hot = F.one_hot(previous_token_ids, num_classes=vocab_size).float()
            
            # Sum over history to get counts: [batch, vocab_size]
            token_counts = history_one_hot.sum(dim=1)
            
            # 2. Apply penalty
            # Broadcast counts to sequence length: [batch, 1, vocab_size]
            penalty_mask = token_counts.unsqueeze(1)
            
            # Subtract penalty (log-space)
            # Higher count = Lower logit
            logits = logits - (penalty_mask * 0.5) # 0.5 is the penalty strength

        # Apply temperature
        logits = logits / self.temperature

        # Apply top-k filtering
        if self.top_k > 0:
            k = min(self.top_k, logits.size(-1)) # Clamp k to vocab size
            indices_to_remove = logits < torch.topk(logits, k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        # Apply nucleus (top-p) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
        sampled_ids = sampled_ids.view(logits.size(0), logits.size(1))

        return sampled_ids

    def _decode_sequence(self, token_ids: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Decode token IDs to event dictionaries.

        Args:
            token_ids: Token IDs [seq_len]

        Returns:
            List of event dictionaries
        """
        events = []
        current_event = {}
        path_fields = {"process_name", "parent_process", "file_path", "command_line", "dll_path", "image_path"}

        for token_id in token_ids:
            token_id_val = token_id.item()

            # Skip special tokens
            if token_id_val in [0, 1, 2, 3]:  # PAD, UNK, SOS, EOS
                continue

            token = self.tokenizer.vocab.decode(token_id_val)

            if ":" not in token:
                continue

            field, value = token.split(":", 1)
            field_upper = field.upper()

            if field_upper == "EVENT":
                # Start of new event
                if current_event:
                    self._finalize_event(current_event)
                    events.append(current_event)
                current_event = {"event_type": value}
            else:
                # Field of current event
                if current_event:
                    # Store fields as-is, including _DIR and _FILE suffixes
                    # We will merge them in _finalize_event
                    # Handle duplicates (e.g. multiple FILE_PATH_DIR?) -> Overwrite is fine for now
                    current_event[field.lower()] = value
                else:
                    # Orphaned field at the start - create implicit PROCESS_START event
                    # This handles when the model generates COMMAND_LINE before EVENT token
                    if field_upper == "COMMAND_LINE":
                        current_event = {"event_type": "PROCESS_START", field.lower(): value}
                    # Ignore other orphaned fields

        # Append last event
        if current_event:
            self._finalize_event(current_event)
            events.append(current_event)

        # Filter out empty events (only event_type, no other fields)
        events = [e for e in events if len(e) > 1]

        return events

    def _finalize_event(self, event: Dict[str, Any]):
        """Merge _DIR and _FILE fields into full paths and extract process_name from command_line"""
        keys = list(event.keys())
        merged_fields = set()

        for key in keys:
            if key.endswith("_dir"):
                base = key[:-4]
                dir_val = event[key]
                file_val = event.get(f"{base}_file", "")

                if file_val:
                    event[base] = f"{dir_val}\\{file_val}"
                else:
                    event[base] = dir_val # Just dir

                merged_fields.add(key)
                merged_fields.add(f"{base}_file")

            elif key.endswith("_file"):
                base = key[:-5]
                if f"{base}_dir" not in event:
                    # Only handle if we haven't processed it via _dir
                    event[base] = event[key] # Just file
                    merged_fields.add(key)

        # Remove the temporary _dir and _file keys
        for key in merged_fields:
            if key in event:
                del event[key]

        # Extract process_name from command_line if missing
        if not event.get('process_name') and event.get('command_line'):
            import re
            cmd = event['command_line']
            # Extract .exe from command line (handles quotes and paths)
            match = re.search(r'([^\\/"]+\.exe)', cmd, re.IGNORECASE)
            if match:
                event['process_name'] = match.group(1)

    def _parse_token_to_event(self, token: str) -> Optional[Dict[str, Any]]:
        """Parse token string to event dictionary"""
        if ":" not in token:
            return None

        field, value = token.split(":", 1)
        field = field.lower()

        # Create minimal event structure
        if field == "event":
            return {"event_type": value, "timestamp": "", "pid": 0}
        else:
            return {field: value}
