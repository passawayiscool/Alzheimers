"""VAE-GAN Model Architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .embeddings import EventEmbedding
from .lstm_transformer import LSTMTransformerEncoder, LSTMTransformerDecoder

class VAEGANModel(nn.Module):
    """
    VAE-GAN Model for Event Generation.
    
    Combines:
    1. VAE: Encodes events to latent space, decodes back to events.
    2. GAN: Discriminator forces generated events to be realistic.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_config = config.get("model", {})
        
        # Dimensions
        self.vocab_size = model_config.get("max_vocab_size", 50000)
        self.embedding_dim = model_config.get("embedding_dim", 256)
        self.hidden_dim = model_config.get("hidden_dim", 512)
        self.latent_dim = model_config.get("latent_dim", 256)
        self.max_event_tokens = model_config.get("max_event_tokens", 512)
        
        # Components
        self.embedding = EventEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_event_tokens=self.max_event_tokens
        )
        
        # Encoder (Event Sequence -> Latent)
        self.encoder_backbone = LSTMTransformerEncoder(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        # VAE Heads (Hidden -> Mu, LogVar)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # Decoder (Latent -> Event Sequence)
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_projection = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.decoder_backbone = LSTMTransformerDecoder(
            output_dim=self.vocab_size,
            hidden_dim=self.hidden_dim
        )
        
        # Discriminator (Event Sequence -> Real/Fake)
        self.discriminator = Discriminator(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent space"""
        # 1. Embed
        embeddings = self.embedding(input_ids, attention_mask)
        
        # Mask
        if attention_mask is not None:
            # attention_mask is [batch, seq_len, tokens]
            # We need [batch, seq_len] for transformer
            # If all tokens are padding (0), then event is padding
            seq_mask = (input_ids[:, :, 0] == 0)
        else:
            seq_mask = None
            
        # 2. Encode backbone
        # WORKAROUND: Pass None for mask to avoid PyTorch 2.x crash
        encoded = self.encoder_backbone(embeddings, src_key_padding_mask=None)
        
        # Pool
        if seq_mask is not None:
            mask_float = (~seq_mask).float().unsqueeze(-1)
            pooled = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)
            
        # 3. VAE Heads
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VAE training.
        
        Args:
            input_ids: [batch, seq_len, tokens_per_event]
            attention_mask: [batch, seq_len, tokens_per_event]
            
        Returns:
            Dictionary of outputs
        """
        # 1. Embed inputs
        # input_ids: [batch, seq_len, tokens] -> embeddings: [batch, seq_len, embed_dim]
        # We need to collapse the token dimension for the encoder
        
        # EventEmbedding handles [batch, seq_len, tokens] -> [batch, seq_len, embed_dim]
        embeddings = self.embedding(input_ids, attention_mask)
        
        # Create sequence mask (batch, seq_len) - True where padding
        # Assuming padding is 0. If any token in event is not 0, it's not padding.
        # Or use the first token to determine padding.
        if attention_mask is not None:
            # attention_mask is [batch, seq_len, tokens]
            # We need [batch, seq_len] for transformer
            # If all tokens are padding (0), then event is padding
            seq_mask = (input_ids[:, :, 0] == 0)
        else:
            seq_mask = None
            
        # 2. Encode
        encoded = self.encoder_backbone(embeddings, src_key_padding_mask=seq_mask)
        
        # Pool to get single vector for latent space (use last hidden state or mean)
        # Here we use mean pooling over sequence, masking padding
        if seq_mask is not None:
            mask_float = (~seq_mask).float().unsqueeze(-1)
            pooled = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)
            
        # 3. VAE Latent Space
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)
        
        # 4. Decode
        # Project latent back to sequence dimension start
        # For autoregressive decoding during training, we use teacher forcing
        # The decoder takes: Target (shifted input) and Memory (Latent Z projected)
        
        # Expand z to match sequence length for "memory" context? 
        # Or just use z as the initial state?
        # Standard Transformer VAE: Z is the memory.
        
        z_projected = self.decoder_input(z).unsqueeze(1) # [batch, 1, hidden]
        # We repeat Z to act as the "memory" for cross-attention
        memory = z_projected.repeat(1, input_ids.size(1), 1) # [batch, seq_len, hidden]
        
        # Causal Mask for Decoder
        seq_len = input_ids.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(input_ids.device)
        
        # Decoder forward
        # Note: In standard VAE, we reconstruct X given Z.
        # Here we use embeddings as target input (teacher forcing)
        # Project embeddings to hidden_dim for decoder
        tgt_projected = self.decoder_projection(embeddings)
        
        generated_logits = self.decoder_backbone(
            tgt=tgt_projected, # Teacher forcing
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=seq_mask,
            memory_key_padding_mask=seq_mask
        )
        
        # 5. Discriminator (on generated output vs real)
        # For training, we need to pass gradients properly.
        # Discriminator takes sequence of embeddings.
        # We can approximate "generated embeddings" by softmaxing logits * embedding matrix?
        # Or just pass the hidden states?
        # Standard VAE-GAN: Discriminator on X_real and X_recon
        
        # For the forward pass, we return the logits for reconstruction loss
        
        return {
            "generated_logits": generated_logits,
            "mu": mu,
            "logvar": logvar,
            "z": z
        }
        
    def generator(self, z: torch.Tensor, max_length: int = 128, start_token: int = 1) -> torch.Tensor:
        """
        Generate logits from latent code (inference mode).
        Performs autoregressive decoding.
        """
        batch_size = z.size(0)
        device = z.device
        
        # Project Z to hidden state
        # z: [batch, latent] -> [batch, hidden]
        z_hidden = self.decoder_input(z)
        
        # Memory for decoder (expand z to sequence length?)
        # In this architecture, we use Z as the initial state or memory.
        # Let's use Z projected as memory for cross-attention
        # We need memory to be [batch, seq_len, hidden].
        # Since we don't have a source sequence, we can repeat Z?
        # Or use a fixed length memory?
        memory = z_hidden.unsqueeze(1).repeat(1, max_length, 1) # [batch, max_len, hidden]
        
        # Start token
        # [batch, 1]
        curr_tokens = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        all_logits = []
        
        for i in range(max_length):
            # Embed current sequence
            # [batch, curr_len, embed_dim]
            tgt_emb = self.embedding(curr_tokens)
            
            # Project to hidden
            tgt_projected = self.decoder_projection(tgt_emb)
            
            # Causal mask
            seq_len = tgt_projected.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)
            
            # Decoder forward
            # [batch, curr_len, vocab_size]
            logits = self.decoder_backbone(
                tgt=tgt_projected,
                memory=memory[:, :seq_len, :], # Match length? Or full memory?
                tgt_mask=tgt_mask
            )
            
            # Get last token logits
            last_logits = logits[:, -1, :] # [batch, vocab]
            all_logits.append(last_logits.unsqueeze(1))
            
            # Greedy decode for next step (or sample?)
            # For generator method, we usually return logits and let caller sample.
            # But we need next token to continue.
            # Let's use greedy for the loop, but return logits for sampling?
            # Or just return logits and let EventGenerator sample?
            # EventGenerator expects [batch, seq_len, vocab]
            
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1).unsqueeze(1) # [batch, 1]
            
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            
        # Stack logits
        # [batch, max_length, vocab]
        return torch.cat(all_logits, dim=1)


class Discriminator(nn.Module):
    """
    Discriminator: Classifies sequence as Real or Fake
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1) # Logits (no sigmoid, use BCEWithLogits)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch, seq_len, input_dim]
            
        Returns:
            Logits [batch, 1]
        """
        # Encode
        encoded = self.encoder(x)
        
        # Pool (mean)
        pooled = encoded.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
