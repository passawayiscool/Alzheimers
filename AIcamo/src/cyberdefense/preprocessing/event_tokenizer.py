"""Event tokenization and vocabulary management"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Manages vocabulary for entity embeddings.

    Maps unique entities (file paths, process names, registry keys, etc.)
    to integer indices for embedding lookup.
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"  # Start of sequence
    EOS_TOKEN = "<EOS>"  # End of sequence

    def __init__(self, max_vocab_size: int = 50000):
        """
        Initialize vocabulary.

        Args:
            max_vocab_size: Maximum vocabulary size (most frequent tokens)
        """
        self.max_vocab_size = max_vocab_size
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_counts: Counter = Counter()

        # Add special tokens
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        for token in [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build_from_tokens(self, tokens: List[str]):
        """
        Build vocabulary from token list.

        Args:
            tokens: List of all tokens from training data
        """
        logger.info(f"Building vocabulary from {len(tokens)} tokens...")

        # Count token frequencies
        self.token_counts = Counter(tokens)

        # Keep most frequent tokens
        most_common = self.token_counts.most_common(
            self.max_vocab_size - len(self.token2idx)
        )

        for token, count in most_common:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

        logger.info(f"Vocabulary built with {len(self.token2idx)} tokens")

    def encode(self, token: str) -> int:
        """
        Convert token to index.

        Args:
            token: Token string

        Returns:
            Token index
        """
        return self.token2idx.get(token, self.token2idx[self.UNK_TOKEN])

    def decode(self, idx: int) -> str:
        """
        Convert index to token.

        Args:
            idx: Token index

        Returns:
            Token string
        """
        return self.idx2token.get(idx, self.UNK_TOKEN)

    def encode_sequence(self, tokens: List[str]) -> List[int]:
        """Encode a sequence of tokens"""
        return [self.encode(token) for token in tokens]

    def decode_sequence(self, indices: List[int]) -> List[str]:
        """Decode a sequence of indices"""
        return [self.decode(idx) for idx in indices]

    def save(self, path: Path):
        """Save vocabulary to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        vocab_data = {
            "token2idx": self.token2idx,
            "idx2token": self.idx2token,
            "token_counts": dict(self.token_counts),
            "max_vocab_size": self.max_vocab_size,
        }

        with open(path, "wb") as f:
            pickle.dump(vocab_data, f)

        logger.info(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from disk"""
        with open(path, "rb") as f:
            vocab_data = pickle.load(f)

        vocab = cls(max_vocab_size=vocab_data["max_vocab_size"])
        vocab.token2idx = vocab_data["token2idx"]
        vocab.idx2token = {int(k): v for k, v in vocab_data["idx2token"].items()}
        vocab.token_counts = Counter(vocab_data["token_counts"])

        logger.info(f"Vocabulary loaded from {path} ({len(vocab.token2idx)} tokens)")
        return vocab

    def __len__(self) -> int:
        return len(self.token2idx)


class EventTokenizer:
    """
    Tokenizes system events into sequences of entity tokens.

    Uses whole-entity embedding approach where each unique file path,
    process name, etc. is treated as a single token.
    """

    def __init__(self, vocab: Optional[Vocabulary] = None):
        """
        Initialize tokenizer.

        Args:
            vocab: Vocabulary instance (None = will build from data)
        """
        self.vocab = vocab

        # Field extractors for each event type
        self.field_extractors = {
            "process_create": ["process_name", "parent_process", "command_line"],
            "PROCESS_START": ["process_name", "parent_process", "file_path", "command_line", "artifact_path"], # Added artifact_path
            "process_exit": ["process_name"],
            "PROCESS_END": ["process_name"],
            "file_read": ["process_name", "file_path"],
            "file_write": ["process_name", "file_path"],
            "file_delete": ["process_name", "file_path"],
            "registry_read": ["process_name", "key_path", "value_name"],
            "registry_write": ["process_name", "key_path", "value_name"],
            "network_connect": ["process_name", "remote_address", "protocol"],
            "NETWORK_CONNECTION": ["process_name", "dest_address", "dest_port", "protocol", "event_id", "artifact_path", "command_line"], # Added for Host GAN
            "network_dns": ["process_name", "dns_query"],
            "DNS_QUERY": ["process_name", "query_name", "query_type", "artifact_path", "command_line"], # Added for Host GAN
            "BROWSER_HISTORY": ["process_name", "url", "title", "artifact_path"], # Added for Host GAN
            "dll_load": ["process_name", "dll_path"],
            "PRIVILEGE_ASSIGNED": ["user"],
            "LOGON": ["user", "logon_type"],
            "LOGOFF": ["user"],
            "LOGON_FAILURE": ["user"],
        }

    def extract_entities(self, event: Dict[str, Any]) -> List[str]:
        """
        Extract entity tokens from event.

        Args:
            event: Event dictionary

        Returns:
            List of entity tokens
        """
        tokens = []

        # Add event type as first token
        event_type = event.get("event_type", "unknown")
        tokens.append(f"EVENT:{event_type}")

        # Extract relevant fields based on event type
        fields = self.field_extractors.get(event_type, [])
        
        # Fields that should be split into Dir and File
        path_fields = {"file_path", "dll_path", "image_path"}

        for field in fields:
            value = event.get(field)
            if value and value != "unknown" and value != "":
                # Check if this field should be split
                if field in path_fields and isinstance(value, str):
                    # Split into Directory and Filename
                    # We use a simple regex to find the last separator
                    match = re.search(r'^(.*)[\\/]([^\\/]+)$', value)
                    if match:
                        directory = match.group(1)
                        filename = match.group(2)
                        tokens.append(f"{field.upper()}_DIR:{directory}")
                        tokens.append(f"{field.upper()}_FILE:{filename}")
                    else:
                        # No separator found, treat as just filename
                        tokens.append(f"{field.upper()}_FILE:{value}")
                else:
                    # Standard handling for non-path fields
                    tokens.append(f"{field.upper()}:{value}")

        # Add metadata if available
        if "user" in event:
            tokens.append(f"USER:{event['user']}")

        return tokens

    def build_vocabulary(self, events: List[Dict[str, Any]], max_vocab_size: int = 50000):
        """
        Build vocabulary from event list.

        Args:
            events: List of event dictionaries
            max_vocab_size: Maximum vocabulary size
        """
        logger.info(f"Building vocabulary from {len(events)} events...")

        all_tokens = []
        for event in events:
            tokens = self.extract_entities(event)
            all_tokens.extend(tokens)

        self.vocab = Vocabulary(max_vocab_size=max_vocab_size)
        self.vocab.build_from_tokens(all_tokens)

    def tokenize_event(self, event: Dict[str, Any]) -> List[int]:
        """
        Tokenize a single event.

        Args:
            event: Event dictionary

        Returns:
            List of token indices
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not initialized. Call build_vocabulary first.")

        tokens = self.extract_entities(event)
        return self.vocab.encode_sequence(tokens)

    def tokenize_events(self, events: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Tokenize multiple events.

        Args:
            events: List of event dictionaries

        Returns:
            List of token index sequences
        """
        return [self.tokenize_event(event) for event in events]

    def detokenize_event(self, token_indices: List[int]) -> Dict[str, Any]:
        """
        Convert token indices back to event-like structure.

        Args:
            token_indices: List of token indices

        Returns:
            Reconstructed event dictionary
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not initialized")

        tokens = self.vocab.decode_sequence(token_indices)

        event = {}
        path_fields = {"process_name", "parent_process", "file_path", "command_line", "dll_path", "image_path"}
        
        for token in tokens:
            if ":" in token:
                field, value = token.split(":", 1)
                field_upper = field.upper()
                
                # Check for DIR/FILE suffixes
                if field_upper.endswith("_DIR"):
                    base_field = field_upper[:-4].lower() # Remove _DIR
                    if base_field in event:
                         # If we already have a file part, prepend dir? 
                         # Or just store dir and wait for file?
                         # Simpler: Just store it and let the file part append to it if needed
                         event[base_field] = value
                    else:
                        event[base_field] = value
                        
                elif field_upper.endswith("_FILE"):
                    base_field = field_upper[:-5].lower() # Remove _FILE
                    if base_field in event:
                        # We have a directory, join them
                        event[base_field] = f"{event[base_field]}\\{value}"
                    else:
                        # No directory, just filename
                        event[base_field] = value
                        
                else:
                    # Standard field
                    event[field.lower()] = value

        return event

    def save(self, path: Path):
        """Save tokenizer to disk"""
        if self.vocab is None:
            raise ValueError("Cannot save tokenizer without vocabulary")

        vocab_path = Path(path) / "vocab.pkl"
        self.vocab.save(vocab_path)

        # Save field extractors
        config_path = Path(path) / "tokenizer_config.json"
        with open(config_path, "w") as f:
            json.dump({"field_extractors": self.field_extractors}, f, indent=2)

        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "EventTokenizer":
        """Load tokenizer from disk"""
        vocab_path = Path(path) / "vocab.pkl"
        vocab = Vocabulary.load(vocab_path)

        tokenizer = cls(vocab=vocab)

        # Load field extractors
        config_path = Path(path) / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                tokenizer.field_extractors = config["field_extractors"]

        logger.info(f"Tokenizer loaded from {path}")
        return tokenizer
