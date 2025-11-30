"""PyTorch dataset and data loaders for event sequences"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
import numpy as np
import logging

from .event_tokenizer import EventTokenizer
from .sequence_builder import SequenceBuilder

logger = logging.getLogger(__name__)


class EventDataset(Dataset):
    """PyTorch dataset for event sequences"""

    def __init__(
        self,
        sequences: List[List[Dict[str, Any]]],
        tokenizer: EventTokenizer,
        max_event_tokens: int = 20,
    ):
        """
        Initialize dataset.

        Args:
            sequences: List of event sequences
            tokenizer: Event tokenizer
            max_event_tokens: Maximum tokens per event
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_event_tokens = max_event_tokens

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.

        Returns:
            Dictionary with:
                - input_ids: Token indices [seq_len, max_event_tokens]
                - attention_mask: Attention mask [seq_len, max_event_tokens]
                - seq_length: Actual sequence length
        """
        sequence = self.sequences[idx]

        # Tokenize each event in sequence
        tokenized_events = []
        for event in sequence:
            tokens = self.tokenizer.tokenize_event(event)
            # Truncate or pad to max_event_tokens
            if len(tokens) > self.max_event_tokens:
                tokens = tokens[: self.max_event_tokens]
            tokenized_events.append(tokens)

        # Pad sequences to same length
        max_len = max(len(evt) for evt in tokenized_events) if tokenized_events else 1
        max_len = min(max_len, self.max_event_tokens)

        input_ids = []
        attention_mask = []

        for tokens in tokenized_events:
            # Pad tokens
            padded = tokens + [self.tokenizer.vocab.token2idx[self.tokenizer.vocab.PAD_TOKEN]] * (
                max_len - len(tokens)
            )
            input_ids.append(padded[:max_len])

            # Create attention mask
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            attention_mask.append(mask[:max_len])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "seq_length": torch.tensor(len(sequence), dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads sequences in batch to same length.
    """
    # Find max sequence length in batch
    max_seq_len = max(item["input_ids"].shape[0] for item in batch)
    max_token_len = max(item["input_ids"].shape[1] for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_seq_lengths = []

    for item in batch:
        seq_len, token_len = item["input_ids"].shape

        # Pad sequence dimension
        pad_seq = max_seq_len - seq_len
        pad_token = max_token_len - token_len

        # Pad input_ids
        if pad_seq > 0 or pad_token > 0:
            input_ids = torch.nn.functional.pad(
                item["input_ids"], (0, pad_token, 0, pad_seq), value=0
            )
        else:
            input_ids = item["input_ids"]

        # Pad attention mask
        if pad_seq > 0 or pad_token > 0:
            attention_mask = torch.nn.functional.pad(
                item["attention_mask"], (0, pad_token, 0, pad_seq), value=0
            )
        else:
            attention_mask = item["attention_mask"]

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_seq_lengths.append(item["seq_length"])

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "seq_length": torch.stack(batch_seq_lengths),
    }


def create_dataloaders(
    sequences: List[List[Dict[str, Any]]],
    tokenizer: EventTokenizer,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        sequences: List of event sequences
        tokenizer: Event tokenizer
        batch_size: Batch size
        train_split: Training data fraction
        val_split: Validation data fraction
        num_workers: Number of worker threads
        shuffle: Whether to shuffle data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data
    n_train = int(len(sequences) * train_split)
    n_val = int(len(sequences) * val_split)

    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train : n_train + n_val]
    test_sequences = sequences[n_train + n_val :]

    logger.info(
        f"Data split: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test"
    )

    # Create datasets
    train_dataset = EventDataset(train_sequences, tokenizer)
    val_dataset = EventDataset(val_sequences, tokenizer)
    test_dataset = EventDataset(test_sequences, tokenizer)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
