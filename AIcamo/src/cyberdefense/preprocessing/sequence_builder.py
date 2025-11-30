"""Build temporal event sequences for training"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    Builds temporal sequences of events for training.

    Creates sliding windows of events to capture temporal dependencies.
    """

    def __init__(
        self,
        max_sequence_length: int = 128,
        min_sequence_length: int = 10,
        stride: int = 64,
    ):
        """
        Initialize sequence builder.

        Args:
            max_sequence_length: Maximum events in a sequence
            min_sequence_length: Minimum events in a sequence
            stride: Step size for sliding window
        """
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.stride = stride

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp string to timezone-aware UTC datetime"""
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                # Assume UTC if naive
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            from datetime import timezone
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    def build_sequences(
        self, events: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Build sequences from events using sliding window.

        Args:
            events: List of events (should be sorted by timestamp)

        Returns:
            List of event sequences
        """
        logger.info(f"Building sequences from {len(events)} events...")

        # Sort events by timestamp
        sorted_events = sorted(
            events,
            key=lambda e: self._parse_timestamp(e.get("timestamp", "1970-01-01T00:00:00")),
        )

        sequences = []
        for i in range(0, len(sorted_events) - self.min_sequence_length + 1, self.stride):
            seq = sorted_events[i : i + self.max_sequence_length]

            if len(seq) >= self.min_sequence_length:
                sequences.append(seq)

        logger.info(f"Built {len(sequences)} sequences")
        return sequences

    def build_sequences_by_process(
        self, events: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Build sequences grouped by process.

        Args:
            events: List of events

        Returns:
            List of event sequences
        """
        logger.info(f"Building process-based sequences from {len(events)} events...")

        # Group events by PID
        pid_events: Dict[int, List[Dict[str, Any]]] = {}

        for event in events:
            pid = event.get("pid", 0)
            if pid not in pid_events:
                pid_events[pid] = []
            pid_events[pid].append(event)

        # Build sequences for each process
        sequences = []
        for pid, proc_events in pid_events.items():
            # Sort by timestamp
            proc_events = sorted(
                proc_events,
                key=lambda e: self._parse_timestamp(e.get("timestamp", "1970-01-01T00:00:00")),
            )

            # Use sliding window on process events
            for i in range(0, len(proc_events) - self.min_sequence_length + 1, self.stride):
                seq = proc_events[i : i + self.max_sequence_length]

                if len(seq) >= self.min_sequence_length:
                    sequences.append(seq)

        logger.info(f"Built {len(sequences)} process-based sequences")
        return sequences

    def build_sequences_with_context(
        self, events: List[Dict[str, Any]], context_window: int = 32
    ) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """
        Build sequences with context and target.

        Args:
            events: List of events
            context_window: Size of context window

        Returns:
            List of (context, target) tuples
        """
        logger.info(f"Building context-target sequences from {len(events)} events...")

        sorted_events = sorted(
            events,
            key=lambda e: self._parse_timestamp(e.get("timestamp", "1970-01-01T00:00:00")),
        )

        sequences = []
        for i in range(context_window, len(sorted_events)):
            context = sorted_events[i - context_window : i]
            target = sorted_events[i : min(i + context_window, len(sorted_events))]

            if len(target) > 0:
                sequences.append((context, target))

        logger.info(f"Built {len(sequences)} context-target sequences")
        return sequences

    def pad_sequence(
        self, sequence: List[List[int]], max_length: int, pad_value: int = 0
    ) -> np.ndarray:
        """
        Pad sequence to max length.

        Args:
            sequence: List of token sequences
            max_length: Maximum sequence length
            pad_value: Padding value

        Returns:
            Padded numpy array
        """
        padded = np.full((max_length, max(len(s) for s in sequence) if sequence else 1), pad_value)

        for i, seq in enumerate(sequence[:max_length]):
            padded[i, : len(seq)] = seq

        return padded

    def create_attention_mask(self, sequence: List[List[int]], max_length: int) -> np.ndarray:
        """
        Create attention mask for sequence.

        Args:
            sequence: List of token sequences
            max_length: Maximum sequence length

        Returns:
            Attention mask (1 for real tokens, 0 for padding)
        """
        mask = np.zeros((max_length, max(len(s) for s in sequence) if sequence else 1))

        for i, seq in enumerate(sequence[:max_length]):
            mask[i, : len(seq)] = 1

        return mask
