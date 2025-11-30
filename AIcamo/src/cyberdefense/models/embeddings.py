"""Embedding layers for event entities"""

import torch
import torch.nn as nn
import math


class EventEmbedding(nn.Module):
    """
    Embedding layer for event entities.

    Maps discrete entity tokens (file paths, process names, etc.)
    to continuous dense vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_event_tokens: int = 20,
        dropout: float = 0.1,
    ):
        """
        Initialize embedding layer.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            max_event_tokens: Maximum tokens per event
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_event_tokens = max_event_tokens

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Positional encoding for token positions within an event
        self.positional_encoding = PositionalEncoding(embedding_dim, max_event_tokens)

        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices [batch, seq_len, max_event_tokens]
            attention_mask: Attention mask [batch, seq_len, max_event_tokens]

        Returns:
            Embeddings [batch, seq_len, embedding_dim]
        """
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(-1)
            
        batch_size, seq_len, num_tokens = input_ids.shape
        
        # Flatten for embedding lookup
        flat_input = input_ids.view(-1, num_tokens)  # [batch * seq_len, num_tokens]

        # Get token embeddings
        token_embeds = self.token_embedding(flat_input)  # [batch * seq_len, num_tokens, embedding_dim]

        # Add positional encoding
        token_embeds = self.positional_encoding(token_embeds)

        # Apply attention mask if provided
        if attention_mask is not None:
            flat_mask = attention_mask.view(-1, num_tokens).unsqueeze(-1)
            token_embeds = token_embeds * flat_mask

        # Aggregate token embeddings to event-level (mean pooling)
        if attention_mask is not None:
            mask_sum = flat_mask.sum(dim=1, keepdim=False).clamp(min=1)
            event_embeds = (token_embeds * flat_mask).sum(dim=1) / mask_sum
        else:
            event_embeds = token_embeds.mean(dim=1)

        # Reshape back to sequence
        event_embeds = event_embeds.view(batch_size, seq_len, self.embedding_dim)

        # Normalize and dropout
        event_embeds = self.layer_norm(event_embeds)
        event_embeds = self.dropout(event_embeds)

        return event_embeds


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x
