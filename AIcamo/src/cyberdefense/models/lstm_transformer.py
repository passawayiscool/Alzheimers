"""LSTM-Transformer Hybrid Architecture"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class LSTMTransformerEncoder(nn.Module):
    """
    Hybrid Encoder: Bidirectional LSTM -> Transformer Encoder
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_lstm_layers: int = 2,
        num_transformer_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Bidirectional LSTM to capture local sequential dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Project LSTM output (2 * hidden) back to hidden_dim for Transformer
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Transformer Encoder for long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch, seq_len, input_dim]
            src_key_padding_mask: Mask for padding [batch, seq_len] (True for padded positions)
            
        Returns:
            Encoded sequence [batch, seq_len, hidden_dim]
        """
        # LSTM pass
        lstm_out, _ = self.lstm(x) # [batch, seq_len, 2*hidden]
        
        # Project to transformer dimension
        proj_out = self.projection(lstm_out) # [batch, seq_len, hidden]
        
        # Transformer pass
        # Note: nn.TransformerEncoder takes src_key_padding_mask where True means ignore
        out = self.transformer(proj_out, src_key_padding_mask=src_key_padding_mask)
        
        return out


class LSTMTransformerDecoder(nn.Module):
    """
    Hybrid Decoder: Transformer Decoder -> LSTM
    """
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        num_lstm_layers: int = 2,
        num_transformer_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_layers)
        
        # LSTM for sequential generation
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output projection
        self.output_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target embeddings [batch, tgt_len, hidden_dim]
            memory: Encoded memory [batch, src_len, hidden_dim]
            tgt_mask: Causal mask for autoregressive training
            
        Returns:
            Logits [batch, tgt_len, output_dim]
        """
        # Transformer Decoder pass
        trans_out = self.transformer(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # LSTM pass
        lstm_out, _ = self.lstm(trans_out)
        
        # Output projection
        logits = self.output_head(lstm_out)
        
        return logits
