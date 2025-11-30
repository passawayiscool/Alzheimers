import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class SimpleLSTMGenerator(nn.Module):
    """
    A simple LSTM-based event generator to replace the complex VAE-GAN.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_config = config.get("model", {})
        
        self.vocab_size = model_config.get("max_vocab_size", 1000)
        self.embedding_dim = model_config.get("embedding_dim", 64)
        self.hidden_size = model_config.get("lstm_hidden_size", 128)
        self.num_layers = model_config.get("lstm_num_layers", 1)
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, input_ids: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: Input -> Embedding -> LSTM -> Linear -> Logits
        """
        embeds = self.embedding(input_ids)
        output, hidden = self.lstm(embeds, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_token: int, max_length: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """
        Simple autoregressive generation.
        """
        device = next(self.parameters()).device
        input_seq = torch.tensor([[start_token]], device=device)
        hidden = None
        generated = [start_token]
        
        for _ in range(max_length):
            logits, hidden = self.forward(input_seq, hidden)
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token.item())
            input_seq = next_token
            
        return torch.tensor(generated, device=device)
