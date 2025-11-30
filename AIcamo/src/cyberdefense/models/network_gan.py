import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkGenerator(nn.Module):
    """
    LSTM-based Generator for Network Traffic.
    Generates a sequence of packet features (Inter-arrival time, Size, Flags)
    conditioned on a 'functional' payload embedding and noise.
    """
    def __init__(self, input_dim=128, hidden_dim=256, seq_len=20, feature_dim=4, condition_dim=0):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim 
        
        # Input: Noise + Payload + Condition
        total_input_dim = input_dim + condition_dim
        
        self.lstm = nn.LSTM(total_input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, z, payload_emb, condition=None):
        """
        z: Latent noise [batch, input_dim]
        payload_emb: Functional payload embedding [batch, input_dim]
        condition: Host latent vector [batch, condition_dim] (Optional)
        """
        # Combine inputs
        x = z + payload_emb
        
        if condition is not None:
            # Concatenate condition? Or add if dimensions match?
            # For simplicity, let's concatenate to the input
            # We need to expand condition to match batch size if needed
            if condition.size(0) != x.size(0):
                condition = condition.expand(x.size(0), -1)
            x = torch.cat([x, condition], dim=1)
        
        # Repeat for sequence length
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1) # [batch, seq_len, total_input_dim]
        
        # Generate sequence
        out, _ = self.lstm(x)
        
        # Map to packet features
        features = self.fc(out) # [batch, seq_len, feature_dim]
        
        # Activation functions for specific features
        # TimeDelta: ReLU (must be positive)
        # Size: ReLU (must be positive)
        # Flags: Sigmoid/Softmax (categorical/binary)
        # Direction: Sigmoid (0 or 1)
        
        # For simplicity in this raw output, we return raw logits or bounded values
        # We'll post-process them in the wrapper
        return features

class NetworkDiscriminator(nn.Module):
    """
    Transformer-based Discriminator.
    Distinguishes real benign traffic from generated camouflaged traffic.
    """
    def __init__(self, feature_dim=4, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model)) # Max seq len 100
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, 1) # Real/Fake
        
    def forward(self, x):
        """
        x: Packet sequence [batch, seq_len, feature_dim]
        """
        # Embed features
        x = self.embedding(x) # [batch, seq_len, d_model]
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Global pooling (mean)
        x = x.mean(dim=1)
        
        # Classification
        validity = torch.sigmoid(self.fc_out(x))
        return validity

class NetworkGAN(nn.Module):
    """
    Combined Network GAN Model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generator = NetworkGenerator(
            input_dim=config.get("latent_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            seq_len=config.get("seq_len", 20),
            feature_dim=config.get("feature_dim", 4),
            condition_dim=config.get("condition_dim", 0)
        )
        self.discriminator = NetworkDiscriminator(
            feature_dim=config.get("feature_dim", 4),
            d_model=config.get("d_model", 128)
        )
        
    def generate(self, batch_size=1, payload=None, condition=None):
        device = next(self.parameters()).device
        z = torch.randn(batch_size, 128).to(device)
        
        if payload is None:
            payload = torch.zeros_like(z).to(device)
            
        return self.generator(z, payload, condition)
