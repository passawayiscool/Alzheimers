import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdefense.models.network_gan import NetworkGAN

def generate_benign_data(num_samples=1000, seq_len=20):
    """
    Generate synthetic 'benign' traffic features for training.
    Features: [TimeDelta, Size, Flags, Direction]
    """
    data = []
    for _ in range(num_samples):
        # Simulate an HTTPS session (bursty, large packets)
        seq = []
        for i in range(seq_len):
            # TimeDelta: Exponential distribution (bursty)
            td = np.random.exponential(0.1)
            
            # Size: Bimodal (small ACKs, large Data)
            if np.random.random() > 0.3:
                size = np.random.normal(1200, 200) # Large data packet
            else:
                size = np.random.normal(60, 20)   # Small ACK
            
            # Normalize
            td = min(td, 1.0)
            size = min(max(size, 0), 1500) / 1500.0
            
            seq.append([td, size, 0.0, 0.0]) # Flags/Dir placeholder
        data.append(seq)
    
    return torch.tensor(data, dtype=torch.float32)

def train_network_gan():
    print("[*] Initializing Network GAN Training...")
    
    # Tiny Config for Fast CPU Debugging
    config = {
        "latent_dim": 32,
        "hidden_dim": 64,
        "seq_len": 20,
        "feature_dim": 4,
        "d_model": 32,
        "condition_dim": 32
    }
    
    # Model
    model = NetworkGAN(config)
    device = torch.device("cpu") # Force CPU for stability
    print(f"    Using device: {device}")
    model.to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(model.generator.parameters(), lr=0.001)
    d_optimizer = optim.Adam(model.discriminator.parameters(), lr=0.0005)
    
    # Loss
    criterion = nn.BCELoss()
    
    # Data
    data_path = "data/network_training_data.pt"
    if not Path(data_path).exists():
        print(f"(!) Data not found at {data_path}. Please run scripts/generate_network_dataset.py first.")
        return

    print(f"    Loading training data from {data_path}...")
    real_data = torch.load(data_path).to(device)
    
    # DEBUG: Use subset
    real_data = real_data[:500] 
    print(f"    Using subset of {len(real_data)} samples for debug.")
    
    batch_size = 32
    
    # Training Loop
    epochs = 10
    print(f"    Starting training for {epochs} epochs...")
    print(f"    {'Epoch':<10} | {'D Loss':<10} | {'G Loss':<10}")
    print("-" * 40)
    
    for epoch in range(epochs):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        batches = 0
        
        for i in range(0, len(real_data), batch_size):
            # Batch
            real_batch = real_data[i:i+batch_size]
            curr_batch_size = real_batch.size(0)
            if curr_batch_size < batch_size: continue
            
            # Labels
            real_labels = torch.ones(curr_batch_size, 1).to(device)
            fake_labels = torch.zeros(curr_batch_size, 1).to(device)
            
            # --- Train Discriminator ---
            d_optimizer.zero_grad()
            
            # Real
            real_preds = model.discriminator(real_batch)
            d_loss_real = criterion(real_preds, real_labels)
            
            # Fake
            z = torch.randn(curr_batch_size, config["latent_dim"]).to(device)
            payload = torch.zeros_like(z).to(device) 
            
            # Simulate Host Condition
            condition = torch.randn(curr_batch_size, config["condition_dim"]).to(device)
            
            fake_batch = model.generator(z, payload, condition)
            
            fake_preds = model.discriminator(fake_batch.detach())
            d_loss_fake = criterion(fake_preds, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            d_loss_epoch += d_loss.item()
            
            # --- Train Generator ---
            g_optimizer.zero_grad()
            
            fake_preds = model.discriminator(fake_batch)
            g_loss = criterion(fake_preds, real_labels) 
            
            g_loss.backward()
            g_optimizer.step()
            
            g_loss_epoch += g_loss.item()
            batches += 1
            
        # Log EVERY epoch
        if batches > 0:
            avg_d = d_loss_epoch / batches
            avg_g = g_loss_epoch / batches
            print(f"    {epoch+1:<10} | {avg_d:.4f}     | {avg_g:.4f}")
            
    # Save
    save_path = Path("checkpoints/network_gan.pth")
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] Training Complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_network_gan()
