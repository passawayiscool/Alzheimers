"""VAE-GAN Trainer"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import json

from ..models.vae_gan import VAEGANModel
from .losses import VAEGANLoss

logger = logging.getLogger(__name__)

class VAEGANTrainer:
    """
    Trainer for VAE-GAN model.
    
    Handles:
    - Alternating training (Discriminator vs Generator)
    - Loss calculation
    - Checkpointing
    - Logging
    """
    def __init__(
        self,
        model: VAEGANModel,
        config: Dict[str, Any],
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training params
        train_config = config.get("training", {})
        self.lr = train_config.get("learning_rate", 2e-4)
        self.clip_norm = train_config.get("gradient_clip_val", 1.0)
        
        # Optimizers
        # Separate optimizers for Generator (VAE parts) and Discriminator
        self.opt_g = optim.Adam(
            list(model.encoder_backbone.parameters()) + 
            list(model.fc_mu.parameters()) + 
            list(model.fc_logvar.parameters()) + 
            list(model.decoder_input.parameters()) + 
            list(model.decoder_backbone.parameters()) +
            list(model.embedding.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999)
        )
        
        self.opt_d = optim.Adam(
            model.discriminator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999)
        )
        
        # Loss function
        self.criterion = VAEGANLoss(
            reconstruction_weight=train_config.get("reconstruction_weight", 1.0),
            kl_weight=train_config.get("kl_weight", 0.1),
            adversarial_weight=train_config.get("adversarial_weight", 0.5)
        )
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {"g_loss": 0.0, "d_loss": 0.0, "recon": 0.0, "kl": 0.0, "adv": 0.0}
        
        for batch_idx, batch in enumerate(dataloader):
            # batch is Dict with 'input_ids', 'attention_mask'
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            # ====================================================
            # 1. Train Discriminator
            # ====================================================
            self.opt_d.zero_grad()
            
            # Real data
            # Embed real data for discriminator
            with torch.no_grad():
                real_embeddings = self.model.embedding(input_ids, attention_mask)
                
            # Fake data
            # Generate fake data from random latent
            batch_size = input_ids.size(0)
            z = torch.randn(batch_size, self.model.latent_dim).to(self.device)
            
            # Decode z to get "fake embeddings"
            # We need the decoder to output embeddings or something similar
            # The decoder outputs logits [batch, seq_len, vocab_size]
            # We can use Gumbel-Softmax to get differentiable approximation of tokens
            # Or just pass the decoder hidden states?
            # For simplicity in this implementation, we'll forward pass the VAE
            # and use the reconstructed output as "fake" for the discriminator
            
            # Forward pass VAE
            outputs = self.model(input_ids, attention_mask)
            
            # Get logits
            fake_logits_seq = outputs["generated_logits"] # [batch, seq_len, vocab]
            
            # Convert logits to embeddings (soft approximation)
            # Softmax over vocab -> [batch, seq_len, vocab]
            probs = F.softmax(fake_logits_seq, dim=-1)
            # Matmul with embedding matrix -> [batch, seq_len, embed_dim]
            fake_embeddings = torch.matmul(probs, self.model.embedding.token_embedding.weight)
            
            # Discriminator forward
            d_real = self.model.discriminator(real_embeddings)
            d_fake = self.model.discriminator(fake_embeddings.detach())
            
            # Discriminator loss
            d_loss = self.criterion.adversarial_loss_discriminator(d_real, d_fake)
            d_loss.backward()
            self.opt_d.step()
            
            # ====================================================
            # 2. Train Generator
            # ====================================================
            self.opt_g.zero_grad()
            
            # Re-compute d_fake for generator update (so gradients flow)
            d_fake_for_g = self.model.discriminator(fake_embeddings)
            
            # Prepare outputs for loss
            loss_outputs = {
                "generated_logits": outputs["generated_logits"],
                "mu": outputs["mu"],
                "logvar": outputs["logvar"],
                "fake_logits": d_fake_for_g
            }
            
            # Calculate losses
            losses = self.criterion(
                loss_outputs, 
                target_ids=input_ids, # Target is input (autoencoder)
                attention_mask=attention_mask
            )
            
            g_loss = losses["total"]
            g_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            
            self.opt_g.step()
            
            # Update metrics
            total_losses["g_loss"] += g_loss.item()
            total_losses["d_loss"] += d_loss.item()
            total_losses["recon"] += losses["reconstruction"].item()
            total_losses["kl"] += losses["kl_divergence"].item()
            total_losses["adv"] += losses["adversarial"].item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"G: {g_loss.item():.4f} D: {d_loss.item():.4f}"
                )
                
        # Average losses
        avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
        return avg_losses
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "opt_g_state_dict": self.opt_g.state_dict(),
            "opt_d_state_dict": self.opt_d.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")
