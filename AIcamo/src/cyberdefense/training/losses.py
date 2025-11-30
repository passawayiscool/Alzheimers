"""Loss functions for VAE-GAN training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VAEGANLoss(nn.Module):
    """
    Combined loss for VAE-GAN training.

    Includes:
    - Reconstruction loss (cross-entropy)
    - KL divergence loss (VAE regularization)
    - Adversarial loss (GAN)
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.1,
        adversarial_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        """
        Initialize loss function.

        Args:
            reconstruction_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence
            adversarial_weight: Weight for adversarial loss
            label_smoothing: Label smoothing for discriminator
        """
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.adversarial_weight = adversarial_weight
        self.label_smoothing = label_smoothing

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence loss for VAE.

        KL(q(z|x) || p(z)) where p(z) is standard normal
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def reconstruction_loss(
        self, logits: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Reconstruction loss (cross-entropy).

        Args:
            logits: Predicted logits [batch, seq_len, vocab_size]
            target_ids: Target token IDs [batch, seq_len, max_event_tokens]
            attention_mask: Attention mask

        Returns:
            Reconstruction loss
        """
        # Flatten target to get single token per event (use first token)
        target = target_ids[:, :, 0]  # [batch, seq_len]

        # Reshape logits and target for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)

        # Compute cross-entropy
        loss = self.ce_loss(logits_flat, target_flat)

        return loss

    def adversarial_loss_discriminator(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Adversarial loss for discriminator.

        Maximize log(D(real)) + log(1 - D(fake))
        """
        batch_size = real_logits.size(0)

        # Real labels (with label smoothing)
        real_labels = torch.ones_like(real_logits) * (1.0 - self.label_smoothing)
        fake_labels = torch.zeros_like(fake_logits) + self.label_smoothing

        # BCE loss
        real_loss = self.bce_loss(real_logits, real_labels)
        fake_loss = self.bce_loss(fake_logits, fake_labels)

        return (real_loss + fake_loss) / 2

    def adversarial_loss_generator(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Adversarial loss for generator.

        Maximize log(D(fake)) = Minimize log(1 - D(fake))
        """
        # Generator wants discriminator to classify fakes as real
        real_labels = torch.ones_like(fake_logits)
        loss = self.bce_loss(fake_logits, real_labels)

        return loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        mode: str = "generator",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            outputs: Model outputs dictionary
            target_ids: Target token IDs
            attention_mask: Attention mask
            mode: 'generator' or 'discriminator'

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}

        if mode == "generator":
            # Reconstruction loss
            recon_loss = self.reconstruction_loss(
                outputs["generated_logits"], target_ids, attention_mask
            )
            losses["reconstruction"] = recon_loss

            # KL divergence
            kl_loss = self.kl_divergence(outputs["mu"], outputs["logvar"])
            losses["kl_divergence"] = kl_loss

            # Adversarial loss (generator)
            adv_loss = self.adversarial_loss_generator(outputs["fake_logits"])
            losses["adversarial"] = adv_loss

            # Total generator loss
            total_loss = (
                self.reconstruction_weight * recon_loss
                + self.kl_weight * kl_loss
                + self.adversarial_weight * adv_loss
            )

        else:  # discriminator mode
            # Adversarial loss (discriminator)
            adv_loss = self.adversarial_loss_discriminator(
                outputs["real_logits"], outputs["fake_logits"]
            )
            losses["adversarial"] = adv_loss

            total_loss = adv_loss

        losses["total"] = total_loss

        return losses
