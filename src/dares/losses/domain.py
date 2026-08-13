"""Shared adversarial domain alignment components.

Used by the ADVENT engine (entropy-map discriminators) and the CyCADA engine
(feature-level alignment).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    """Small CNN classifying whether an input map belongs to the source or
    target domain.

    Maps a ``(B, C, H, W)`` input (e.g. a feature map or an entropy map) to a
    scalar domain logit per sample ``(B,)``: source maps should be classified
    as ``1`` and target maps as ``0``.

    Args:
        in_channels (int): Number of input channels.
        base (int): Base number of channels of the convolutional stack.
        num_layers (int): Number of stride-2 convolution layers.
    """

    def __init__(
        self, in_channels: int, base: int = 32, num_layers: int = 3
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(channels, base, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(base))
            layers.append(nn.ReLU(inplace=True))
            channels = base
        layers.append(nn.Conv2d(base, base, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(base, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes per-sample domain logits.

        Args:
            x (torch.Tensor): Input map of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Domain logits of shape ``(B,)``.
        """
        x = x.float()
        features = self.convs(x)
        pooled = features.mean(dim=(2, 3))
        return self.fc(pooled).squeeze(-1)


def adversarial_loss(
    disc_s: torch.Tensor, disc_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the discriminator and adversarial (feature-extractor) losses.

    Source samples carry domain label ``1`` and target samples label ``0``.

    Args:
        disc_s (torch.Tensor): Source domain logits ``(B,)``.
        disc_t (torch.Tensor): Target domain logits ``(B,)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(loss_discriminator,
            loss_adversarial)``; the adversarial loss pushes the feature
            extractor to make target maps be classified as source.
    """
    ones = torch.ones_like(disc_s)
    zeros = torch.zeros_like(disc_t)
    loss_dis = F.binary_cross_entropy_with_logits(disc_s, ones) + F.binary_cross_entropy_with_logits(
        disc_t, zeros
    )
    loss_adv = F.binary_cross_entropy_with_logits(disc_t, ones)
    return loss_dis, loss_adv
