"""Cycle-consistent adversarial domain adaptation (CyCADA) losses.

Implements the image-level components of CyCADA (Hoffman et al., 2018)
simplified for UDA semantic segmentation: two small pixel generators for
cycle-consistent source <-> target translation, a PatchGAN image
discriminator, and the corresponding cycle / identity / adversarial loss
functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block of two ``3x3 Conv + BatchNorm + ReLU`` layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    The residual shortcut is an identity mapping when ``in_channels ==
    out_channels`` and a 1x1 convolution (with BatchNorm) otherwise, matching
    the ResUNet head's ``ResBlock``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, C_out, H, W)``.
        """
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class PixelGenerator(nn.Module):
    """Compact U-Net translating between the source and target domains.

    Maps ``(B, in_channels, H, W) -> (B, in_channels, H, W)``. The encoder
    downsamples to ``[base, 2*base, 4*base]`` channels with stride-2
    convolutions (BN + ReLU); the decoder bilinearly upsamples back to
    ``base``; a final 3x3 convolution (no activation) reconstructs the input
    channels.

    Args:
        in_channels (int): Number of input / output channels (default 4).
        base (int): Base channel count of the encoder (default 32).
    """

    def __init__(self, in_channels: int = 4, base: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base = base

        self.encoder = nn.ModuleList()
        channels = in_channels
        for out_channels in (base, 2 * base, 4 * base):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            channels = out_channels

        self.decoder = nn.ModuleList()
        for out_channels in (2 * base, base, base):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            channels = out_channels

        self.final_conv = nn.Conv2d(base, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Translates an image across domains.

        Args:
            x (torch.Tensor): Input image of shape ``(B, in_channels, H, W)``
                with ``H`` and ``W`` multiples of 8.

        Returns:
            torch.Tensor: Translated image of shape
                ``(B, in_channels, H, W)``.
        """
        for down in self.encoder:
            x = down(x)
        for up in self.decoder:
            x = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
            x = up(x)
        return self.final_conv(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator producing a patch-wise realism map.

    Applies three stride-2 conv blocks (``Conv4x4 + BN + LeakyReLU(0.2)``) at
    ``base -> 2*base -> 4*base`` channels followed by a final ``Conv4x4`` to a
    single-channel patch logits map of shape ``(B, 1, H/8, W/8)``.

    Args:
        in_channels (int): Number of input channels (default 4).
        base (int): Base number of channels (default 32).
    """

    def __init__(self, in_channels: int = 4, base: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base = base

        layers: list[nn.Module] = []
        channels = in_channels
        for out_channels in (base, 2 * base, 4 * base):
            layers.append(
                nn.Conv2d(
                    channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = out_channels
        layers.append(nn.Conv2d(4 * base, 1, kernel_size=4, padding=1))
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the patch logits for an image.

        Args:
            x (torch.Tensor): Input image of shape ``(B, C, H, W)`` with
                ``H`` and ``W`` multiples of 8.

        Returns:
            torch.Tensor: Patch logits of shape ``(B, 1, H/8, W/8)``.
        """
        return self.convs(x.float())


def cycle_consistency_loss(
    g_st: PixelGenerator,
    g_ts: PixelGenerator,
    x_s: torch.Tensor,
    x_t: torch.Tensor,
) -> torch.Tensor:
    """L1 cycle-consistency loss for the two pixel generators.

    Args:
        g_st (PixelGenerator): Source -> target generator.
        g_ts (PixelGenerator): Target -> source generator.
        x_s (torch.Tensor): Source image of shape ``(B, C, H, W)``.
        x_t (torch.Tensor): Target image of shape ``(B, C, H, W)``.

    Returns:
        torch.Tensor: Scalar ``L1(g_ts(g_st(x_s)), x_s) +
            L1(g_st(g_ts(x_t)), x_t)``.
    """
    return F.l1_loss(g_ts(g_st(x_s)), x_s) + F.l1_loss(g_st(g_ts(x_t)), x_t)


def identity_loss(
    g_st: PixelGenerator,
    g_ts: PixelGenerator,
    x_s: torch.Tensor,
    x_t: torch.Tensor,
) -> torch.Tensor:
    """L1 identity-mapping loss encouraging domain-invariant translation.

    Args:
        g_st (PixelGenerator): Source -> target generator.
        g_ts (PixelGenerator): Target -> source generator.
        x_s (torch.Tensor): Source image of shape ``(B, C, H, W)``.
        x_t (torch.Tensor): Target image of shape ``(B, C, H, W)``.

    Returns:
        torch.Tensor: Scalar ``L1(g_st(x_t), x_t) + L1(g_ts(x_s), x_s)``.
    """
    return F.l1_loss(g_st(x_t), x_t) + F.l1_loss(g_ts(x_s), x_s)


def patch_adversarial_loss(d_fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator-side PatchGAN adversarial loss.

    Penalizes the generator for producing images that the discriminator does
    not classify as real (target label ``1`` for every patch).

    Args:
        d_fake_logits (torch.Tensor): Patch logits of fake images, shape
            ``(B, 1, Ph, Pw)``.

    Returns:
        torch.Tensor: Scalar ``BCE_with_logits(d_fake_logits, ones)``.
    """
    ones = torch.ones_like(d_fake_logits)
    return F.binary_cross_entropy_with_logits(d_fake_logits, ones)


def patch_discriminator_loss(
    d_real: torch.Tensor, d_fake: torch.Tensor
) -> torch.Tensor:
    """Discriminator-side PatchGAN loss.

    Real images carry label ``1`` and fake images label ``0``, per patch.

    Args:
        d_real (torch.Tensor): Patch logits of real images, shape
            ``(B, 1, Ph, Pw)``.
        d_fake (torch.Tensor): Patch logits of fake images, shape
            ``(B, 1, Ph, Pw)``.

    Returns:
        torch.Tensor: Scalar ``BCE_with_logits(d_real, ones) +
            BCE_with_logits(d_fake, zeros)``.
    """
    ones = torch.ones_like(d_real)
    zeros = torch.zeros_like(d_fake)
    return F.binary_cross_entropy_with_logits(
        d_real, ones
    ) + F.binary_cross_entropy_with_logits(d_fake, zeros)
