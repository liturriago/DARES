"""DeepLabV3+ segmentation head (ASPP-based decoder with low-level features)."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dares.config import ModelConfig


class _ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU building block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size (1 or 3).
        dilation (int): Dilation of the convolution (default 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the conv-bn-relu block.

        Args:
            x (torch.Tensor): Input tensor ``(B, C_in, H, W)``.

        Returns:
            torch.Tensor: Output tensor ``(B, C_out, H, W)``.
        """
        return self.relu(self.bn(self.conv(x)))


class DeepLabV3PHead(nn.Module):
    """DeepLabV3+ style decoder with Atrous Spatial Pyramid Pooling.

    Consumes ``S16`` (ASPP input) and ``S4`` (low-level features) from the
    backbone feature dict and reconstructs a full-resolution dense feature map
    plus pixel-wise logits.

    Args:
        backbone_out_channels (dict[str, int]): Number of output channels per
            backbone feature level (keys ``"S2"`` ... ``"S32"``).
        config (ModelConfig): Model configuration providing ``num_classes``,
            ``dropout_rate``, ``deeplab_aspp_channels`` and
            ``deeplab_low_level_channels``.

    Attributes:
        feature_dim (int): Dimension of the full-resolution feature map,
            equal to ``deeplab_aspp_channels`` (default 256).

    ``features_fullres`` has embedding dimension ``D = aspp_ch`` (default 256).
    """

    def __init__(
        self,
        backbone_out_channels: dict[str, int],
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate
        aspp_ch = config.deeplab_aspp_channels
        self.feature_dim = aspp_ch
        low_ch = config.deeplab_low_level_channels
        s16_ch = backbone_out_channels["S16"]
        s4_ch = backbone_out_channels["S4"]

        self.aspp = nn.ModuleList()
        self.aspp.append(_ConvBNReLU(s16_ch, aspp_ch, kernel_size=1))
        for dilation in [6, 12, 18]:
            self.aspp.append(
                _ConvBNReLU(s16_ch, aspp_ch, kernel_size=3, dilation=dilation)
            )
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(s16_ch, aspp_ch, kernel_size=1),
        )
        self.aspp_proj = _ConvBNReLU(5 * aspp_ch, aspp_ch, kernel_size=1)

        self.low_proj = _ConvBNReLU(s4_ch, low_ch, kernel_size=1)

        self.decoder1 = _ConvBNReLU(aspp_ch + low_ch, aspp_ch, kernel_size=3)
        self.decoder2 = _ConvBNReLU(aspp_ch, aspp_ch, kernel_size=3)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout2d(self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.classifier = nn.Conv2d(aspp_ch, self.num_classes, kernel_size=1)

    def forward(
        self, features: "OrderedDict[str, torch.Tensor]"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes the multi-scale features into full-resolution maps.

        Args:
            features (OrderedDict[str, torch.Tensor]): Backbone feature maps;
                only ``S16`` and ``S4`` are consumed.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(features_fullres, logits)``
                both of shape ``(B, D, H, W)`` and ``(B, C, H, W)`` at the full
                input resolution.
        """
        x16 = features["S16"]
        aspp_branches = [branch(x16) for branch in self.aspp]
        pooled = self.aspp_pool(x16)
        pooled = F.interpolate(
            pooled,
            size=x16.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        aspp_out = self.aspp_proj(torch.cat([*aspp_branches, pooled], dim=1))

        low = self.low_proj(features["S4"])
        low_size = low.shape[-2:]
        x = self._upsample(aspp_out, low_size)
        x = self.decoder1(torch.cat([x, low], dim=1))
        x = self.decoder2(x)

        full_size = torch.Size([low.shape[-2] * 4, low.shape[-1] * 4])
        features_fullres = self._upsample(x, full_size)
        logits = self.classifier(self.dropout(features_fullres))
        return features_fullres, logits

    @staticmethod
    def _upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Bilinearly upsamples ``x`` to ``size`` (no-op if already matching).

        Args:
            x (torch.Tensor): Tensor to upsample ``(B, C, H, W)``.
            size (tuple[int, int]): Target spatial size ``(H, W)``.

        Returns:
            torch.Tensor: Upsampled tensor of shape ``(B, C, H', W')``.
        """
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
