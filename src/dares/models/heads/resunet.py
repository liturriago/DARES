"""ResUNet segmentation head (U-Net style decoder with residual blocks)."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dares.config import ModelConfig

_LEVELS: list[str] = ["S32", "S16", "S8", "S4", "S2"]


class ResBlock(nn.Module):
    """Residual block of two ``3x3 Conv + BatchNorm + ReLU`` layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    The residual shortcut is an identity mapping when ``in_channels ==
    out_channels`` and a 1x1 convolution (with BatchNorm) otherwise.
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
            x (torch.Tensor): Input tensor ``(B, C_in, H, W)``.

        Returns:
            torch.Tensor: Output tensor ``(B, C_out, H, W)``.
        """
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResUNetHead(nn.Module):
    """U-Net style decoder with residual blocks and skip connections.

    Consumes the backbone multi-scale feature dict and reconstructs a
    full-resolution dense feature map plus pixel-wise logits.

    Args:
        backbone_out_channels (dict[str, int]): Number of output channels per
            backbone feature level (keys ``"S2"`` ... ``"S32"``).
        config (ModelConfig): Model configuration providing ``num_classes``,
            ``dropout_rate`` and the ``resunet_channels`` channel plan.

    Raises:
        ValueError: If ``config.resunet_channels`` does not contain exactly 5
            levels (deepest S32 to shallowest S2).

    Attributes:
        feature_dim (int): Dimension of the full-resolution feature map,
            equal to ``plan[-1]`` (default 32).

    The channel plan is ordered from the deepest level (S32) to the shallowest
    (S2). ``features_fullres`` has embedding dimension ``D = plan[-1]``.
    """

    def __init__(
        self,
        backbone_out_channels: dict[str, int],
        config: ModelConfig,
    ) -> None:
        super().__init__()
        plan = config.resunet_channels
        if len(plan) != 5:
            raise ValueError(
                f"resunet_channels must contain exactly 5 levels, got {len(plan)}"
            )

        self._plan = list(plan)
        self.feature_dim = self._plan[-1]
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate

        self.skips = nn.ModuleDict()
        for i, level in enumerate(_LEVELS):
            self.skips[level] = nn.Sequential(
                nn.Conv2d(
                    backbone_out_channels[level],
                    plan[i],
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(plan[i]),
            )

        self.blocks = nn.ModuleDict()
        self.blocks["S32"] = ResBlock(plan[0], plan[0])
        for i in range(1, 5):
            level = _LEVELS[i]
            self.blocks[level] = ResBlock(plan[i - 1] + plan[i], plan[i])

        self.final_conv = nn.Conv2d(
            plan[4], plan[4], kernel_size=3, padding=1, bias=False
        )

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout2d(self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.classifier = nn.Conv2d(plan[4], self.num_classes, kernel_size=1)

    def forward(
        self, features: "OrderedDict[str, torch.Tensor]"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes the multi-scale features into full-resolution maps.

        Args:
            features (OrderedDict[str, torch.Tensor]): Backbone feature maps at
                strides 2..32 (keys ``"S2"`` ... ``"S32"``).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(features_fullres, logits)``
                both of shape ``(B, D, H, W)`` and ``(B, C, H, W)`` at the full
                input resolution.
        """
        x = self.blocks["S32"](self.skips["S32"](features["S32"]))
        for i in range(1, 5):
            level = _LEVELS[i]
            skip = self.skips[level](features[level])
            x = self._upsample(x, skip.shape[-2:])
            x = self.blocks[level](torch.cat([x, skip], dim=1))

        full_size = torch.Size([features["S2"].shape[-2] * 2, features["S2"].shape[-1] * 2])
        features_fullres = self.final_conv(self._upsample(x, full_size))
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
