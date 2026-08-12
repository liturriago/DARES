"""ResNet-50 backbone encoder for DARES."""

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50  # type: ignore[import-untyped]

from .base import Encoder, adapt_first_conv


class ResNet50Encoder(Encoder):
    """ResNet-50 encoder producing feature maps at strides 2 to 32.

    The native stem (``conv1`` + ``bn1`` + ``relu``) yields the ``S2`` level,
    ``maxpool`` + ``layer1`` the ``S4`` level, and ``layer2`` / ``layer3`` /
    ``layer4`` the ``S8`` / ``S16`` / ``S32`` levels. The first convolution is
    adapted to the configured number of input channels.

    Args:
        in_channels (int): Number of input channels.
        pretrained (bool): Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, in_channels: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.strides = {"S2": 2, "S4": 4, "S8": 8, "S16": 16, "S32": 32}
        self.out_channels = {
            "S2": 64,
            "S4": 256,
            "S8": 512,
            "S16": 1024,
            "S32": 2048,
        }
        weights: Any = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet50(weights=weights)
        self.model.conv1 = adapt_first_conv(self.model.conv1, in_channels)

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict([("S2", x)])
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        out["S4"] = x
        x = self.model.layer2(x)
        out["S8"] = x
        x = self.model.layer3(x)
        out["S16"] = x
        x = self.model.layer4(x)
        out["S32"] = x
        return out
