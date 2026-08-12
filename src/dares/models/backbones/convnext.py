"""ConvNeXt backbone encoders for DARES."""

from collections import OrderedDict
from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision.models import (  # type: ignore[import-untyped]
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_large,
    convnext_tiny,
)

from .base import Encoder, _make_stride2_stem, adapt_first_conv

_STAGE_INDICES = {"S4": 0, "S8": 3, "S16": 5, "S32": 7}

_NATIVE_WIDTHS: dict[str, list[int]] = {
    "convnext_tiny": [96, 192, 384, 768],
    "convnext_base": [128, 256, 512, 1024],
    "convnext_large": [192, 384, 768, 1536],
}


class ConvNeXtEncoder(Encoder):
    """ConvNeXt encoder producing feature maps at strides 2 to 32.

    The native ``features`` Sequential is ``[stem, stage0_blocks, downsample,
    stage1_blocks, downsample, stage2_blocks, downsample, stage3_blocks]``.
    ``S4`` is the stem output, while ``S8`` / ``S16`` / ``S32`` are taken after
    each following downsample + block stage. An extra parallel stem produces the
    ``S2`` level at the native stem width.

    Args:
        size (Literal): ConvNeXt variant, one of ``"convnext_tiny"``,
            ``"convnext_base"`` or ``"convnext_large"``.
        in_channels (int): Number of input channels.
        pretrained (bool): Whether to load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        size: Literal["convnext_tiny", "convnext_base", "convnext_large"],
        in_channels: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.strides = {"S2": 2, "S4": 4, "S8": 8, "S16": 16, "S32": 32}
        widths = _NATIVE_WIDTHS[size]
        self.out_channels = {
            "S2": widths[0],
            "S4": widths[0],
            "S8": widths[1],
            "S16": widths[2],
            "S32": widths[3],
        }
        weights: Any = (
            {
                "convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                "convnext_base": ConvNeXt_Base_Weights.IMAGENET1K_V1,
                "convnext_large": ConvNeXt_Large_Weights.IMAGENET1K_V1,
            }[size]
            if pretrained
            else None
        )
        if size == "convnext_tiny":
            model = convnext_tiny(weights=weights)
        elif size == "convnext_base":
            model = convnext_base(weights=weights)
        else:
            model = convnext_large(weights=weights)
        self.model = model
        stem_conv = self._find_stem_conv()
        self.model.features[0][0] = adapt_first_conv(stem_conv, in_channels)
        self.s2_stem = _make_stride2_stem(in_channels, widths[0], activation="gelu")

    def _find_stem_conv(self) -> nn.Conv2d:
        for module in self.model.features[0].modules():
            if isinstance(module, nn.Conv2d):
                return module
        raise RuntimeError("No Conv2d found in the ConvNeXt stem")

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict(
            [("S2", self.s2_stem(x).contiguous())]
        )
        start = 0
        for key, end in sorted(_STAGE_INDICES.items(), key=lambda kv: kv[1]):
            for module in self.model.features[start : end + 1]:
                x = module(x)
            out[key] = x.contiguous()
            start = end + 1
        return out
