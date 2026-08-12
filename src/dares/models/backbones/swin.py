"""Swin Transformer backbone encoders for DARES."""

from collections import OrderedDict
from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision.models import (  # type: ignore[import-untyped]
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    swin_b,
    swin_s,
    swin_t,
)

from .base import Encoder, _make_stride2_stem, adapt_first_conv

_STAGE_INDICES = {"S4": 0, "S8": 3, "S16": 5, "S32": 7}

_NATIVE_WIDTHS: dict[str, list[int]] = {
    "swin_t": [96, 192, 384, 768],
    "swin_s": [96, 192, 384, 768],
    "swin_b": [128, 256, 512, 1024],
}


class SwinEncoder(Encoder):
    """Swin Transformer encoder producing feature maps at strides 2 to 32.

    The native ``features`` Sequential is ``[patch_embed, stage0_blocks,
    PatchMerging, stage1_blocks, PatchMerging, stage2_blocks, ...]``. ``S4`` is
    the patch-embedding output, while ``S8`` / ``S16`` / ``S32`` are taken after
    each block stage. Internal stages work in ``(B, H, W, C)`` layout, so each
    captured feature map is permuted back to ``(B, C, H, W)``. An extra parallel
    stem produces the ``S2`` level at the embedding width. The native path
    receives the raw input unchanged.

    Args:
        size (Literal): Swin variant, one of ``"swin_t"``, ``"swin_s"`` or
            ``"swin_b"``.
        in_channels (int): Number of input channels.
        pretrained (bool): Whether to load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        size: Literal["swin_t", "swin_s", "swin_b"],
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
                "swin_t": Swin_T_Weights.IMAGENET1K_V1,
                "swin_s": Swin_S_Weights.IMAGENET1K_V1,
                "swin_b": Swin_B_Weights.IMAGENET1K_V1,
            }[size]
            if pretrained
            else None
        )
        if size == "swin_t":
            model = swin_t(weights=weights)
        elif size == "swin_s":
            model = swin_s(weights=weights)
        else:
            model = swin_b(weights=weights)
        self.model = model
        patch_embed = self.model.features[0]
        conv = next(
            module for module in patch_embed.modules() if isinstance(module, nn.Conv2d)
        )
        patch_embed[0] = adapt_first_conv(conv, in_channels)
        self.s2_stem = _make_stride2_stem(in_channels, widths[0], activation="gelu")

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict(
            [("S2", self.s2_stem(x).contiguous())]
        )
        start = 0
        for key, end in sorted(_STAGE_INDICES.items(), key=lambda kv: kv[1]):
            for module in self.model.features[start : end + 1]:
                x = module(x)
            out[key] = x.permute(0, 3, 1, 2).contiguous()
            start = end + 1
        return out
