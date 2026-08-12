"""Backbone factory: builds an encoder from its name."""

from typing import Any, cast

import torch.nn as nn

from dares.config import ModelConfig

from .convnext import ConvNeXtEncoder
from .resnet import ResNet50Encoder
from .swin import SwinEncoder


def build_backbone(name: str, config: ModelConfig) -> nn.Module:
    """Builds a multi-scale backbone encoder by name.

    Args:
        name (str): Backbone name, one of ``"resnet50"``, ``"convnext_tiny"``,
            ``"convnext_base"``, ``"convnext_large"``, ``"swin_t"``,
            ``"swin_s"`` or ``"swin_b"``.
        config (ModelConfig): Model configuration holding ``in_channels`` and
            ``pretrained`` settings.

    Returns:
        nn.Module: Instantiated encoder module returning an ``OrderedDict`` of
            feature maps keyed ``"S2"`` ... ``"S32"``.

    Raises:
        ValueError: If ``name`` is not a registered backbone.
    """
    if name == "resnet50":
        return ResNet50Encoder(config.in_channels, config.pretrained)
    if name in ("convnext_tiny", "convnext_base", "convnext_large"):
        return ConvNeXtEncoder(cast(Any, name), config.in_channels, config.pretrained)
    if name in ("swin_t", "swin_s", "swin_b"):
        return SwinEncoder(cast(Any, name), config.in_channels, config.pretrained)
    raise ValueError(f"Unknown backbone name: {name!r}")
