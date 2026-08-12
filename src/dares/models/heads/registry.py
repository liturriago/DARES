"""Head factory: builds a segmentation head from its name."""

import torch.nn as nn

from dares.config import ModelConfig
from dares.models.heads.deeplabv3p import DeepLabV3PHead
from dares.models.heads.resunet import ResUNetHead


def build_head(
    name: str,
    backbone_out_channels: dict[str, int],
    config: ModelConfig,
) -> nn.Module:
    """Builds a segmentation head by name.

    Args:
        name (str): Head name, one of ``"resunet"`` or ``"deeplabv3p"``.
        backbone_out_channels (dict[str, int]): Number of output channels per
            backbone feature level (keys ``"S2"`` ... ``"S32"``).
        config (ModelConfig): Model configuration for the head.

    Returns:
        nn.Module: Instantiated head module.

    Raises:
        ValueError: If ``name`` is not a registered head.
    """
    if name == "resunet":
        return ResUNetHead(backbone_out_channels, config)
    if name == "deeplabv3p":
        return DeepLabV3PHead(backbone_out_channels, config)
    raise ValueError(f"Unknown head name: {name!r}")
