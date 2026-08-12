"""Tests for the DARES segmentation heads (decoders)."""

import sys
from collections import OrderedDict
from types import ModuleType

import pytest
import torch

from dares.config import ModelConfig

# ``dares/models/__init__.py`` imports ``dares.models.segmentation`` (a
# separately developed module). Stub it so importing ``dares.models.heads``
# works while that module is not yet implemented.
_SEG_STUB = ModuleType("dares.models.segmentation")
_SEG_STUB.SegmentationModel = None
_SEG_STUB.build_model = None
sys.modules.setdefault("dares.models.segmentation", _SEG_STUB)

from dares.models.heads.registry import build_head

RESNET50_CHANNELS = {"S2": 64, "S4": 256, "S8": 512, "S16": 1024, "S32": 2048}
CONVNEXT_CHANNELS = {"S2": 96, "S4": 96, "S8": 192, "S16": 384, "S32": 768}

STRIDES = {"S2": 2, "S4": 4, "S8": 8, "S16": 16, "S32": 32}

HEAD_DIMS = {"resunet": 32, "deeplabv3p": 256}


def make_fake_features(
    backbone_out_channels: dict[str, int], h: int, w: int
) -> "OrderedDict[str, torch.Tensor]":
    """Builds a fake multi-scale backbone feature dict.

    Args:
        backbone_out_channels (dict[str, int]): Channels per level
            (keys ``"S2"`` ... ``"S32"``).
        h (int): Input image height.
        w (int): Input image width.

    Returns:
        OrderedDict[str, torch.Tensor]: Random feature maps at strides 2..32.
    """
    return OrderedDict(
        (level, torch.randn(1, c, h // STRIDES[level], w // STRIDES[level]))
        for level, c in backbone_out_channels.items()
    )


@pytest.mark.parametrize("head_name", ["resunet", "deeplabv3p"])
@pytest.mark.parametrize(
    "backbone_out_channels",
    [RESNET50_CHANNELS, CONVNEXT_CHANNELS],
    ids=["resnet50", "convnext"],
)
@pytest.mark.parametrize("hw", [128, 224])
def test_head_output_shapes(head_name, backbone_out_channels, hw):
    """Head outputs match the full input resolution for fake features."""
    num_classes = 2
    config = ModelConfig(head=head_name, num_classes=num_classes, pretrained=False)
    head = build_head(head_name, backbone_out_channels, config)
    head.eval()

    features = make_fake_features(backbone_out_channels, hw, hw)
    features_fullres, logits = head(features)

    dim = HEAD_DIMS[head_name]
    assert features_fullres.shape == (1, dim, hw, hw)
    assert logits.shape == (1, num_classes, hw, hw)
    assert logits.shape[-2:] == features_fullres.shape[-2:]


def test_resunet_invalid_channel_plan():
    """ResUNetHead rejects a channel plan that is not length 5."""
    config = ModelConfig(head="resunet", resunet_channels=[64, 32], pretrained=False)
    with pytest.raises(ValueError):
        build_head("resunet", RESNET50_CHANNELS, config)


def test_build_head_unknown_name():
    """build_head raises ValueError for an unregistered head name."""
    config = ModelConfig(head="resunet", pretrained=False)
    with pytest.raises(ValueError):
        build_head("does_not_exist", RESNET50_CHANNELS, config)


def test_heads_integration_real_backbones():
    """Heads reconstruct full-resolution outputs from real DARES backbones."""
    backbone_registry = pytest.importorskip("dares.models.backbones.registry")
    build_backbone = backbone_registry.build_backbone

    x = torch.randn(1, 4, 128, 128)
    for backbone_name in ["resnet50", "convnext_tiny"]:
        for head_name in ["resunet", "deeplabv3p"]:
            config = ModelConfig(
                backbone=backbone_name, head=head_name, pretrained=False
            )
            backbone = build_backbone(backbone_name, config)
            backbone.eval()
            features = backbone(x)
            head = build_head(head_name, backbone.out_channels, config)
            head.eval()
            features_fullres, logits = head(features)
            assert features_fullres.shape == (1, HEAD_DIMS[head_name], 128, 128)
            assert logits.shape == (1, config.num_classes, 128, 128)
