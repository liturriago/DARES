"""Tests for the DARES backbone encoders."""

import torch

from dares.config import ModelConfig
from dares.models.backbones.base import adapt_first_conv
from dares.models.backbones.registry import build_backbone

BACKBONES = [
    "resnet50",
    "convnext_tiny",
    "convnext_base",
    "convnext_large",
    "swin_t",
    "swin_s",
    "swin_b",
]

LEVELS = ["S2", "S4", "S8", "S16", "S32"]


def test_adapt_first_conv_to_four_channels():
    """adapt_first_conv widens a 3-channel conv to 4 input channels."""
    source = torch.nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
    source.weight.data.normal_()
    adapted = adapt_first_conv(source, 4)
    assert isinstance(adapted, torch.nn.Conv2d)
    assert adapted.weight.shape == (16, 4, 7, 7)
    assert adapted.weight[:, :3].shape == (16, 3, 7, 7)
    assert adapted.weight[:, 3].shape == (16, 7, 7)
    assert torch.allclose(adapted.weight[:, 3], source.weight.mean(dim=1))
    assert adapted.bias is not None
    assert torch.allclose(adapted.bias, source.bias)
    assert adapted.stride == source.stride
    assert adapted.padding == source.padding


def test_adapt_first_conv_returns_unchanged_for_matching_channels():
    """adapt_first_conv returns the original conv when channels already match."""
    source = torch.nn.Conv2d(4, 16, kernel_size=3)
    assert adapt_first_conv(source, 4) is source


def test_build_backbone_unknown_name():
    """build_backbone raises ValueError for an unregistered backbone name."""
    config = ModelConfig(backbone="resnet50", pretrained=False)
    try:
        build_backbone("does_not_exist", config)
    except ValueError:
        return
    raise AssertionError("build_backbone should have raised ValueError")


def _check_encoder(backbone_name: str, hw: int):
    config = ModelConfig(backbone=backbone_name, in_channels=4, pretrained=False)
    encoder = build_backbone(backbone_name, config)
    assert isinstance(encoder.out_channels, dict)
    assert set(encoder.out_channels) == set(LEVELS)
    assert len(encoder.strides) == 5
    assert isinstance(encoder.pretrained, bool)

    x = torch.randn(1, 4, hw, hw)
    with torch.no_grad():
        features = encoder(x)
    assert list(features) == LEVELS
    for level in LEVELS:
        feature = features[level]
        assert feature.shape == (
            1,
            encoder.out_channels[level],
            hw // encoder.strides[level],
            hw // encoder.strides[level],
        )
        assert feature.is_contiguous()


def test_all_backbones_128():
    """All backbones emit the right feature shapes for a 128x128 input."""
    for backbone_name in BACKBONES:
        _check_encoder(backbone_name, 128)


def test_all_backbones_64():
    """All backbones emit the right feature shapes for a 64x64 input."""
    for backbone_name in BACKBONES:
        _check_encoder(backbone_name, 64)
