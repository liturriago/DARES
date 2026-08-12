"""Integration tests for the SegmentationModel wrapper and the model factory."""

import pytest
import torch
import torch.nn as nn

from dares.config import ModelConfig
from dares.models import SegmentationModel, build_model
from dares.models.heads.registry import build_head

BACKBONES = [
    "resnet50",
    "convnext_tiny",
    "convnext_base",
    "convnext_large",
    "swin_t",
    "swin_s",
    "swin_b",
]
HEADS = ["resunet", "deeplabv3p"]

RESNET50_CHANNELS = {"S2": 64, "S4": 256, "S8": 512, "S16": 1024, "S32": 2048}


def _make_config(backbone: str, head: str) -> ModelConfig:
    return ModelConfig(
        backbone=backbone,
        head=head,
        in_channels=4,
        num_classes=2,
        pretrained=False,
    )


@pytest.mark.parametrize("head_name", HEADS)
@pytest.mark.parametrize("backbone_name", BACKBONES)
def test_model_forward_modes(backbone_name, head_name):
    """All backbone x head combos emit correct shapes for class/feature/both."""
    model = build_model(_make_config(backbone_name, head_name))
    model.eval()
    x = torch.randn(2, 4, 128, 128)

    with torch.no_grad():
        logits = model(x)
        features = model(x, mode="feature")
        both = model(x, mode="both")

    assert logits.shape == (2, 2, 128, 128)
    assert features.shape == (2, model.feature_dim, 128, 128)
    assert model.feature_dim == model.head.feature_dim
    assert isinstance(both, tuple) and len(both) == 2
    assert both[0].shape == features.shape
    assert both[1].shape == logits.shape

    with pytest.raises(ValueError):
        model(x, mode="nope")


@pytest.mark.parametrize(
    "backbone_name,head_name",
    [("resnet50", "resunet"), ("swin_t", "deeplabv3p")],
)
def test_model_non_square_input(backbone_name, head_name):
    """Representative combos handle non-square 224x224 inputs."""
    model = build_model(_make_config(backbone_name, head_name))
    model.eval()
    x = torch.randn(1, 4, 224, 224)

    with torch.no_grad():
        logits = model(x)
        features = model(x, mode="feature")

    assert logits.shape == (1, 2, 224, 224)
    assert features.shape == (1, model.feature_dim, 224, 224)


@pytest.mark.parametrize("head_name", HEADS)
def test_backward_updates_backbone_and_head_gradients(head_name):
    """A backward pass produces gradients in both the backbone and the head."""
    model = build_model(_make_config("resnet50", head_name))
    model.train()
    x = torch.randn(2, 4, 128, 128)

    loss = model(x, mode="class").mean()
    loss.backward()

    backbone_grads = [
        param.grad
        for param in model.backbone.parameters()
        if param.requires_grad
    ]
    head_grads = [param.grad for param in model.head.parameters() if param.requires_grad]

    assert any(grad is not None for grad in backbone_grads)
    assert any(grad is not None for grad in head_grads)


@pytest.mark.parametrize("head_name", HEADS)
def test_freeze_unfreeze_backbone(head_name):
    """freeze_backbone disables backbone grads; unfreeze restores them."""
    model = build_model(_make_config("resnet50", head_name))
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    model.freeze_backbone()
    assert all(param.requires_grad is False for param in backbone_params)
    assert all(param.requires_grad is True for param in head_params)

    model.unfreeze_backbone()
    assert all(param.requires_grad is True for param in backbone_params)
    assert all(param.requires_grad is True for param in head_params)


def test_build_model_returns_segmentation_model():
    """build_model returns a SegmentationModel with composed submodules."""
    model = build_model(_make_config("resnet50", "resunet"))
    assert isinstance(model, SegmentationModel)
    assert isinstance(model, nn.Module)
    assert isinstance(model.backbone, nn.Module)
    assert isinstance(model.head, nn.Module)


def test_build_head_unknown_name_raises():
    """build_head raises ValueError for an unregistered head name."""
    config = ModelConfig(head="resunet", pretrained=False)
    with pytest.raises(ValueError):
        build_head("does_not_exist", RESNET50_CHANNELS, config)
