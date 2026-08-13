"""Self-contained tests for the CyCADA training engine and its losses."""

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines.cycada import CyCADATrainer
from dares.losses.cycada import PixelGenerator, cycle_consistency_loss
from dares.models.segmentation import build_model


def make_h5(
    file_path: Path,
    num_patches: int,
    include_masks: bool = True,
    height: int = 64,
    width: int = 64,
) -> None:
    """Writes a tiny synthetic HDF5 container (images + optional masks)."""
    np.random.seed(0)
    with h5py.File(file_path, "w") as f:
        images = np.random.rand(num_patches, 4, height, width).astype(np.float32)
        f.create_dataset(
            "images",
            data=images,
            compression="lzf",
            chunks=(1, 4, height, width),
        )
        if include_masks:
            masks = np.random.randint(0, 2, size=(num_patches, height, width)).astype(
                np.uint8
            )
            f.create_dataset(
                "masks",
                data=masks,
                compression="lzf",
                chunks=(1, height, width),
            )


@pytest.fixture
def loaders(tmp_path: Path):
    """Builds source (labeled, 8 train patches) and target loaders in tmp_path."""
    for domain in ("source", "target"):
        for split, include_masks in (
            ("train", domain == "source"),
            ("val", True),
            ("test", True),
        ):
            num_patches = 8 if split == "train" else 4
            make_h5(
                tmp_path / f"{domain}_{split}.h5",
                num_patches=num_patches,
                include_masks=include_masks,
            )
    data_config = DataConfig(
        source_dir=tmp_path,
        target_dir=tmp_path,
        batch_size=2,
        num_workers=0,
    )
    loader = DARESDataLoader(data_config)
    return loader.get_source_loaders(), loader.get_target_loaders()


def build_engine(source_loaders, target_loaders) -> CyCADATrainer:
    """Builds a small resnet50 + resunet CyCADA engine on CPU."""
    torch.manual_seed(42)
    model = build_model(
        ModelConfig(
            backbone="resnet50",
            head="resunet",
            in_channels=4,
            num_classes=2,
            pretrained=False,
        )
    )
    config = TrainConfig(
        method="cycada",
        epochs=1,
        lr=1e-4,
        lambda_cycle=1.0,
        lambda_identity=0.1,
        lambda_pixel=1.0,
        lambda_feat=1.0,
        device="cpu",
        use_amp=False,
        seed=42,
    )
    return CyCADATrainer(
        model, source_loaders, target_loaders, config, torch.device("cpu")
    )


def test_fit_returns_model_and_tracks_history(loaders):
    """fit() returns the trained model, improves mIoU, and fills history."""
    source_loaders, target_loaders = loaders
    engine = build_engine(source_loaders, target_loaders)

    result = engine.fit()

    assert isinstance(result, nn.Module)
    assert engine.best_miou > 0.0
    for key in (
        "loss_total",
        "loss_task",
        "loss_cycle",
        "loss_identity",
        "loss_pix_adv",
        "loss_feat_adv",
    ):
        assert key in engine.history
        assert len(engine.history[key]) == 1


def test_train_epoch_returns_finite_scalars(loaders):
    """train_epoch() returns finite scalars for every loss and epoch_time."""
    source_loaders, target_loaders = loaders
    engine = build_engine(source_loaders, target_loaders)

    metrics = engine.train_epoch()

    assert set(metrics) == {
        "loss_total",
        "loss_task",
        "loss_cycle",
        "loss_identity",
        "loss_pix_adv",
        "loss_feat_adv",
        "epoch_time",
    }
    assert metrics["epoch_time"] >= 0.0
    for value in metrics.values():
        assert math.isfinite(value)


def test_feat_adv_disabled_during_warmup(loaders):
    """During the warm-up the feature-adversarial term is excluded from total."""
    source_loaders, target_loaders = loaders
    torch.manual_seed(42)
    model = build_model(
        ModelConfig(
            backbone="resnet50",
            head="resunet",
            in_channels=4,
            num_classes=2,
            pretrained=False,
        )
    )
    config = TrainConfig(
        method="cycada",
        epochs=1,
        lr=1e-4,
        lambda_cycle=1.0,
        lambda_identity=0.1,
        lambda_pixel=1.0,
        lambda_feat=1.0,
        warmup_epochs=5,
        device="cpu",
        use_amp=False,
        seed=42,
    )
    engine = CyCADATrainer(
        model, source_loaders, target_loaders, config, torch.device("cpu")
    )

    metrics = engine.train_epoch()

    expected = (
        metrics["loss_task"]
        + 1.0 * metrics["loss_cycle"]
        + 0.1 * metrics["loss_identity"]
        + 1.0 * metrics["loss_pix_adv"]
    )
    assert metrics["loss_total"] == pytest.approx(expected, rel=1e-3)


def test_generators_change_pixels(loaders):
    """After one epoch the generators no longer map images to themselves."""
    source_loaders, target_loaders = loaders
    engine = build_engine(source_loaders, target_loaders)

    engine.train_epoch()

    fixed = torch.randn(2, 4, 64, 64)
    engine.g_st.eval()
    with torch.no_grad():
        translated = engine.g_st(fixed)

    assert translated.shape == fixed.shape
    assert not torch.allclose(translated, fixed, atol=1e-6)


class _IdentityGenerator(nn.Module):
    """Trivial generator that maps every image to itself."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ShiftGenerator(nn.Module):
    """Generator that rolls images one pixel along the width axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, shifts=1, dims=-1)


def test_cycle_consistency_loss_unit():
    """Cycle loss is zero for identity generators and positive otherwise."""
    x_s = torch.randn(2, 4, 64, 64)
    x_t = torch.randn(2, 4, 64, 64)

    identity = _IdentityGenerator()
    assert cycle_consistency_loss(identity, identity, x_s, x_t).item() == 0.0

    shift = _ShiftGenerator()
    assert cycle_consistency_loss(shift, shift, x_s, x_t).item() > 0.0

    real = PixelGenerator(in_channels=4)
    assert cycle_consistency_loss(real, real, x_s, x_t).item() > 0.0
