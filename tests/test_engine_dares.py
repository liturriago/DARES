"""Tests for the DARES (alpha-Renyi alignment) training engine."""

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines.dares import DARESTrainer
from dares.models.segmentation import build_model


def make_h5(path: Path, n: int = 8, h: int = 64, w: int = 64) -> None:
    """Writes a tiny synthetic HDF5 container with images and binary masks."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        images = np.random.rand(n, 4, h, w).astype(np.float32)
        f.create_dataset(
            "images",
            data=images,
            compression="lzf",
            chunks=(1, 4, h, w),
        )
        masks = np.random.randint(0, 2, size=(n, h, w)).astype(np.uint8)
        f.create_dataset(
            "masks",
            data=masks,
            compression="lzf",
            chunks=(1, h, w),
        )


def _make_six_containers(tmp_path: Path) -> None:
    """Creates the six split containers (source/target x train/val/test)."""
    for domain in ("source", "target"):
        for split, n in (("train", 8), ("val", 4), ("test", 4)):
            make_h5(tmp_path / f"{domain}_{split}.h5", n=n)


def _build_fixtures(tmp_path: Path, **train_overrides):
    """Returns (model, source_loaders, target_loaders, config, device)."""
    _make_six_containers(tmp_path)
    data_config = DataConfig(
        source_dir=tmp_path,
        target_dir=tmp_path,
        batch_size=2,
        num_workers=0,
    )
    loader = DARESDataLoader(data_config)
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
        method="dares",
        epochs=1,
        lr=1e-4,
        lambda_renyi=0.1,
        tau=0.85,
        n_max=1024,
        sigma="auto",
        alpha=2,
        device="cpu",
        use_amp=False,
        seed=42,
        **train_overrides,
    )
    device = torch.device("cpu")
    return (
        model,
        loader.get_source_loaders(),
        loader.get_target_loaders(),
        config,
        device,
    )


def test_fit_returns_model_and_tracks_history(tmp_path):
    """fit() returns the model and records the DARES loss history."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path
    )
    engine = DARESTrainer(
        model, source_loaders, target_loaders, config, device
    )

    trained = engine.fit()

    assert isinstance(trained, nn.Module)
    assert trained is engine.model
    assert engine.best_miou > 0.0
    for key in ("loss_total", "loss_ce", "loss_renyi"):
        assert key in engine.history
        assert len(engine.history[key]) == 1
        assert math.isfinite(engine.history[key][0])


def test_train_epoch_returns_finite_metrics(tmp_path):
    """train_epoch() returns finite scalars including epoch_time."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path
    )
    engine = DARESTrainer(
        model, source_loaders, target_loaders, config, device
    )

    metrics = engine.train_epoch()

    assert set(metrics) == {
        "loss_total",
        "loss_ce",
        "loss_renyi",
        "lambda_active",
        "valid_classes",
        "epoch_time",
    }
    for value in metrics.values():
        assert math.isfinite(value)
    assert metrics["epoch_time"] >= 0.0
    assert metrics["lambda_active"] == pytest.approx(0.1)


def test_warmup_disables_alignment(tmp_path):
    """During warmup epochs the active alignment weight is zero."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path, warmup_epochs=5
    )
    engine = DARESTrainer(
        model, source_loaders, target_loaders, config, device
    )

    metrics = engine.train_epoch()

    assert metrics["lambda_active"] == 0.0
