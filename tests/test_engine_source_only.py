"""Tests for the source-only (no adaptation) training engine."""

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines.source_only import SourceOnlyTrainer
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


def _build_fixtures(tmp_path: Path):
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
        method="source_only",
        epochs=1,
        lr=1e-4,
        device="cpu",
        use_amp=False,
        seed=42,
    )
    device = torch.device("cpu")
    return (
        model,
        loader.get_source_loaders(),
        loader.get_target_loaders(),
        config,
        device,
    )


def test_fit_returns_model_and_tracks_best_miou(tmp_path):
    """fit() returns the model and records a positive target-val mIoU."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path
    )
    engine = SourceOnlyTrainer(
        model, source_loaders, target_loaders, config, device
    )

    trained = engine.fit()

    assert isinstance(trained, nn.Module)
    assert trained is engine.model
    assert engine.best_miou > 0.0
    assert len(engine.history["train_loss"]) == 1
    assert len(engine.history["train_acc"]) == 1
    assert math.isfinite(engine.history["train_loss"][0])
    assert math.isfinite(engine.history["train_acc"][0])


def test_train_epoch_returns_finite_metrics(tmp_path):
    """train_epoch() returns finite floats and a non-negative epoch_time."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path
    )
    engine = SourceOnlyTrainer(
        model, source_loaders, target_loaders, config, device
    )

    metrics = engine.train_epoch()

    assert set(metrics) == {"train_loss", "train_acc", "epoch_time"}
    for value in metrics.values():
        assert math.isfinite(value)
    assert metrics["epoch_time"] >= 0.0
    assert metrics["train_loss"] > 0.0
    assert 0.0 <= metrics["train_acc"] <= 1.0


def test_direct_loss_is_positive_and_finite(tmp_path):
    """A single source batch yields a positive, finite cross-entropy loss."""
    model, source_loaders, _, config, device = _build_fixtures(tmp_path)
    engine = SourceOnlyTrainer(
        model, source_loaders, {}, config, device
    )
    engine.model.train()

    batch = next(iter(source_loaders["train"]))
    imgs, masks = batch[0].to(device), batch[1].to(device)
    logits = engine.model(imgs, mode="class")
    loss = engine.criterion(logits, masks)

    loss_value = loss.item()
    assert loss_value > 0.0
    assert math.isfinite(loss_value)
