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
    config_kwargs = {
        "method": "dares",
        "epochs": 1,
        "lr": 1e-4,
        "lambda_renyi": 0.1,
        "tau": 0.85,
        "n_max": 1024,
        "sigma": "auto",
        "alpha": 2,
        "device": "cpu",
        "use_amp": False,
        "seed": 42,
    }
    config_kwargs.update(train_overrides)
    config = TrainConfig(**config_kwargs)
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


def test_grid_size_is_forwarded_to_renyi_loss(tmp_path):
    """The config grid_size reaches the RenyiLoss sampling operator."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path, grid_size=16
    )
    engine = DARESTrainer(
        model, source_loaders, target_loaders, config, device
    )

    assert engine.renyi_loss.grid_size == 16


def test_lambda_ramp_grows_with_training_progress(tmp_path):
    """The CREDA ramp monotonically grows lambda toward lambda_renyi."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path, epochs=4, schedule_delta=8, warmup_epochs=None
    )
    engine = DARESTrainer(
        model, source_loaders, target_loaders, config, device
    )

    lambdas = [engine.train_epoch()["lambda_active"] for _ in range(4)]

    assert lambdas[0] < config.lambda_renyi
    assert lambdas[-1] == pytest.approx(config.lambda_renyi, abs=1e-3)
    assert all(a <= b for a, b in zip(lambdas, lambdas[1:]))


def test_schedule_delta_zero_keeps_constant_lambda(tmp_path):
    """schedule_delta=0 disables the ramp and fixes lambda at lambda_renyi."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path, epochs=3, schedule_delta=0
    )
    engine = DARESTrainer(
        model, source_loaders, target_loaders, config, device
    )

    lambdas = [engine.train_epoch()["lambda_active"] for _ in range(3)]

    assert lambdas == [config.lambda_renyi] * 3


def test_lambda_ramp_respects_warmup(tmp_path):
    """Alignment stays off during warmup and ramps in afterwards."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path, epochs=4, warmup_epochs=2
    )
    engine = DARESTrainer(
        model, source_loaders, target_loaders, config, device
    )

    lambdas = [engine.train_epoch()["lambda_active"] for _ in range(4)]

    assert lambdas[0] == 0.0
    assert lambdas[1] == 0.0
    assert lambdas[2] > 0.0
    assert lambdas[3] > 0.0
