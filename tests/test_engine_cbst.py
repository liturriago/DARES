"""Tests for the CBST engine and its class-balanced pseudo-labeling losses."""

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines.cbst import CBSTTrainer
from dares.losses.cbst import CBSTPseudoLabeling, CBSTSelfTrainingLoss
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
        method="cbst",
        epochs=1,
        lr=1e-4,
        lambda_self=1.0,
        pseudo_threshold=0.9,
        pseudo_topk_ratio=0.5,
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
    engine = CBSTTrainer(model, source_loaders, target_loaders, config, device)

    trained = engine.fit()

    assert isinstance(trained, nn.Module)
    assert trained is engine.model
    assert engine.best_miou > 0.0
    assert len(engine.history["loss_total"]) == 1
    assert len(engine.history["loss_ce"]) == 1
    assert len(engine.history["loss_self"]) == 1
    assert math.isfinite(engine.history["loss_total"][0])
    assert math.isfinite(engine.history["loss_ce"][0])
    assert math.isfinite(engine.history["loss_self"][0])


def test_train_epoch_returns_finite_metrics(tmp_path):
    """train_epoch() returns finite floats and a non-negative epoch_time."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path
    )
    engine = CBSTTrainer(model, source_loaders, target_loaders, config, device)

    metrics = engine.train_epoch()

    assert set(metrics) == {"loss_total", "loss_ce", "loss_self", "epoch_time"}
    for value in metrics.values():
        assert math.isfinite(value)
    assert metrics["epoch_time"] >= 0.0
    assert metrics["loss_ce"] > 0.0


def test_pseudo_labeler_selects_high_confidence_pixels():
    """Weights are 1 only where the confidence exceeds the threshold."""
    num_classes = 2
    labeler = CBSTPseudoLabeling(num_classes, topk_ratio=1.0, threshold=0.9)
    logits = torch.zeros(1, num_classes, 1, 4)
    logits[:, 0, :, 0] = 5.0   # conf ~0.993  -> kept
    logits[:, 0, :, 1] = 2.0   # conf ~0.881  -> below threshold
    logits[:, 0, :, 3] = 10.0  # conf ~0.9999 -> kept

    pseudo, weights = labeler(logits)

    assert pseudo.shape == (1, 1, 4)
    assert torch.all(pseudo == 0)
    expected = torch.zeros(1, 1, 4)
    expected[:, :, 0] = 1.0
    expected[:, :, 3] = 1.0
    torch.testing.assert_close(weights, expected)


def test_pseudo_labeler_uniform_logits_yields_no_weights():
    """Uniform logits (confidence 1/C < threshold) select no pixels."""
    num_classes = 4
    labeler = CBSTPseudoLabeling(num_classes, topk_ratio=0.5, threshold=0.9)
    logits = torch.zeros(2, num_classes, 8, 8)

    pseudo, weights = labeler(logits)

    assert pseudo.shape == (2, 8, 8)
    assert pseudo.dtype == torch.int64
    assert weights.sum() == 0


def test_self_training_loss_zero_weights_is_zero():
    """A fully masked target batch yields a zero loss."""
    loss_fn = CBSTSelfTrainingLoss()
    logits = torch.randn(2, 3, 4, 4)
    pseudo = torch.zeros(2, 4, 4, dtype=torch.long)
    weights = torch.zeros(2, 4, 4)

    loss = loss_fn(logits, pseudo, weights)

    assert float(loss) == 0.0


def test_self_training_loss_correct_pixels_is_small():
    """Cross-entropy on confident, correct pseudo-pixels stays small."""
    loss_fn = CBSTSelfTrainingLoss()
    logits = torch.zeros(1, 2, 2, 2)
    logits[:, 0] = 2.0
    logits[:, 1] = 0.0
    pseudo = torch.zeros(1, 2, 2, dtype=torch.long)
    weights = torch.ones(1, 2, 2)

    loss = float(loss_fn(logits, pseudo, weights))

    assert loss > 0.0
    assert loss < 1.0
    assert math.isfinite(loss)