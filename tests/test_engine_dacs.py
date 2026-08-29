"""Tests for the DACS engine and its cross-domain mixing helpers."""

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines.dacs import DACSTrainer
from dares.losses.dacs import class_mix, color_jitter, gaussian_blur, pseudo_label
from dares.models.segmentation import build_model


def make_h5(path: Path, n: int = 8, h: int = 64, w: int = 64) -> None:
    """Writes a tiny synthetic HDF5 container with images and binary masks."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        images = np.random.rand(n, 4, h, w).astype(np.float32)
        f.create_dataset("images", data=images, compression="lzf", chunks=(1, 4, h, w))
        masks = np.random.randint(0, 2, size=(n, h, w)).astype(np.uint8)
        f.create_dataset("masks", data=masks, compression="lzf", chunks=(1, h, w))


def _make_six_containers(tmp_path: Path) -> None:
    for domain in ("source", "target"):
        for split, n in (("train", 8), ("val", 4), ("test", 4)):
            make_h5(tmp_path / f"{domain}_{split}.h5", n=n)


def _build_fixtures(tmp_path: Path):
    _make_six_containers(tmp_path)
    data_config = DataConfig(
        source_dir=tmp_path, target_dir=tmp_path, batch_size=2, num_workers=0
    )
    loader = DARESDataLoader(data_config)
    model = build_model(
        ModelConfig(
            backbone="resnet50", head="resunet", in_channels=4, num_classes=2, pretrained=False
        )
    )
    config = TrainConfig(
        method="dacs",
        epochs=1,
        lr=1e-4,
        device="cpu",
        use_amp=False,
        seed=42,
    )
    device = torch.device("cpu")
    return model, loader.get_source_loaders(), loader.get_target_loaders(), config, device


def test_fit_returns_model_and_tracks_history(tmp_path):
    """fit() returns the model and records finite loss history."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
    engine = DACSTrainer(model, source_loaders, target_loaders, config, device)

    trained = engine.fit()

    assert isinstance(trained, nn.Module)
    assert trained is engine.model
    assert engine.best_miou > 0.0
    for key in ("loss_total", "loss_ce", "loss_mix", "lambda_unsup"):
        assert len(engine.history[key]) == 1
        assert math.isfinite(engine.history[key][0])


def test_train_epoch_returns_finite_metrics(tmp_path):
    """train_epoch() returns the DACS metric set with finite values."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
    engine = DACSTrainer(model, source_loaders, target_loaders, config, device)

    metrics = engine.train_epoch()

    assert set(metrics) == {"loss_total", "loss_ce", "loss_mix", "lambda_unsup", "epoch_time"}
    for value in metrics.values():
        assert math.isfinite(value)
    assert metrics["epoch_time"] >= 0.0
    assert 0.0 <= metrics["lambda_unsup"] <= 1.0


def test_pseudo_label_thresholds_low_confidence():
    """Low-confidence pixels are marked with the ignore index."""
    logits = torch.zeros(1, 2, 1, 2)
    logits[:, 0] = 5.0  # high confidence -> class 0 kept
    logits[:, 1, :, 1] = 5.0  # ties -> low confidence (0.5) -> ignored

    pseudo, confident = pseudo_label(logits, threshold=0.9)

    assert pseudo.shape == (1, 1, 2)
    assert pseudo[0, 0, 0] == 0
    assert pseudo[0, 0, 1] == 255
    assert confident[0, 0, 0]
    assert not confident[0, 0, 1]


def test_class_mix_shapes_and_masks():
    """class_mix pastes selected source classes onto the target image."""
    torch.manual_seed(0)
    src_img = torch.zeros(2, 4, 8, 8)
    tgt_img = torch.ones(2, 4, 8, 8)
    src_label = torch.zeros(2, 8, 8, dtype=torch.long)
    tgt_pseudo = torch.ones(2, 8, 8, dtype=torch.long)

    mixed_img, mixed_label = class_mix(src_img, src_label, tgt_img, tgt_pseudo)

    assert mixed_img.shape == (2, 4, 8, 8)
    assert mixed_label.shape == (2, 8, 8)
    # All pixels are 0 or 1 (no leakage outside the two images).
    assert torch.all((mixed_img == 0.0) | (mixed_img == 1.0))
    assert torch.all((mixed_label == 0) | (mixed_label == 1))


def test_color_jitter_and_blur_are_finite():
    """Augmentations preserve shape / dtype and stay finite."""
    torch.manual_seed(1)
    x = torch.rand(2, 4, 16, 16)

    jittered = color_jitter(x, 0.5, 0.5, 0.5, 0.25)
    blurred = gaussian_blur(x, kernel_size=5, sigma=(0.1, 2.0))

    assert jittered.shape == x.shape
    assert blurred.shape == x.shape
    assert torch.isfinite(jittered).all()
    assert torch.isfinite(blurred).all()
