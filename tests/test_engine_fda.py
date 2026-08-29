"""Tests for the FDA engine and its Fourier / entropy helpers."""

import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines.fda import FDATrainer
from dares.losses.fda import charbonnier_entropy, fourier_domain_adaptation
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
        method="fda",
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
    engine = FDATrainer(model, source_loaders, target_loaders, config, device)

    trained = engine.fit()

    assert isinstance(trained, nn.Module)
    assert trained is engine.model
    assert engine.best_miou > 0.0
    for key in ("loss_total", "loss_ce", "loss_entropy"):
        assert len(engine.history[key]) == 1
        assert math.isfinite(engine.history[key][0])


def test_train_epoch_returns_finite_metrics(tmp_path):
    """train_epoch() returns the FDA metric set with finite values."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
    engine = FDATrainer(model, source_loaders, target_loaders, config, device)

    metrics = engine.train_epoch()

    assert set(metrics) == {"loss_total", "loss_ce", "loss_entropy", "epoch_time"}
    for value in metrics.values():
        assert math.isfinite(value)
    assert metrics["epoch_time"] >= 0.0


def test_fda_beta_zero_keeps_source():
    """beta ~ 0 leaves the source image essentially unchanged."""
    torch.manual_seed(0)
    src = torch.rand(2, 4, 32, 32)
    tgt = torch.rand(2, 4, 32, 32)

    adapted = fourier_domain_adaptation(src, tgt, beta=1e-6)

    assert adapted.shape == src.shape
    assert torch.isfinite(adapted).all()
    assert torch.allclose(adapted, src, atol=1e-4)


def test_fda_swap_changes_image():
    """A non-trivial beta transfers target style (image changes)."""
    torch.manual_seed(0)
    src = torch.rand(2, 4, 32, 32)
    tgt = torch.rand(2, 4, 32, 32) + 10.0

    adapted = fourier_domain_adaptation(src, tgt, beta=0.2)

    assert not torch.allclose(adapted, src, atol=1e-3)


def test_charbonnier_entropy_is_finite():
    """Charbonnier entropy is finite and non-negative."""
    logits = torch.randn(2, 2, 16, 16)
    loss = charbonnier_entropy(logits, eta=2.0)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0
