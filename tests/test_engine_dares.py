"""Tests for the DARES (DARES hardened alignment) training engine."""

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
from dares.losses.dares_loss import DARESLoss
from dares.models.segmentation import build_model

METRIC_KEYS = {
    "loss_total",
    "loss_seg",
    "loss_align",
    "loss_anti_collapse",
    "loss_repulsion",
    "h2_source_mean",
    "h2_target_mean",
    "lambda_eff",
    "n_valid_classes",
    "n_rep_pairs",
    "epoch_time",
}


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
        "device": "cpu",
        "use_amp": False,
        "seed": 42,
        "warmup_steps": 0,  # alignment active from the first step in tests
        "ramp_steps": 10,
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
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    trained = engine.fit()

    assert isinstance(trained, nn.Module)
    assert trained is engine.model
    assert engine.best_miou > 0.0
    for key in ("loss_total", "loss_seg", "loss_align", "loss_anti_collapse"):
        assert key in engine.history
        assert len(engine.history[key]) == 1
        assert math.isfinite(engine.history[key][0])


def test_train_epoch_returns_all_diagnostics(tmp_path):
    """train_epoch() returns the full DARES diagnostic set + epoch_time."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path
    )
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    metrics = engine.train_epoch()

    assert set(metrics) == METRIC_KEYS
    for key, value in metrics.items():
        if key != "loss_repulsion":
            assert math.isfinite(value), key
    assert metrics["epoch_time"] >= 0.0


def test_reference_params_are_the_deep_encoder_block(tmp_path):
    """The engine uses the deepest shared encoder block as reference params."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path
    )
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    expected = engine.model.backbone.reference_params
    assert len(engine.ref_params) > 0
    assert all(
        any(r is e for e in expected) for r in engine.ref_params
    )


def test_update_lambda_warmup_zeroes_effective_lambda():
    """Before warmup_steps, lambda_eff stays 0 and step advances."""
    crit = DARESLoss(num_classes=2, warmup_steps=5)
    params = [
        torch.nn.Parameter(torch.randn(4, 4)),
        torch.nn.Parameter(torch.randn(4)),
    ]
    loss_seg = torch.randn((), requires_grad=True)
    loss_aux = torch.randn((), requires_grad=True)

    lam = crit.update_lambda(loss_seg, loss_aux, params)

    assert lam == 0.0
    assert crit.lambda_eff.item() == 0.0
    assert crit.step.item() == 1


def test_update_lambda_grows_after_warmup():
    """After warmup, lambda_eff rises with the sigmoid ramp (nonzero)."""
    crit = DARESLoss(num_classes=2, warmup_steps=0, ramp_steps=10)
    p = torch.nn.Parameter(torch.randn(4, 4))

    loss_seg = (p * 1.0).sum()
    loss_aux = (p * 0.5).sum()

    crit.step.fill_(5)  # inside ramp (p = 5/10)
    lam = crit.update_lambda(loss_seg, loss_aux, [p])

    assert lam > 0.0
    assert lam <= crit.lambda_max
    assert crit.lambda_eff.item() == pytest.approx(lam, abs=1e-5)


def test_dares_loss_forward_backward_finite():
    """A single DARES loss forward + update_lambda + backward is finite."""
    torch.manual_seed(0)
    crit = DARESLoss(num_classes=2, warmup_steps=0)
    fs = torch.randn(2, 8, 8, 8, requires_grad=True)
    ft = torch.randn(2, 8, 8, 8, requires_grad=True)
    ls = torch.randint(0, 2, (2, 8, 8))
    logits_s = torch.randn(2, 2, 8, 8, requires_grad=True)
    lt = torch.randn(2, 2, 8, 8)

    total, parts = crit(fs, logits_s, ls, ft, lt)
    total.backward()

    assert torch.isfinite(total)
    assert parts["loss_seg"].requires_grad
    assert parts["loss_aux"].requires_grad
    assert fs.grad is not None
    assert logits_s.grad is not None
    assert bool(torch.isfinite(fs.grad).all())


def test_dares_loss_asymmetric_anchor_source_grad_zero():
    """Source features receive no alignment gradient (asymmetric anchoring)."""
    torch.manual_seed(1)
    crit = DARESLoss(num_classes=2, warmup_steps=0)
    fs = torch.randn(2, 8, 8, 8, requires_grad=True)
    ft = torch.randn(2, 8, 8, 8, requires_grad=True)
    ls = torch.randint(0, 2, (2, 8, 8))
    logits_s = torch.randn(2, 2, 8, 8)
    lt = torch.randn(2, 2, 8, 8)

    _, parts = crit(fs, logits_s, ls, ft, lt)
    parts["loss_aux"].backward()

    assert fs.grad is not None
    assert torch.all(fs.grad == 0)


def test_dares_loss_amp_safe_under_autocast():
    """Runs the loss under torch.autocast without dtype errors."""
    torch.manual_seed(2)
    crit = DARESLoss(num_classes=2, warmup_steps=0)
    fs = torch.randn(2, 8, 8, 8)
    ft = torch.randn(2, 8, 8, 8)
    ls = torch.randint(0, 2, (2, 8, 8))
    logits_s = torch.randn(2, 2, 8, 8)
    lt = torch.randn(2, 2, 8, 8)

    with torch.autocast(device_type="cpu", enabled=True):
        total, parts = crit(fs, logits_s, ls, ft, lt)

    assert torch.isfinite(total)
    assert total.dtype == torch.float32


def test_warmup_epochs_freeze_unfreeze_lifecycle(tmp_path):
    """fit() freezes the backbone for warmup_epochs then unfreezes it, and
    keeps lambda_eff at 0 while step < warmup_steps."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path, epochs=3, warmup_epochs=2, warmup_steps=1000
    )
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    # Before fit(): backbone ref params are unfrozen by default.
    assert all(p.requires_grad for p in engine.ref_params)

    engine.fit()

    # After warmup_epochs, the backbone must be fully unfrozen.
    assert all(p.requires_grad for p in engine.ref_params)
    # With 8 train patches / batch_size 2 = 4 batches/epoch * 2 epochs = 8 < 1000,
    # lambda_eff stayed 0 throughout warmup.
    assert all(l == 0.0 for l in engine.history["lambda_eff"])


def test_lambda_eff_zero_during_warmup_steps(tmp_path):
    """Alignment weight is exactly 0 while the criterion step < warmup_steps."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path, warmup_steps=10_000, warmup_epochs=None
    )
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    metrics = engine.train_epoch()

    assert metrics["lambda_eff"] == 0.0
    assert int(engine.criterion.step.item()) > 0
