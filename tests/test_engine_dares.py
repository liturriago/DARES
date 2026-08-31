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
    "loss_em",
    "h2_source_mean",
    "h2_target_mean",
    "delta_align_mean",
    "delta_repulsion_mean",
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
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
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
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    metrics = engine.train_epoch()

    assert set(metrics) == METRIC_KEYS
    for key, value in metrics.items():
        if key != "loss_repulsion":
            assert math.isfinite(value), key
    assert metrics["epoch_time"] >= 0.0


def test_reference_params_are_the_deep_encoder_block(tmp_path):
    """The engine uses the deepest shared encoder block as reference params."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    expected = engine.model.backbone.reference_params
    assert len(engine.ref_params) > 0
    assert all(any(r is e for e in expected) for r in engine.ref_params)


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


def test_scaler_update_called_after_each_step(tmp_path, monkeypatch):
    """Each optimizer step is followed by scaler.update() (GradScaler cycle).

    Regression: the engine routed through BaseTrainer._backward_step (which
    steps but does not update) and dropped the explicit scaler.update() call.
    On CUDA with a live GradScaler this raises 'step() has already been called
    since the last update()' on the second batch. This test asserts the step /
    update ordering contract by counting calls.
    """
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    calls = {"step": 0, "update": 0}
    orig_step = engine.scaler.step
    orig_update = engine.scaler.update

    def counting_step(optimizer):
        calls["step"] += 1
        return orig_step(optimizer)

    def counting_update():
        calls["update"] += 1
        return orig_update()

    monkeypatch.setattr(engine.scaler, "step", counting_step)
    monkeypatch.setattr(engine.scaler, "update", counting_update)

    metrics = engine.train_epoch()

    batches = int(metrics["epoch_time"] >= 0.0)  # at least one batch ran
    assert calls["step"] >= 1
    # update() must run at least once; a step may be skipped by the NaN guard
    # but an update is always issued after it (step <= update).
    assert calls["update"] >= 1
    assert calls["step"] <= calls["update"]


def test_criterion_is_on_model_device(tmp_path):
    """The DARESLoss buffers must live on the same device as the model.

    Regression: the criterion was built without .to(device), so its registered
    buffers (step / lambda_eff / ema_g_seg / ema_g_aux) stayed on CPU while the
    model and gradients were on cuda. update_lambda's GradNorm then hit
    'Expected all tensors to be on the same device' once step >= warmup_steps.
    """
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    model_dev = next(model.parameters()).device
    assert engine.criterion.step.device == model_dev
    assert engine.criterion.lambda_eff.device == model_dev
    assert engine.criterion.ema_g_seg.device == model_dev
    assert engine.criterion.ema_g_aux.device == model_dev
    # All reference params must be on the same device too.
    assert all(p.device == model_dev for p in engine.ref_params)


def test_skipped_step_does_not_break_scaler_update(tmp_path, monkeypatch):
    """If the NaN guard skips the optimizer step, scaler.update() must not run.

    Regression: the engine called scaler.update() unconditionally after
    _backward_step. When gradients were non-finite (guard returns False) the
    step was skipped, no inf check was recorded by the scaler, and update()
    asserted 'No inf checks were recorded prior to update.'.

    The engine now finalizes the AMP cycle only after a real step. We force the
    gradient guard to fail on every batch (exactly what happens on non-finite
    gradients) and check that the AMP cycle is never finalized.
    """
    model, source_loaders, target_loaders, config, device = _build_fixtures(tmp_path)
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    update_calls = []
    orig_update = engine.scaler.update

    def counting_update():
        update_calls.append(1)
        return orig_update()

    monkeypatch.setattr(engine.scaler, "update", counting_update)

    # Simulate the NaN gradient guard tripping on every batch: the step is
    # skipped by _backward_step, so scaler.update() must never be called.
    monkeypatch.setattr(engine, "_guard_step", lambda optimizer, params=None: False)

    metrics = engine.train_epoch()  # must not raise

    assert metrics["epoch_time"] >= 0.0
    assert len(update_calls) == 0  # update() skipped alongside the skipped step


def test_criterion_wires_align_form_and_em(tmp_path):
    """The engine forwards align_form / EM settings into the DARESLoss."""
    model, source_loaders, target_loaders, config, device = _build_fixtures(
        tmp_path, align_form="mi", use_renyi_em=False, lambda_em=0.2
    )
    engine = DARESTrainer(model, source_loaders, target_loaders, config, device)

    assert engine.criterion.align_form == "mi"
    assert engine.criterion.use_renyi_em is False
    assert engine.criterion.lambda_em == 0.2


def test_train_config_defaults_are_headline():
    """TrainConfig defaults select the headline MI + Renyi-EM, trust-region setup."""
    cfg = TrainConfig(method="dares")

    assert cfg.align_form == "mi"
    assert cfg.use_renyi_em is True
    assert cfg.lambda_em == 0.05
    assert cfg.trust_region is True
