"""Tests for the DARES v2 (MIL-CREDA hardened) training engine.

Validates registry wiring, the extended metric set (``loss_local``), the
hyper-parameter plumbing into ``DARESLossV2`` and that the original V1
engine keeps its exact contract after the shared refactor.
"""

import math
from pathlib import Path

import pytest
import torch

from dares.config import ExperimentConfig, TrainConfig
from dares.engines.dares import DARESTrainer
from dares.engines.dares_v2 import DARESV2Trainer
from dares.engines.registry import build_engine
from dares.losses.dares_loss import DARESLoss
from dares.losses.dares_loss_v2 import DARESLossV2

from test_engine_dares import METRIC_KEYS, _build_fixtures

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_builds_v2(tmp_path):
    """build_engine('dares_v2') returns the v2 engine with the v2 criterion."""
    model, src, tgt, config, device = _build_fixtures(
        tmp_path, method="dares_v2"
    )
    engine = build_engine("dares_v2", model, src, tgt, config, device)

    assert isinstance(engine, DARESV2Trainer)
    assert isinstance(engine, DARESTrainer)
    assert isinstance(engine.criterion, DARESLossV2)


def test_v1_engine_contract_is_unchanged(tmp_path):
    """The refactor must not alter the original engine's behaviour or keys."""
    model, src, tgt, config, device = _build_fixtures(tmp_path)
    engine = DARESTrainer(model, src, tgt, config, device)

    assert type(engine.criterion) is DARESLoss

    metrics = engine.train_epoch()
    assert set(metrics) == METRIC_KEYS
    for key, value in metrics.items():
        if key != "loss_repulsion":
            assert math.isfinite(value), key


def test_v2_train_epoch_returns_extended_diagnostics(tmp_path):
    """train_epoch() exposes every V1 metric plus loss_local."""
    model, src, tgt, config, device = _build_fixtures(
        tmp_path, method="dares_v2"
    )
    engine = DARESV2Trainer(model, src, tgt, config, device)

    metrics = engine.train_epoch()

    assert set(metrics) == METRIC_KEYS | {"loss_local"}
    for key, value in metrics.items():
        if key != "loss_repulsion":
            assert math.isfinite(value), key
    assert metrics["loss_local"] >= 0.0


def test_v2_wires_new_hyperparameters(tmp_path):
    """The engine forwards the MIL-CREDA knobs from TrainConfig to the loss."""
    model, src, tgt, config, device = _build_fixtures(
        tmp_path,
        method="dares_v2",
        lambda_local=0.3,
        tau_local=0.7,
        soft_class_weights=False,
        bounded_align=False,
        normalize_seg=True,
    )
    engine = build_engine("dares_v2", model, src, tgt, config, device)

    crit = engine.criterion
    assert crit.lambda_local == 0.3
    assert crit.tau_local == 0.7
    assert crit.soft_class_weights is False
    assert crit.bounded_align is False
    assert crit.normalize_seg is True


def test_v2_warmup_keeps_lambda_zero(tmp_path):
    """During warm-up the v2 engine behaves exactly like the v1 one."""
    model, src, tgt, config, device = _build_fixtures(
        tmp_path, method="dares_v2", warmup_steps=10_000, warmup_epochs=None
    )
    engine = DARESV2Trainer(model, src, tgt, config, device)

    metrics = engine.train_epoch()

    assert metrics["lambda_eff"] == 0.0
    assert int(engine.criterion.step.item()) > 0


def test_v2_criterion_inherits_trust_region(tmp_path):
    """The shared GradNorm machinery (buffers, update_lambda) is intact."""
    model, src, tgt, config, device = _build_fixtures(
        tmp_path, method="dares_v2"
    )
    engine = DARESV2Trainer(model, src, tgt, config, device)
    crit = engine.criterion

    p = torch.nn.Parameter(torch.randn(4, 4))
    crit.step.fill_(crit.ramp_steps)
    lam = crit.update_lambda((p * 1.0).sum(), (p * 0.5).sum(), [p])

    assert 0.0 < lam <= crit.lambda_max
    assert crit.lambda_eff.item() == pytest.approx(lam, abs=1e-5)


def _v2_yaml_files():
    """Every dares_v2 config shipped under configs/ (LIME, architectures, ablations)."""
    return sorted((REPO_ROOT / "configs").rglob("dares_v2*.yaml"))


def test_dares_v2_yaml_configs_parse():
    """All shipped v2 configs validate, carry the new hyperparameters and
    keep their UTF-8 comments intact."""
    files = _v2_yaml_files()
    assert len(files) >= 15
    for path in files:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.training.method == "dares_v2", path
        assert cfg.training.lambda_local >= 0.0, path
        assert cfg.training.tau_local > 0.0, path
        assert isinstance(cfg.training.bounded_align, bool), path
        assert isinstance(cfg.training.soft_class_weights, bool), path
        raw = path.read_text(encoding="utf-8")
        assert "é" in raw and "ó" in raw, path
