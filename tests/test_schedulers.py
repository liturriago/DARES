"""Tests for the CREDA dynamic learning-rate scheduler."""

import pytest
import torch
from torch.optim import Adam

from dares.config import TrainConfig
from dares.training.schedulers import CREDALRScheduler, build_scheduler


def _optimizer(lr: float = 1e-4):
    params = torch.nn.Parameter(torch.zeros(4))
    return Adam([params], lr=lr)


def test_creda_scheduler_starts_at_base_lr():
    opt = _optimizer(lr=1e-4)
    sched = CREDALRScheduler(opt, total_epochs=45, alpha=20.0, beta=0.75)
    # Base LR applies during epoch 1 (LRScheduler init sets last_epoch=0).
    assert opt.param_groups[0]["lr"] == 1e-4
    sched.step()  # advance to epoch 2 -> p = 1/45
    expected = 1e-4 * (1.0 + 20.0 / 45.0) ** (-0.75)
    assert opt.param_groups[0]["lr"] == pytest.approx(expected)


def test_creda_scheduler_monotonic_decrease():
    opt = _optimizer(lr=1e-4)
    sched = CREDALRScheduler(opt, total_epochs=45)
    lrs = []
    for _ in range(45):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))
    assert lrs[-1] < 1e-4


def test_creda_scheduler_final_factor():
    """At p = 1 the factor equals (1 + alpha)^(-beta)."""
    opt = _optimizer(lr=2e-3)
    sched = CREDALRScheduler(opt, total_epochs=45, alpha=20.0, beta=0.75)
    for _ in range(45):
        sched.step()
    expected = 2e-3 * (1.0 + 20.0) ** (-0.75)
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6)


def test_build_scheduler_modes():
    opt = _optimizer()
    dynamic = build_scheduler(opt, TrainConfig(lr_schedule=True, epochs=45))
    assert isinstance(dynamic, CREDALRScheduler)

    opt2 = _optimizer()
    exponential = build_scheduler(opt2, TrainConfig(lr_schedule=False, gamma=0.94))
    assert isinstance(exponential, torch.optim.lr_scheduler.ExponentialLR)
