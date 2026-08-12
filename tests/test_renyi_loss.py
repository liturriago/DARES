"""Tests for the DARES alpha-Renyi class-conditional alignment loss."""

import math

import torch

from dares.losses.renyi import RenyiLoss

METRIC_KEYS = {
    "loss_renyi",
    "valid_classes",
    "n_source_sampled",
    "n_target_sampled",
}


def _confident_logits(shape: tuple[int, int, int, int]) -> torch.Tensor:
    """Logits strongly biased toward class 0 (high per-pixel confidence)."""
    logits = torch.zeros(shape)
    logits[:, 0] = 10.0
    logits[:, 1] = 0.0
    return logits


def test_forward_is_finite_and_reports_metrics():
    """Forward on random tensors yields a finite alignment and full metrics."""
    torch.manual_seed(0)
    loss = RenyiLoss(num_classes=2)

    features_s = torch.randn(2, 8, 8, 8)
    labels_s = torch.randint(0, 2, (2, 8, 8))
    features_t = torch.randn(2, 8, 8, 8)
    logits_t = torch.randn(2, 2, 8, 8)
    logits_t[:, 0] += 4.0

    alignment, metrics = loss(features_s, labels_s, features_t, logits_t)

    assert torch.isfinite(alignment)
    assert math.isfinite(alignment.item())
    assert set(metrics) == METRIC_KEYS
    assert metrics["valid_classes"] >= 1
    assert metrics["n_source_sampled"] >= 0
    assert metrics["n_target_sampled"] >= 0
    assert math.isfinite(metrics["loss_renyi"])


def test_identical_sets_align_better_than_separated():
    """Class-0 features with identical means align better than far-apart ones."""
    torch.manual_seed(1)
    loss = RenyiLoss(num_classes=2)
    labels_s = torch.zeros(1, 8, 8, dtype=torch.long)
    logits_t = _confident_logits((1, 2, 8, 8))

    fs_identical = 0.5 * torch.randn(1, 8, 8, 8)
    ft_identical = 0.5 * torch.randn(1, 8, 8, 8)

    fs_separated = 0.5 * torch.randn(1, 8, 8, 8)
    ft_separated = 10.0 + 0.5 * torch.randn(1, 8, 8, 8)

    al_identical, m_identical = loss(
        fs_identical, labels_s, ft_identical, logits_t
    )
    al_separated, m_separated = loss(
        fs_separated, labels_s, ft_separated, logits_t
    )

    assert m_identical["valid_classes"] == 1
    assert m_separated["valid_classes"] == 1
    assert al_identical.item() > al_separated.item()


def test_single_class_forward_is_finite():
    """All pixels in one class yields a finite alignment and no exception."""
    torch.manual_seed(2)
    loss = RenyiLoss(num_classes=2)

    features_s = torch.randn(1, 8, 8, 8)
    labels_s = torch.zeros(1, 8, 8, dtype=torch.long)
    features_t = torch.randn(1, 8, 8, 8)
    logits_t = _confident_logits((1, 2, 8, 8))

    alignment, metrics = loss(features_s, labels_s, features_t, logits_t)

    assert math.isfinite(alignment.item())
    assert metrics["valid_classes"] in (0, 1)


def test_alignment_backpropagates_to_features():
    """backward() populates gradients on the source and target features."""
    torch.manual_seed(3)
    loss = RenyiLoss(num_classes=2)

    features_s = torch.randn(1, 8, 8, 8, requires_grad=True)
    features_t = torch.randn(1, 8, 8, 8, requires_grad=True)
    labels_s = torch.zeros(1, 8, 8, dtype=torch.long)
    logits_t = _confident_logits((1, 2, 8, 8))

    alignment, _ = loss(features_s, labels_s, features_t, logits_t)
    alignment.backward()

    assert features_s.grad is not None
    assert features_t.grad is not None
    assert features_s.grad.abs().sum().item() > 0.0
    assert features_t.grad.abs().sum().item() > 0.0
