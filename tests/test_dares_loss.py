"""Focused tests for the hardened DARES loss mechanics.

Covers the three hardened safeguards from ``Docs/KimiReport.txt``:
anti-collapse entropy floors, asymmetric (stop-gradient) anchoring, the
margin-hinged inter-class repulsion and the GradNorm-lite trust region.
"""

import math

import pytest
import torch

from dares.losses.dares_loss import DARESLoss


def _feats(n=2, h=16, w=16, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    fs = torch.randn(n, d, h, w, generator=g)
    ft = torch.randn(n, d, h, w, generator=g)
    ls = torch.randint(0, 2, (n, h, w), generator=g)
    return fs, ft, ls


def test_anti_collapse_penalizes_degenerate(class_collapsed=True):
    """Collapsed class features push H2 below the floor and activate L_ac."""
    crit = DARESLoss(num_classes=2, warmup_steps=0, eta_floor=1.0)
    n, h, w, d = 2, 16, 16, 8
    fs = torch.randn(n, d, h, w)
    ft = torch.zeros(n, 1, h, w).repeat(1, d, 1, 1)  # collapsed target
    ls = torch.ones(n, h, w, dtype=torch.long)
    logits_s = torch.randn(n, 2, h, w)
    logits_t = torch.randn(n, 2, h, w)
    logits_t[:, 1] += 5.0  # confident class 1 pseudo-labels

    _, parts = crit(fs, logits_s, ls, ft, logits_t)

    # A collapsed (zero-variance) target cloud yields H2_target == 0.
    assert parts["h2_target_mean"].item() == pytest.approx(0.0, abs=1e-3)
    assert parts["loss_anti_collapse"].item() > 0.0


def test_anti_collapse_inactive_for_healthy():
    """Well-spread classes keep H2 above the floor so L_ac stays ~0."""
    torch.manual_seed(0)
    crit = DARESLoss(num_classes=2, warmup_steps=0, eta_floor=1.0)
    fs, ft, ls = _feats(seed=3)
    # Scale up the dispersion so both H2_source and H2_target comfortably clear
    # the eta_floor and the entropy-gap floor.
    fs = fs * 4.0
    ft = ft * 4.0
    logits_s = torch.randn(2, 2, 16, 16)
    logits_t = torch.randn(2, 2, 16, 16)
    logits_t[:, 0] += 3.0

    _, parts = crit(fs, logits_s, ls, ft, logits_t)

    assert parts["h2_target_mean"].item() >= 1.0
    assert parts["loss_anti_collapse"].item() < 1e-6


def test_asymmetric_anchor_source_aux_grad_is_zero():
    """L_aux pushes no gradient into source features (sg(feat_s))."""
    crit = DARESLoss(num_classes=2, warmup_steps=0)
    fs, ft, ls = _feats(seed=1)
    fs.requires_grad_(True)
    ft.requires_grad_(True)
    logits_s = torch.randn(2, 2, 16, 16, requires_grad=True)
    logits_t = torch.randn(2, 2, 16, 16)
    logits_t[:, 1] += 3.0

    _, parts = crit(fs, logits_s, ls, ft, logits_t)
    parts["loss_aux"].backward()

    assert torch.all(fs.grad == 0)
    assert ft.grad is not None


def test_repulsion_active_for_overlapping_target_classes():
    """Overlapping target classes produce a positive repulsion loss."""
    crit = DARESLoss(num_classes=2, warmup_steps=0, repulsion_margin=0.2)
    fs = torch.randn(2, 8, 16, 16)
    # Target features identical across the two classes -> overlap.
    ft = torch.randn(4, 8, 16, 16)
    ls = torch.randint(0, 2, (2, 16, 16))
    logits_s = torch.randn(2, 2, 16, 16)
    logits_t = torch.zeros(4, 2, 16, 16)
    logits_t[:, 0, :, :8] = 5.0  # class 0 confident on left half
    logits_t[:, 1, :, 8:] = 5.0  # class 1 confident on right half

    _, parts = crit(fs, logits_s, ls, ft, logits_t)

    assert parts["n_rep_pairs"].item() >= 1
    assert parts["loss_repulsion"].item() >= 0.0
    assert torch.isfinite(parts["delta_repulsion_mean"])


def test_trust_region_caps_aux_gradient_ratio():
    """lambda_eff <= grad_ratio * g_seg / g_aux (with EMA smoothing)."""
    crit = DARESLoss(num_classes=2, warmup_steps=0, grad_ratio=1.0)
    p = torch.nn.Parameter(torch.randn(6, 6))
    loss_seg = (p * 2.0).sum()
    loss_aux = (p * 0.2).sum()

    crit.step.fill_(crit.ramp_steps)  # fully ramped in
    lam = crit.update_lambda(loss_seg, loss_aux, [p])

    # g_aux/g_seg = 0.1, ratio = 1*0.1 -> lam = lambda_max * s * 0.1
    assert lam <= crit.lambda_max
    assert lam > 0.0


def test_isolated_fp32_under_amp():
    """Kernel math stays finite and fp32 even inside an autocast region."""
    crit = DARESLoss(num_classes=2, warmup_steps=0)
    fs, ft, ls = _feats(seed=7)
    logits_s = torch.randn(2, 2, 16, 16)
    logits_t = torch.randn(2, 2, 16, 16)

    with torch.autocast(device_type="cpu", enabled=True):
        total, parts = crit(fs, logits_s, ls, ft, logits_t)

    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    assert parts["loss_aux"].dtype == torch.float32


def test_missing_class_skipped_gracefully():
    """A class absent from the batch is skipped without NaN."""
    crit = DARESLoss(num_classes=2, min_samples=8, warmup_steps=0)
    fs = torch.randn(1, 8, 16, 16)
    ft = torch.randn(1, 8, 16, 16)
    ls = torch.zeros(1, 16, 16, dtype=torch.long)  # only class 0
    logits_s = torch.randn(1, 2, 16, 16)
    logits_t = torch.randn(1, 2, 16, 16)

    total, parts = crit(fs, logits_s, ls, ft, logits_t)

    assert torch.isfinite(total)
    assert parts["n_valid_classes"].item() <= 1


def test_update_lambda_safe_with_frozen_ref_params():
    """update_lambda must not crash on reference params with requires_grad=False
    (Phase 1, backbone frozen) even when the loss is disconnected from them."""
    crit = DARESLoss(num_classes=2, warmup_steps=0)
    # Frozen reference block (as during backbone warm-up).
    p = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
    loss_seg = (torch.randn(4, 4) @ p.detach() * 1.0).sum()  # disconnected
    loss_aux = torch.randn((), requires_grad=True)            # disconnected leaf

    lam = crit.update_lambda(loss_seg, loss_aux, [p])

    assert lam == 0.0  # no gradient anchors -> ratio 0
    assert torch.isfinite(crit.lambda_eff)


def test_step_counter_is_monotonic_across_update_lambda():
    """self.step advances by exactly one per update_lambda call (per batch)."""
    crit = DARESLoss(num_classes=2, warmup_steps=0)
    p = torch.nn.Parameter(torch.randn(4, 4), requires_grad=True)
    base = int(crit.step.item())

    for i in range(5):
        out = torch.randn(8, 4) @ p
        crit.update_lambda(out.sum(), (out * 0.5).sum(), [p])

    assert int(crit.step.item()) == base + 5
