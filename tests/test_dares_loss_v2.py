"""Focused tests for the DARES v2 (MIL-CREDA hardened) alignment loss.

Covers the three upgrades over V1: the bounded class-global term built from
the mixed matrix and the conservative entropy bounds, the local
correspondence term with the personalized source reference, and the soft
class weights -- plus the V1-parity path when every v2 feature is disabled.
"""

import math

import pytest
import torch

from dares.losses.dares_loss import DARESLoss
from dares.losses.dares_loss_v2 import DARESLossV2


def _feats(n=2, h=16, w=16, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    fs = torch.randn(n, d, h, w, generator=g)
    ft = torch.randn(n, d, h, w, generator=g)
    ls = torch.randint(0, 2, (n, h, w), generator=g)
    return fs, ft, ls


def _logits_t(seed=0, n=2, h=16, w=16):
    g = torch.Generator().manual_seed(seed)
    lt = torch.randn(n, 2, h, w, generator=g)
    lt[:, 0, :, :8] += 3.0
    lt[:, 1, :, 8:] += 3.0
    return lt


def test_v2_requires_at_least_two_classes():
    """The log2(C) confidence normalizer fails fast on a degenerate head."""
    with pytest.raises(ValueError, match="at least two classes"):
        DARESLossV2(num_classes=1, warmup_steps=0)


def test_bounded_global_in_unit_interval():
    """With bounded_align the global term lands in (0, 1] per class."""
    crit = DARESLossV2(num_classes=2, warmup_steps=0, bounded_align=True)
    fs, ft, ls = _feats(seed=5)
    total, parts = crit(fs, torch.randn(2, 2, 16, 16), ls, ft, _logits_t(seed=6))

    assert parts["n_valid_classes"].item() >= 1
    assert 0.0 < parts["loss_align"].item() <= 1.0 + 1e-6
    assert torch.isfinite(total)


def test_bounded_global_keeps_gradient_at_alignment():
    """At delta == 0 the ReLU(V1) dies; the affine bound map keeps pulling."""
    crit = DARESLossV2(num_classes=2, warmup_steps=0, bounded_align=True)
    s2 = torch.tensor(4.0, requires_grad=True)
    tr = torch.tensor(20.0)
    h = crit._h2_bits(s2.detach(), tr)
    loss_c, delta_c = crit._bounded_global(
        s2.detach(), tr, s2, tr, s2.detach(), h, h, n_s=16.0, n_eff_t=16.0
    )
    # Aligned clouds (s2_st = s2_t, equal traces): delta is exactly 0...
    assert delta_c.item() == pytest.approx(0.0, abs=1e-5)
    # ...but the bounded loss keeps a positive value (U / (U - L)) and gradient.
    assert loss_c.item() == pytest.approx(4.0 / 9.0, rel=1e-5)
    loss_c.backward()
    assert s2.grad is not None
    assert s2.grad.item() != 0.0


def test_bounded_global_disabled_matches_relu_radius():
    """bounded_align=False routes through the V1 information radius."""
    crit = DARESLossV2(num_classes=2, warmup_steps=0, bounded_align=False)
    s2 = torch.tensor(4.0, requires_grad=True)
    tr = torch.tensor(20.0)
    loss_c, _ = crit._bounded_global(
        s2.detach(), tr, s2, tr, s2.detach(), torch.tensor(2.0), torch.tensor(2.0),
        n_s=16.0, n_eff_t=16.0,
    )
    assert loss_c.item() == pytest.approx(0.0, abs=1e-6)  # relu(0) = 0


def test_local_term_matches_manual_equations():
    """Eqs. (28)-(31) recomputed by hand reproduce the aggregated loss."""
    crit = DARESLossV2(num_classes=2, warmup_steps=0, tau_local=0.5)
    xs = torch.tensor([[0.0], [1.0], [2.0]])
    xt = torch.tensor([[0.0], [1.9]], requires_grad=True)
    sigma = torch.tensor(1.0)
    K_s = crit._rbf(xs, xs, sigma)
    K_st = crit._rbf(xs, xt, sigma)
    wc = torch.tensor([0.8, 0.3])

    loss = crit._local_term(K_s, K_st, wc)

    kts = K_st.transpose(0, 1)
    pi = torch.softmax(kts / 0.5, dim=1)
    assert torch.allclose(pi.sum(dim=1), torch.ones(2), atol=1e-6)
    d2 = 1.0 - 2.0 * (pi * kts).sum(1) + (pi * (pi @ K_s)).sum(1)
    expected = ((0.5 * d2).clamp_min(0.0) * wc).sum() / (wc.sum() + crit.eps)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert 0.0 <= loss.item() <= 1.0
    loss.backward()
    assert xt.grad is not None
    assert bool((xt.grad != 0).any())


def test_local_term_vanishes_for_exact_reference():
    """A target pixel sitting on a source pixel has ~zero distance (tau -> 0)."""
    crit = DARESLossV2(num_classes=2, warmup_steps=0, tau_local=1e-3)
    xs = torch.tensor([[0.0], [3.0], [7.0]])
    xt = torch.tensor([[3.0]])
    sigma = torch.tensor(2.0)
    loss = crit._local_term(
        crit._rbf(xs, xs, sigma), crit._rbf(xs, xt, sigma), torch.tensor([1.0])
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_soft_class_weights_multiply_confidence():
    """Soft weights scale the Eq. (24) confidence by the class mass g_{j,c}."""
    pt = torch.tensor([[0.9, 0.1], [0.55, 0.45]])
    w_t = torch.tensor([1.0, 0.35])
    idx = torch.tensor([0, 1])

    soft = DARESLossV2(num_classes=2, warmup_steps=0, soft_class_weights=True)
    hard = DARESLossV2(num_classes=2, warmup_steps=0, soft_class_weights=False)

    assert torch.allclose(soft._class_weights(pt, w_t, idx, 0), w_t * pt[:, 0])
    assert torch.allclose(hard._class_weights(pt, w_t, idx, 0), w_t)


def test_v2_reduces_to_v1_when_all_upgrades_disabled():
    """local=0, bounded=False, soft=False (with v1's ac/rep weights) == V1."""
    fs, ft, ls = _feats(seed=5)
    logits_s = torch.randn(2, 2, 16, 16)
    lt = _logits_t(seed=6)

    torch.manual_seed(42)
    _, p1 = DARESLoss(num_classes=2, warmup_steps=0)(fs, logits_s, ls, ft, lt)
    torch.manual_seed(42)
    _, p2 = DARESLossV2(
        num_classes=2,
        warmup_steps=0,
        beta=1.0,
        gamma=0.5,
        lambda_local=0.0,
        bounded_align=False,
        soft_class_weights=False,
    )(fs, logits_s, ls, ft, lt)

    for key in ("loss_align", "loss_anti_collapse", "loss_repulsion", "loss_em"):
        assert p2[key].item() == pytest.approx(p1[key].item(), rel=1e-5), key
    assert p2["loss_aux"].item() == pytest.approx(p1["loss_aux"].item(), rel=1e-5)
    assert p2["loss_local"].item() == 0.0


def test_v2_defaults_are_lean():
    """The MIL-CREDA essence: ac floor and repulsion are opt-in in v2."""
    crit = DARESLossV2(num_classes=2)
    assert crit.beta == 0.0
    assert crit.gamma == 0.0
    assert crit.lambda_local == 0.5
    assert crit.bounded_align is True
    assert crit.soft_class_weights is True


def test_gated_terms_report_zero_and_are_not_computed():
    """beta=gamma=lambda_local=0 -> exact-zero diagnostics, aux == align only."""
    crit = DARESLossV2(
        num_classes=2, warmup_steps=0, beta=0.0, gamma=0.0, lambda_local=0.0
    )
    fs, ft, ls = _feats(seed=4)
    _, parts = crit(fs, torch.randn(2, 2, 16, 16), ls, ft, _logits_t(seed=5))

    assert parts["n_valid_classes"].item() >= 1
    assert parts["loss_anti_collapse"].item() == 0.0
    assert parts["loss_repulsion"].item() == 0.0
    assert parts["loss_local"].item() == 0.0
    assert parts["n_rep_pairs"].item() == 0
    assert torch.isfinite(parts["delta_repulsion_mean"])  # 0.0, not NaN
    assert parts["loss_aux"].item() == pytest.approx(
        crit.lambda_align * parts["loss_align"].item(), rel=1e-6
    )


def test_local_term_weighted_into_aux():
    """loss_aux carries lambda_local * loss_local alongside the other terms."""
    crit = DARESLossV2(
        num_classes=2, warmup_steps=0,
        lambda_align=0.0, beta=0.0, gamma=0.0, lambda_local=2.0,
    )
    fs, ft, ls = _feats(seed=9)
    _, parts = crit(fs, torch.randn(2, 2, 16, 16), ls, ft, _logits_t(seed=10))

    assert parts["loss_aux"].item() == pytest.approx(
        2.0 * parts["loss_local"].item(), rel=1e-6
    )
    assert parts["loss_local"].item() >= 0.0


def test_normalize_seg_scales_by_supremum():
    """normalize_seg divides the CE by its exact bound ln(1 + 1/eps)."""
    fs, ft, ls = _feats(seed=2)
    logits_s = torch.randn(2, 2, 16, 16)
    lt = _logits_t(seed=3)

    _, raw = DARESLossV2(num_classes=2, warmup_steps=0, normalize_seg=False)(
        fs, logits_s, ls, ft, lt
    )
    _, norm = DARESLossV2(num_classes=2, warmup_steps=0, normalize_seg=True)(
        fs, logits_s, ls, ft, lt
    )
    bound = math.log(1.0 + 1.0 / 1e-8)
    assert norm["loss_seg"].item() == pytest.approx(raw["loss_seg"].item() / bound, rel=1e-5)


def test_source_gets_no_auxiliary_gradient():
    """The asymmetric anchoring is preserved: no v2 term trains the source."""
    crit = DARESLossV2(num_classes=2, warmup_steps=0)
    fs, ft, ls = _feats(seed=1)
    fs.requires_grad_(True)
    ft.requires_grad_(True)
    logits_s = torch.randn(2, 2, 16, 16, requires_grad=True)
    lt = _logits_t(seed=6)

    _, parts = crit(fs, logits_s, ls, ft, lt)
    parts["loss_aux"].backward()

    # Lean v2: with beta=gamma=0 no path reaches the source at all (grad None);
    # with the safeguards active the path exists but carries exact zeros.
    assert fs.grad is None or torch.all(fs.grad == 0)
    assert ft.grad is not None


def test_v2_is_amp_safe_and_fp32():
    crit = DARESLossV2(num_classes=2, warmup_steps=0)
    fs, ft, ls = _feats(seed=7)
    with torch.autocast(device_type="cpu", enabled=True):
        total, parts = crit(fs, torch.randn(2, 2, 16, 16), ls, ft, _logits_t(seed=8))
    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    assert parts["loss_local"].dtype == torch.float32


def test_missing_class_skipped_gracefully():
    """A single-class batch still yields finite v2 terms."""
    crit = DARESLossV2(num_classes=2, min_samples=8, warmup_steps=0)
    fs = torch.randn(1, 8, 16, 16)
    ft = torch.randn(1, 8, 16, 16)
    ls = torch.zeros(1, 16, 16, dtype=torch.long)
    logits_s = torch.randn(1, 2, 16, 16)
    lt = torch.randn(1, 2, 16, 16)
    lt[:, 0] += 3.0

    total, parts = crit(fs, logits_s, ls, ft, lt)

    assert torch.isfinite(total)
    assert parts["n_valid_classes"].item() <= 1
    assert torch.isfinite(parts["loss_local"])


def test_update_lambda_is_inherited():
    """The GradNorm trust region machinery is shared with V1 unchanged."""
    crit = DARESLossV2(num_classes=2, warmup_steps=5)
    p = torch.nn.Parameter(torch.randn(4, 4))
    lam = crit.update_lambda((p * 2.0).sum(), (p * 0.2).sum(), [p])
    assert lam == 0.0
    assert int(crit.step.item()) == 1
