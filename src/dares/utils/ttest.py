"""Paired significance utilities for DARES segmentation experiments.

Compares per-patch score distributions without requiring SciPy:

* ``welch_ttest`` -- independent samples, unequal variances (domain gap:
  source-test vs target-test patches).
* ``student_ttest`` -- independent samples, pooled variance.
* ``paired_ttest`` -- same patches scored by two checkpoints
  (``evaluate.py --compare``).
* ``wilcoxon_signed_rank`` -- paired non-parametric test on per-patch
  scores (``scripts/ttest.py`` model-vs-model tables). mIoU/DICE per patch
  are bounded and skewed, so normality cannot be assumed.
* :func:`per_sample_miou_scores` -- collects one mIoU per patch so the
  tests above have i.i.d. samples (global confusion-matrix metrics are a
  single point and cannot feed a significance test).
* :func:`per_sample_metric_scores` -- collects per-patch mIoU and DICE.
* :func:`run_ttest` -- dispatcher honoring ``StatsConfig`` (``min_samples``,
  ``alpha``, ``alternative``).
* :func:`holm_adjust` -- Holm-Bonferroni step-down correction applied
  within each comparison table.
* :func:`significance_stars` -- ``***``/``**``/``*``/``ns`` markers.

The two-sided Student-t p-value uses the exact tail via the regularized
incomplete beta function (no SciPy needed)::

    p_two_sided = I_{nu/(nu+t^2)}(nu/2, 1/2)

The Wilcoxon p-value is exact (signed-rank sum distribution via dynamic
programming) for tie-free samples with ``n <= 50`` and uses the
tie-corrected normal approximation with continuity correction otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from torch.amp import autocast

TTestKind = Literal["welch", "student", "paired"]
Alternative = Literal["two-sided", "greater", "less"]


@dataclass
class TTestResult:
    """Outcome of a single t-test."""

    test: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_diff: float
    t_stat: float
    dof: float
    p_value: float
    alpha: float
    alternative: str
    ci_low: float
    ci_high: float
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (plain floats/ints/bools)."""
        return {
            "test": self.test,
            "n_a": int(self.n_a),
            "n_b": int(self.n_b),
            "mean_a": float(self.mean_a),
            "mean_b": float(self.mean_b),
            "mean_diff": float(self.mean_diff),
            "t_stat": float(self.t_stat),
            "dof": float(self.dof),
            "p_value": float(self.p_value),
            "alpha": float(self.alpha),
            "alternative": str(self.alternative),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "significant": bool(self.significant),
        }


# ---------------------------------------------------------------------------
# Incomplete beta (Numerical Recipes / Cephes) -- pure stdlib.
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation for the incomplete beta function."""
    max_it, eps, fpmin = 200, 3.0e-7, 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = c * d
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` in ``[0, 1]``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"betai requires a,b > 0, got {a}, {b}.")
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t: float, dof: float) -> float:
    """Two-sided Student-t p-value for ``|T| >= |t|``."""
    if dof <= 0.0:
        raise ValueError(f"dof must be > 0, got {dof}.")
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    x = dof / (dof + t * t)
    return max(0.0, min(1.0, _betai(dof / 2.0, 0.5, x)))


def _t_one_sided_p(t: float, dof: float, alternative: Alternative) -> float:
    """One/two-sided p-value honoring ``alternative``."""
    two = _t_two_sided_p(t, dof)
    if alternative == "two-sided":
        return two
    half = two / 2.0
    if alternative == "greater":  # H1: diff > 0  <=> t > 0
        return half if t > 0 else 1.0 - half
    if alternative == "less":  # H1: diff < 0  <=> t < 0
        return half if t < 0 else 1.0 - half
    raise ValueError(f"unknown alternative {alternative!r}.")


def _t_crit_two_sided(alpha: float, dof: float) -> float:
    """Critical value with two-sided tail ``alpha`` (for CIs)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    lo, hi = 0.0, 1.0
    while _t_two_sided_p(hi, dof) > alpha:
        hi *= 2.0
        if hi > 1e6:
            return float("inf")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _t_two_sided_p(mid, dof) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Core statistics (stdlib only; accepts torch tensors or sequences).
# ---------------------------------------------------------------------------


def _as_floats(xs: Sequence[float] | torch.Tensor) -> list[float]:
    if isinstance(xs, torch.Tensor):
        return [float(v) for v in xs.detach().cpu().reshape(-1).tolist()]
    return [float(v) for v in xs]


def _mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("need at least one sample.")
    return math.fsum(xs) / len(xs)


def _var_ddof1(xs: list[float], mean: float) -> float:
    n = len(xs)
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}.")
    return math.fsum((v - mean) ** 2 for v in xs) / (n - 1)


def welch_ttest(
    a: Sequence[float] | torch.Tensor,
    b: Sequence[float] | torch.Tensor,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> TTestResult:
    """Welch's independent-samples t-test (unequal variances)."""
    xa, xb = _as_floats(a), _as_floats(b)
    if len(xa) < 2 or len(xb) < 2:
        raise ValueError("welch_ttest needs >= 2 samples per group.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    ma, mb = _mean(xa), _mean(xb)
    va, vb = _var_ddof1(xa, ma), _var_ddof1(xb, mb)
    na, nb = len(xa), len(xb)
    diff = ma - mb
    se2 = va / na + vb / nb
    if se2 <= 0.0:  # both groups constant
        t = 0.0 if diff == 0.0 else math.inf * math.copysign(1.0, diff)
        dof = float(na + nb - 2)
        p = 1.0 if diff == 0.0 else 0.0
        half = 0.0
    else:
        se = math.sqrt(se2)
        t = diff / se
        num = se2 * se2
        den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        dof = num / den if den > 0.0 else 1.0
        p = _t_one_sided_p(t, dof, alternative)
        half = _t_crit_two_sided(alpha, dof) * se
    sig = bool(p < alpha) if not math.isnan(p) else False
    return TTestResult(
        test="welch",
        n_a=na,
        n_b=nb,
        mean_a=ma,
        mean_b=mb,
        mean_diff=diff,
        t_stat=t,
        dof=float(dof),
        p_value=float(p),
        alpha=float(alpha),
        alternative=str(alternative),
        ci_low=float(diff - half),
        ci_high=float(diff + half),
        significant=sig,
    )


def student_ttest(
    a: Sequence[float] | torch.Tensor,
    b: Sequence[float] | torch.Tensor,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> TTestResult:
    """Student's independent-samples t-test (pooled variance)."""
    xa, xb = _as_floats(a), _as_floats(b)
    if len(xa) < 2 or len(xb) < 2:
        raise ValueError("student_ttest needs >= 2 samples per group.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    ma, mb = _mean(xa), _mean(xb)
    va, vb = _var_ddof1(xa, ma), _var_ddof1(xb, mb)
    na, nb = len(xa), len(xb)
    diff = ma - mb
    dof = float(na + nb - 2)
    pooled = ((na - 1) * va + (nb - 1) * vb) / dof if dof > 0 else 0.0
    se2 = pooled * (1.0 / na + 1.0 / nb)
    if se2 <= 0.0:
        t = 0.0 if diff == 0.0 else math.inf * math.copysign(1.0, diff)
        p = 1.0 if diff == 0.0 else 0.0
        half = 0.0
    else:
        se = math.sqrt(se2)
        t = diff / se
        p = _t_one_sided_p(t, dof, alternative)
        half = _t_crit_two_sided(alpha, dof) * se
    sig = bool(p < alpha) if not math.isnan(p) else False
    return TTestResult(
        test="student",
        n_a=na,
        n_b=nb,
        mean_a=ma,
        mean_b=mb,
        mean_diff=diff,
        t_stat=t,
        dof=float(dof),
        p_value=float(p),
        alpha=float(alpha),
        alternative=str(alternative),
        ci_low=float(diff - half),
        ci_high=float(diff + half),
        significant=sig,
    )


def paired_ttest(
    a: Sequence[float] | torch.Tensor,
    b: Sequence[float] | torch.Tensor,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> TTestResult:
    """Paired t-test on per-patch differences ``a - b`` (same patches)."""
    xa, xb = _as_floats(a), _as_floats(b)
    if len(xa) != len(xb):
        raise ValueError(
            f"paired_ttest needs equal lengths, got {len(xa)} vs {len(xb)}."
        )
    if len(xa) < 2:
        raise ValueError("paired_ttest needs >= 2 pairs.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    diffs = [u - v for u, v in zip(xa, xb)]
    md = _mean(diffs)
    vd = _var_ddof1(diffs, md)
    n = len(diffs)
    dof = float(n - 1)
    if vd <= 0.0:
        t = 0.0 if md == 0.0 else math.inf * math.copysign(1.0, md)
        p = 1.0 if md == 0.0 else 0.0
        half = 0.0
    else:
        se = math.sqrt(vd / n)
        t = md / se
        p = _t_one_sided_p(t, dof, alternative)
        half = _t_crit_two_sided(alpha, dof) * se
    sig = bool(p < alpha) if not math.isnan(p) else False
    return TTestResult(
        test="paired",
        n_a=n,
        n_b=n,
        mean_a=_mean(xa),
        mean_b=_mean(xb),
        mean_diff=md,
        t_stat=t,
        dof=float(dof),
        p_value=float(p),
        alpha=float(alpha),
        alternative=str(alternative),
        ci_low=float(md - half),
        ci_high=float(md + half),
        significant=sig,
    )


def run_ttest(
    a: Sequence[float] | torch.Tensor,
    b: Sequence[float] | torch.Tensor,
    test: TTestKind = "welch",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    min_samples: int = 2,
) -> TTestResult:
    """Dispatches to the requested t-test after a ``min_samples`` guard."""
    xa, xb = _as_floats(a), _as_floats(b)
    if test == "paired" and len(xa) != len(xb):
        raise ValueError(
            f"paired test needs equal lengths, got {len(xa)} vs {len(xb)}."
        )
    need = max(int(min_samples), 2)
    if len(xa) < need or len(xb) < need:
        raise ValueError(
            f"ttest needs >= {need} samples per group, "
            f"got {len(xa)} vs {len(xb)}."
        )
    if test == "welch":
        return welch_ttest(xa, xb, alpha=alpha, alternative=alternative)
    if test == "student":
        return student_ttest(xa, xb, alpha=alpha, alternative=alternative)
    if test == "paired":
        return paired_ttest(xa, xb, alpha=alpha, alternative=alternative)
    raise ValueError(f"unknown t-test {test!r}.")


# ---------------------------------------------------------------------------
# Per-patch scores + reporting helpers.
# ---------------------------------------------------------------------------


@torch.no_grad()
def per_sample_miou_scores(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    use_amp: bool = False,
) -> list[float]:
    """Collects one mIoU per patch (respects ``ignore_index``).

    Args:
        model: Segmentation model (``mode='class'``).
        loader: Labeled loader (test split).
        device: Computing device.
        num_classes: Number of output classes.
        ignore_index: Label value excluded from the per-patch IoU.
        use_amp: Whether to run inference under AMP.

    Returns:
        List of per-patch mIoU floats in loader order (one entry per patch
        with at least one labeled pixel batch; unlabeled batches are skipped).
    """
    from dares.utils.metrics import MetricTracker

    model.eval()
    scores: list[float] = []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1]
        if labels is None:
            continue
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs, mode="class")
        preds = torch.argmax(logits, dim=1).cpu()
        labs = labels.long().cpu()
        for i in range(preds.shape[0]):
            _, miou = MetricTracker.compute_iou(
                preds[i].reshape(-1),
                labs[i].reshape(-1),
                num_classes,
                ignore_index=ignore_index,
            )
            scores.append(float(miou))
    return scores


def format_result(result: TTestResult) -> str:
    """One-line human summary (also printed by train/evaluate scripts)."""
    flag = "SIGNIFICANT" if result.significant else "not significant"
    return (
        f"[t-test:{result.test}] n={result.n_a} vs {result.n_b} "
        f"means {result.mean_a:.4f} vs {result.mean_b:.4f} "
        f"diff {result.mean_diff:+.4f} [{result.ci_low:+.4f}, {result.ci_high:+.4f}] "
        f"t={result.t_stat:.3f} dof={result.dof:.1f} "
        f"p={result.p_value:.4g} alpha={result.alpha} ({flag}, {result.alternative})"
    )


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test (paired, non-parametric) + Holm correction.
# ---------------------------------------------------------------------------


@dataclass
class WilcoxonResult:
    """Outcome of a single paired Wilcoxon signed-rank test (raw p-value)."""

    test: str
    n: int  # number of non-zero paired differences
    n_zero: int  # number of zero differences (discarded, per Wilcoxon)
    mean_a: float
    mean_b: float
    mean_diff: float
    w_pos: float  # sum of ranks of positive differences
    w_neg: float  # sum of ranks of negative differences
    statistic: float  # W+ (sum of positive ranks)
    p_value: float  # raw (uncorrected) p-value
    alpha: float
    alternative: str
    method: str  # "exact" | "normal"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (plain floats/ints/strs)."""
        return {
            "test": str(self.test),
            "n": int(self.n),
            "n_zero": int(self.n_zero),
            "mean_a": float(self.mean_a),
            "mean_b": float(self.mean_b),
            "mean_diff": float(self.mean_diff),
            "w_pos": float(self.w_pos),
            "w_neg": float(self.w_neg),
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "alpha": float(self.alpha),
            "alternative": str(self.alternative),
            "method": str(self.method),
        }


def _average_ranks(values: list[float]) -> tuple[list[float], list[int]]:
    """1-based average ranks of ``values`` plus the tie-group sizes."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_groups: list[int] = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        if j > i:
            tie_groups.append(j - i + 1)
        i = j + 1
    return ranks, tie_groups


def _wilcoxon_exact_p(w_obs: float, n: int, alternative: Alternative) -> float:
    """Exact Wilcoxon p-value (tie-free ranks are exactly ``1..n``)."""
    max_s = n * (n + 1) // 2
    dp = [0] * (max_s + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for s in range(max_s, i - 1, -1):
            dp[s] += dp[s - i]
    total = float(1 << n)
    w = int(round(w_obs))
    lo = sum(dp[s] for s in range(0, min(w, max_s) + 1)) / total
    hi = sum(dp[s] for s in range(max(w, 0), max_s + 1)) / total
    if alternative == "two-sided":
        return max(0.0, min(1.0, 2.0 * min(lo, hi)))
    if alternative == "greater":  # H1: median(a - b) > 0
        return max(0.0, min(1.0, hi))
    if alternative == "less":  # H1: median(a - b) < 0
        return max(0.0, min(1.0, lo))
    raise ValueError(f"unknown alternative {alternative!r}.")


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via ``erf`` (stdlib only)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _wilcoxon_normal_p(
    w_obs: float, n: int, tie_groups: list[int], alternative: Alternative
) -> float:
    """Tie-corrected normal approximation with continuity correction."""
    mean = n * (n + 1) / 4.0
    var = (
        n * (n + 1) * (2 * n + 1) / 24.0
        - sum(t**3 - t for t in tie_groups) / 48.0
    )
    if var <= 0.0:
        return 1.0 if w_obs == mean else 0.0
    sd = math.sqrt(var)
    if w_obs > mean:
        z = (w_obs - 0.5 - mean) / sd
    elif w_obs < mean:
        z = (w_obs + 0.5 - mean) / sd
    else:
        z = 0.0
    if alternative == "two-sided":
        return max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z)))))
    if alternative == "greater":
        return max(0.0, min(1.0, 1.0 - _normal_cdf(z)))
    if alternative == "less":
        return max(0.0, min(1.0, _normal_cdf(z)))
    raise ValueError(f"unknown alternative {alternative!r}.")


_EXACT_WILCOXON_MAX_N = 50


def wilcoxon_signed_rank(
    a: Sequence[float] | torch.Tensor,
    b: Sequence[float] | torch.Tensor,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test on per-patch scores.

    Scores must come from the same patches in the same order (paired).
    Zero differences are discarded, per the Wilcoxon definition. Pixels
    with label ``ignore_index`` must already be excluded upstream (see
    :func:`per_sample_metric_scores`).

    Args:
        a: Scores of model A (one per patch).
        b: Scores of model B (same patches, same order).
        alpha: Significance level (stored for reporting; the returned
            p-value is raw/uncorrected).
        alternative: ``"two-sided"`` (medians differ), ``"greater"``
            (median of ``a - b`` > 0) or ``"less"``.

    Returns:
        WilcoxonResult with the raw p-value and ``method`` (``"exact"``
        for tie-free ``n <= 50``, ``"normal"`` otherwise).
    """
    xa, xb = _as_floats(a), _as_floats(b)
    if len(xa) != len(xb):
        raise ValueError(
            f"wilcoxon needs equal lengths, got {len(xa)} vs {len(xb)}."
        )
    if len(xa) < 2:
        raise ValueError("wilcoxon needs >= 2 pairs.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    diffs = [u - v for u, v in zip(xa, xb)]
    n_zero = sum(1 for d in diffs if d == 0.0)
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    ma, mb = _mean(xa), _mean(xb)
    if n == 0:  # identical scores: no evidence against H0
        return WilcoxonResult(
            test="wilcoxon",
            n=0,
            n_zero=n_zero,
            mean_a=ma,
            mean_b=mb,
            mean_diff=0.0,
            w_pos=0.0,
            w_neg=0.0,
            statistic=0.0,
            p_value=1.0,
            alpha=float(alpha),
            alternative=str(alternative),
            method="exact",
        )
    ranks, tie_groups = _average_ranks([abs(d) for d in nz])
    w_pos = math.fsum(r for d, r in zip(nz, ranks) if d > 0.0)
    w_neg = math.fsum(r for d, r in zip(nz, ranks) if d < 0.0)
    if not tie_groups and n <= _EXACT_WILCOXON_MAX_N:
        p = _wilcoxon_exact_p(w_pos, n, alternative)
        method = "exact"
    else:
        p = _wilcoxon_normal_p(w_pos, n, tie_groups, alternative)
        method = "normal"
    return WilcoxonResult(
        test="wilcoxon",
        n=n,
        n_zero=n_zero,
        mean_a=ma,
        mean_b=mb,
        mean_diff=ma - mb,
        w_pos=float(w_pos),
        w_neg=float(w_neg),
        statistic=float(w_pos),
        p_value=float(p),
        alpha=float(alpha),
        alternative=str(alternative),
        method=method,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (original order).

    Args:
        p_values: Raw p-values of one comparison table (one split × one
            metric; e.g. 10 ablation pairs or 3 baseline pairs).

    Returns:
        Adjusted p-values in the input order, monotonically enforced and
        capped at 1.0.
    """
    p = [float(v) for v in p_values]
    if any(not 0.0 <= v <= 1.0 or math.isnan(v) for v in p):
        raise ValueError("holm_adjust needs p-values in [0, 1].")
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def significance_stars(p_value: float) -> str:
    """``***``/``**``/``*``/``ns`` marker for an (adjusted) p-value."""
    p = float(p_value)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


@torch.no_grad()
def per_sample_metric_scores(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    use_amp: bool = False,
) -> dict[str, list[float]]:
    """Collects per-patch mIoU and DICE (``ignore_index`` pixels excluded).

    Args:
        model: Segmentation model (``mode='class'``).
        loader: Labeled loader (test split).
        device: Computing device.
        num_classes: Number of output classes.
        ignore_index: Label value excluded from the per-patch metrics
            (``255`` for water/NoData pixels).
        use_amp: Whether to run inference under AMP.

    Returns:
        ``{"miou": [...], "dice": [...]}`` with one entry per patch in
        loader order (unlabeled batches are skipped).
    """
    from dares.utils.metrics import MetricTracker

    model.eval()
    miou_scores: list[float] = []
    dice_scores: list[float] = []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1]
        if labels is None:
            continue
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs, mode="class")
        preds = torch.argmax(logits, dim=1).cpu()
        labs = labels.long().cpu()
        for i in range(preds.shape[0]):
            _, miou = MetricTracker.compute_iou(
                preds[i].reshape(-1),
                labs[i].reshape(-1),
                num_classes,
                ignore_index=ignore_index,
            )
            _, dice = MetricTracker.compute_dice(
                preds[i].reshape(-1),
                labs[i].reshape(-1),
                num_classes,
                ignore_index=ignore_index,
            )
            miou_scores.append(float(miou))
            dice_scores.append(float(dice))
    return {"miou": miou_scores, "dice": dice_scores}
