"""
Model-vs-model significance script: per-patch Wilcoxon signed-rank + Holm.

Every listed checkpoint is scored patch-by-patch (mIoU and DICE, pixels
with label ``ignore_index`` excluded) on the same test split(s), so scores
are paired across models. Each comparison pair gets a Wilcoxon signed-rank
test; raw p-values within one table (split x metric) are Holm-Bonferroni
adjusted and flagged ``***``/``**``/``*``/``ns``.

No retraining happens here: the test runs on saved predictions. A
non-significant (``ns``) pair is a valid result and is reported as such.

Examples:
    python scripts/ttest.py --config configs/ttest/ablation_table6.yaml
    python scripts/ttest.py --config configs/ttest/baselines_table3.yaml
"""
import argparse
import json
import math
from pathlib import Path

import torch

from dares.config import TTestExperimentConfig
from dares.data.loader import DARESDataLoader
from dares.models import build_model
from dares.utils.reproducibility import set_seed
from dares.utils.ttest import (
    holm_adjust,
    per_sample_metric_scores,
    significance_stars,
    wilcoxon_signed_rank,
)


def _load_model(cfg: TTestExperimentConfig, ckpt_path: str, device: torch.device):
    """Loads one checkpoint into a fresh model of the configured arch."""
    model = build_model(cfg.model)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation of a score list."""
    n = len(xs)
    mean = math.fsum(xs) / n
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in xs) / (n - 1)) if n > 1 else 0.0
    return float(mean), float(std)


def main(config_path: str) -> None:
    """Scores every checkpoint and writes the paired comparison tables.

    Args:
        config_path (str): Path to a model-vs-model t-test YAML
            (``configs/ttest/*.yaml``).
    """
    cfg = TTestExperimentConfig.from_yaml(config_path)
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    set_seed(cfg.training.seed)
    use_amp = bool(cfg.training.use_amp and device.type == "cuda")

    out_dir = Path(cfg.ttest.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast on missing checkpoints: a table with holes is worse than none.
    missing = [
        f"{name}: {path}"
        for name, path in cfg.ttest.models.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "ttest checkpoint(s) not found:\n  " + "\n  ".join(missing)
        )

    # 1. Data (test splits only; scoring never trains or augments).
    data_manager = DARESDataLoader(cfg.data)
    split_loaders = {
        "source_test": data_manager.get_source_loaders()["test"],
        "target_test": data_manager.get_target_loaders()["test"],
    }

    # 2. Per-patch scores, paired by construction (same loader order).
    per_patch: dict[str, dict[str, dict[str, list[float]]]] = {}
    skipped: dict[str, str] = {}
    for split in cfg.ttest.splits:
        loader = split_loaders[split]
        per_patch[split] = {m: {} for m in cfg.ttest.metrics}
        for name, ckpt_path in cfg.ttest.models.items():
            print(f"Scoring {name} on {split}...")
            model = _load_model(cfg, ckpt_path, device)
            scores = per_sample_metric_scores(
                model,
                loader,
                device,
                cfg.model.num_classes,
                ignore_index=cfg.training.ignore_index,
                use_amp=use_amp,
            )
            for metric in cfg.ttest.metrics:
                per_patch[split][metric][name] = scores[metric]
        n_patches = len(next(iter(per_patch[split][cfg.ttest.metrics[0]].values())))
        if n_patches < cfg.ttest.min_samples:
            skipped[split] = (
                f"only {n_patches} patches, need >= {cfg.ttest.min_samples}"
            )
            print(f"[wilcoxon] {split} skipped: {skipped[split]}")

    with open(out_dir / "per_patch_scores.json", "w") as f:
        json.dump(per_patch, f, indent=2)

    # 3. Paired tables: one Holm family per (split, metric).
    tables: dict = {}
    for split in cfg.ttest.splits:
        if split in skipped:
            tables[split] = {"status": "skipped", "reason": skipped[split]}
            continue
        tables[split] = {}
        for metric in cfg.ttest.metrics:
            raws = [
                wilcoxon_signed_rank(
                    per_patch[split][metric][a],
                    per_patch[split][metric][b],
                    alpha=cfg.ttest.alpha,
                    alternative=cfg.ttest.alternative,
                )
                for a, b in cfg.ttest.comparisons
            ]
            p_holms = holm_adjust([r.p_value for r in raws])
            pairs = []
            for (a, b), res, p_holm in zip(cfg.ttest.comparisons, raws, p_holms):
                stars = significance_stars(p_holm)
                mean_a, std_a = _mean_std(per_patch[split][metric][a])
                mean_b, std_b = _mean_std(per_patch[split][metric][b])
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "n": res.n,
                        "n_zero": res.n_zero,
                        "mean_a": mean_a,
                        "std_a": std_a,
                        "mean_b": mean_b,
                        "std_b": std_b,
                        "mean_diff": res.mean_diff,
                        "w_statistic": res.statistic,
                        "wilcoxon_method": res.method,
                        "p_raw": res.p_value,
                        "p_holm": float(p_holm),
                        "stars": stars,
                        "significant": bool(p_holm < cfg.ttest.alpha),
                    }
                )
            summary = {
                name: {"mean": m, "std": s, "n": len(scores)}
                for name, (m, s, scores) in (
                    (name, (*_mean_std(per_patch[split][metric][name]),
                            per_patch[split][metric][name]))
                    for name in cfg.ttest.models
                )
            }
            tables[split][metric] = {
                "metric": f"per_patch_{metric}",
                "model_summary": summary,
                "pairs": pairs,
            }

    with open(out_dir / "wilcoxon_results.json", "w") as f:
        json.dump(
            {
                "test": cfg.ttest.test,
                "alpha": cfg.ttest.alpha,
                "alternative": cfg.ttest.alternative,
                "tables": tables,
            },
            f,
            indent=2,
        )

    # 4. Console report.
    for split in cfg.ttest.splits:
        if split in skipped:
            continue
        for metric in cfg.ttest.metrics:
            table = tables[split][metric]
            print(f"\n=== {split} / per-patch {metric} (Wilcoxon + Holm) ===")
            for name, s in table["model_summary"].items():
                print(f"  {name:<16} {s['mean']:.4f} +- {s['std']:.4f} (n={s['n']})")
            for p in table["pairs"]:
                print(
                    f"  {p['a']:<16} vs {p['b']:<16} "
                    f"W={p['w_statistic']:.1f} p_raw={p['p_raw']:.4g} "
                    f"p_holm={p['p_holm']:.4g} {p['stars']}"
                )
    print(f"\nResults saved in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DARES model-vs-model significance (per-patch Wilcoxon + Holm)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a model-vs-model t-test YAML (configs/ttest/*.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
