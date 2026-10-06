"""Tests for the per-patch mIoU t-test module and its script wiring."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dares.config import ExperimentConfig  # noqa: E402
from dares.utils.ttest import (  # noqa: E402
    paired_ttest,
    per_sample_miou_scores,
    run_ttest,
    welch_ttest,
)


def test_welch_identical_distributions_not_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(0.7, 0.05, size=30).tolist()
    res = welch_ttest(x, list(x))
    assert res.t_stat == pytest.approx(0.0, abs=1e-9)
    assert res.p_value == pytest.approx(1.0, abs=1e-9)
    assert res.significant is False
    assert res.ci_low <= 0.0 <= res.ci_high


def test_welch_separated_distributions_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(0.8, 0.03, size=30).tolist()
    b = rng.normal(0.5, 0.03, size=30).tolist()
    res = welch_ttest(a, b)
    assert res.p_value < 0.05
    assert res.significant is True
    assert res.mean_diff > 0.2


def test_paired_detects_consistent_gain():
    base = [0.60, 0.62, 0.58, 0.61, 0.59, 0.63, 0.60, 0.62]
    improved = [v + 0.05 for v in base]
    res = paired_ttest(improved, base)
    assert res.mean_diff == pytest.approx(0.05, abs=1e-9)
    assert res.p_value < 0.05
    assert res.significant is True


def test_paired_identical_scores_p_one():
    x = [0.7] * 8
    res = paired_ttest(x, list(x))
    assert res.t_stat == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)
    assert res.significant is False


def test_zero_variance_different_constants_is_significant():
    res = welch_ttest([0.8] * 6, [0.5] * 6)
    assert res.p_value == pytest.approx(0.0)
    assert res.significant is True
    assert abs(res.t_stat) > 1e6


def test_run_ttest_min_samples_guard():
    with pytest.raises(ValueError, match=">= 4 samples"):
        run_ttest([0.7, 0.6, 0.5], [0.7, 0.6, 0.5], min_samples=4)


def test_run_ttest_paired_length_mismatch():
    with pytest.raises(ValueError, match="equal lengths"):
        run_ttest([0.7, 0.6], [0.7], test="paired")


def test_run_ttest_unknown_kind():
    with pytest.raises(ValueError, match="unknown t-test"):
        run_ttest([0.7, 0.6, 0.5], [0.6, 0.5, 0.4], test="bogus")


def test_run_ttest_alternatives_agree_on_two_sided():
    a = [0.75, 0.78, 0.72, 0.76, 0.74, 0.77]
    b = [0.70, 0.71, 0.69, 0.72, 0.70, 0.71]
    two = run_ttest(a, b, test="welch", alternative="two-sided")
    greater = run_ttest(a, b, test="welch", alternative="greater")
    assert greater.p_value == pytest.approx(two.p_value / 2.0, rel=1e-6)


def test_per_sample_scores_respect_ignore_index():
    class ConstModel(torch.nn.Module):
        def forward(self, x, mode="class"):
            b, _, h, w = x.shape
            return torch.zeros(b, 2, h, w)  # always predicts class 0

    imgs = torch.randn(2, 4, 8, 8)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    labels[0, :4, :] = 255  # half the first patch is ignored
    loader = [[imgs, labels]]
    scores = per_sample_miou_scores(
        ConstModel(), loader, torch.device("cpu"), 2, ignore_index=255
    )
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_stats_config_defaults_disabled_and_parses_enabled():
    cfg_dict = {
        "data": {
            "source_dir": ".",
            "target_dir": ".",
        },
        "model": {},
        "training": {},
        "experiment": {},
    }
    cfg = ExperimentConfig(**cfg_dict)
    assert cfg.stats.enabled is False
    assert cfg.stats.test == "welch"
    assert cfg.stats.alpha == pytest.approx(0.05)
    assert cfg.stats.min_samples == 8

    cfg_dict["stats"] = {
        "enabled": True,
        "test": "paired",
        "alpha": 0.01,
        "min_samples": 3,
        "alternative": "greater",
    }
    cfg = ExperimentConfig(**cfg_dict)
    assert cfg.stats.enabled is True
    assert cfg.stats.test == "paired"
    assert cfg.stats.alpha == pytest.approx(0.01)


def _make_h5(path: Path, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 2, size=(n, 32, 32)).astype(np.uint8)
    images = rng.random((n, 4, 32, 32)).astype(np.float32)
    images[:, 0] = masks
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images, compression="lzf",
                         chunks=(1, 4, 32, 32))
        f.create_dataset("masks", data=masks, compression="lzf",
                         chunks=(1, 32, 32))


def _write_config(tmp_path: Path, out_dir: str, **stats) -> Path:
    for domain in ("source", "target"):
        for split, n in (("train", 6), ("val", 3), ("test", 8)):
            _make_h5(tmp_path / f"{domain}_{split}.h5", n=n)
    config = {
        "data": {
            "source_dir": str(tmp_path),
            "target_dir": str(tmp_path),
            "batch_size": 2,
            "patch_size": 32,
            "num_workers": 0,
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "use_augmentation": False,
        },
        "model": {
            "backbone": "resnet50",
            "head": "resunet",
            "in_channels": 4,
            "num_classes": 2,
            "pretrained": False,
            "dropout_rate": 0.0,
        },
        "training": {
            "method": "source_only",
            "epochs": 1,
            "lr": 1e-4,
            "device": "cpu",
            "use_amp": False,
            "seed": 42,
        },
        "experiment": {
            "name": "test_ttest",
            "version": 1,
            "output_dir": out_dir,
            "save_results": True,
        },
        "stats": {
            "enabled": True,
            "test": "welch",
            "alpha": 0.05,
            "min_samples": 3,
            "alternative": "two-sided",
            **stats,
        },
    }
    path = tmp_path / "config_ttest.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def test_train_script_does_not_write_ttest_json(tmp_path):
    """train.py no longer runs significance tests (model-vs-model lives on)."""
    from scripts.train import main as train_main

    out_dir = str(tmp_path / "outputs")
    config_path = _write_config(tmp_path, out_dir)
    train_main(str(config_path))
    assert not (Path(out_dir) / "test_ttest.json").exists()
    assert (Path(out_dir) / "test_metrics.json").is_file()


def test_evaluate_script_writes_domain_and_paired_ttest(tmp_path):
    from scripts.evaluate import main as evaluate_main
    from scripts.train import main as train_main

    out_dir = str(tmp_path / "outputs")
    config_path = _write_config(tmp_path, out_dir)
    train_main(str(config_path))
    model_path = str(Path(out_dir) / "model_final.pth")
    eval_dir = str(tmp_path / "evaluation")
    # Paired vs itself: identical scores -> p == 1, not significant.
    evaluate_main(str(config_path), model_path, output_dir=eval_dir,
                  compare_model=model_path)
    ttest_path = Path(eval_dir) / "evaluation_ttest.json"
    assert ttest_path.is_file()
    with open(ttest_path) as f:
        payload = json.load(f)
    assert "source_vs_target" in payload
    assert "model_vs_compare" in payload
    assert payload["model_vs_compare"]["p_value"] == pytest.approx(1.0)
    assert payload["model_vs_compare"]["significant"] is False


def test_wilcoxon_exact_known_value():
    """All-positive diffs 1..5 give exact two-sided p = 2/32."""
    from dares.utils.ttest import wilcoxon_signed_rank

    res = wilcoxon_signed_rank([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert res.method == "exact"
    assert res.statistic == pytest.approx(15.0)
    assert res.p_value == pytest.approx(0.0625)
    assert res.n == 5 and res.n_zero == 0


def test_wilcoxon_identical_scores_p_one():
    from dares.utils.ttest import wilcoxon_signed_rank

    res = wilcoxon_signed_rank([0.7] * 8, [0.7] * 8)
    assert res.p_value == pytest.approx(1.0)
    assert res.n == 0 and res.n_zero == 8


def test_wilcoxon_shifted_scores_significant():
    from dares.utils.ttest import wilcoxon_signed_rank

    rng = np.random.default_rng(7)
    base = rng.normal(0.60, 0.05, size=60).tolist()
    improved = [v + 0.08 for v in base]
    res = wilcoxon_signed_rank(improved, base)
    assert res.p_value < 0.001
    assert res.mean_diff == pytest.approx(0.08, abs=1e-9)


def test_wilcoxon_ties_use_normal_method():
    from dares.utils.ttest import wilcoxon_signed_rank

    a = [0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 0.4, 0.55]
    b = [0.4, 0.5, 0.6, 0.6, 0.7, 0.8, 0.3, 0.45]
    res = wilcoxon_signed_rank(a, b)
    assert res.method == "normal"
    assert 0.0 <= res.p_value <= 1.0


def test_wilcoxon_greater_is_half_of_two_sided():
    from dares.utils.ttest import wilcoxon_signed_rank

    a = [0.75, 0.78, 0.72, 0.76, 0.74, 0.77, 0.79, 0.73]
    b = [0.70, 0.71, 0.69, 0.72, 0.70, 0.71, 0.72, 0.70]
    two = wilcoxon_signed_rank(a, b, alternative="two-sided")
    greater = wilcoxon_signed_rank(a, b, alternative="greater")
    assert greater.p_value == pytest.approx(two.p_value / 2.0, rel=1e-6)


def test_wilcoxon_length_mismatch_raises():
    from dares.utils.ttest import wilcoxon_signed_rank

    with pytest.raises(ValueError, match="equal lengths"):
        wilcoxon_signed_rank([0.7, 0.6], [0.7])


def test_holm_adjust_hand_computed():
    from dares.utils.ttest import holm_adjust

    assert holm_adjust([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert holm_adjust([0.5]) == pytest.approx([0.5])
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust([1.5])


def test_significance_stars_boundaries():
    from dares.utils.ttest import significance_stars

    assert significance_stars(0.0009) == "***"
    assert significance_stars(0.009) == "**"
    assert significance_stars(0.049) == "*"
    assert significance_stars(0.05) == "ns"
    assert significance_stars(1.0) == "ns"


def test_per_sample_metric_scores_miou_and_dice():
    from dares.utils.ttest import per_sample_metric_scores

    class ConstModel(torch.nn.Module):
        def forward(self, x, mode="class"):
            b, _, h, w = x.shape
            return torch.zeros(b, 2, h, w)  # always predicts class 0

    imgs = torch.randn(2, 4, 8, 8)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    labels[0, :4, :] = 255
    scores = per_sample_metric_scores(
        ConstModel(), [[imgs, labels]], torch.device("cpu"), 2, ignore_index=255
    )
    assert set(scores) == {"miou", "dice"}
    assert len(scores["miou"]) == 2 and len(scores["dice"]) == 2
    assert all(0.0 <= s <= 1.0 for s in scores["miou"] + scores["dice"])


def test_ttest_yaml_configs_parse_with_expected_pairs():
    from dares.config import TTestExperimentConfig

    abl = TTestExperimentConfig.from_yaml(
        str(ROOT / "configs" / "ttest" / "ablation_table6.yaml")
    )
    assert len(abl.ttest.models) == 5
    assert len(abl.ttest.comparisons) == 10
    assert abl.ttest.splits == ["target_test", "source_test"]
    assert abl.ttest.metrics == ["miou", "dice"]

    base = TTestExperimentConfig.from_yaml(
        str(ROOT / "configs" / "ttest" / "baselines_table3.yaml")
    )
    assert len(base.ttest.models) == 4
    assert len(base.ttest.comparisons) == 3


def test_ttest_config_rejects_unknown_comparison_model(tmp_path):
    import pydantic

    from dares.config import TTestExperimentConfig

    cfg_dict = {
        "data": {"source_dir": ".", "target_dir": "."},
        "model": {},
        "ttest": {
            "models": {"a": "a.pth", "b": "b.pth"},
            "comparisons": [["a", "ghost"]],
            "output_dir": str(tmp_path),
        },
    }
    with pytest.raises(pydantic.ValidationError, match="ghost"):
        TTestExperimentConfig(**cfg_dict)


def _write_ttest_script_config(tmp_path, ckpts: dict) -> Path:
    for domain in ("source", "target"):
        for split, n in (("train", 6), ("val", 3), ("test", 8)):
            _make_h5(tmp_path / f"{domain}_{split}.h5", n=n)
    config = {
        "data": {
            "source_dir": str(tmp_path),
            "target_dir": str(tmp_path),
            "batch_size": 2,
            "patch_size": 32,
            "num_workers": 0,
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "use_augmentation": False,
        },
        "model": {
            "backbone": "resnet50",
            "head": "resunet",
            "in_channels": 4,
            "num_classes": 2,
            "pretrained": False,
            "dropout_rate": 0.0,
        },
        "ttest": {
            "test": "wilcoxon",
            "alpha": 0.05,
            "min_samples": 2,
            "alternative": "two-sided",
            "splits": ["target_test"],
            "metrics": ["miou", "dice"],
            "output_dir": str(tmp_path / "ttest_out"),
            "models": ckpts,
            "comparisons": [["m1", "m2"]],
        },
    }
    path = tmp_path / "config_ttest_script.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def _make_h5(path: Path, n: int = 8) -> None:
    rng = np.random.default_rng(3)
    masks = rng.integers(0, 2, size=(n, 32, 32)).astype(np.uint8)
    images = rng.random((n, 4, 32, 32)).astype(np.float32)
    images[:, 0] = masks
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images, compression="lzf",
                         chunks=(1, 4, 32, 32))
        f.create_dataset("masks", data=masks, compression="lzf",
                         chunks=(1, 32, 32))


def test_ttest_script_identical_checkpoints_all_ns(tmp_path):
    """Same checkpoint twice: identical scores, p_raw == p_holm == 1, ns."""
    import torch

    from dares.config import ModelConfig
    from dares.models import build_model
    from scripts.ttest import main as ttest_main

    model = build_model(
        ModelConfig(backbone="resnet50", head="resunet", in_channels=4,
                    num_classes=2, pretrained=False)
    )
    ckpt = tmp_path / "w.pth"
    torch.save({"model": model.state_dict()}, ckpt)
    config_path = _write_ttest_script_config(
        tmp_path, {"m1": str(ckpt), "m2": str(ckpt)}
    )
    ttest_main(str(config_path))

    out = tmp_path / "ttest_out"
    with open(out / "wilcoxon_results.json") as f:
        results = json.load(f)
    with open(out / "per_patch_scores.json") as f:
        scores = json.load(f)
    assert len(scores["target_test"]["miou"]["m1"]) == 8
    table = results["tables"]["target_test"]["miou"]
    assert table["model_summary"]["m1"]["mean"] == pytest.approx(
        table["model_summary"]["m2"]["mean"]
    )
    assert len(table["pairs"]) == 1
    pair = table["pairs"][0]
    assert pair["p_raw"] == pytest.approx(1.0)
    assert pair["p_holm"] == pytest.approx(1.0)
    assert pair["stars"] == "ns"
    assert pair["significant"] is False


def test_ttest_script_missing_checkpoint_fails_fast(tmp_path):
    from scripts.ttest import main as ttest_main

    config_path = _write_ttest_script_config(
        tmp_path, {"m1": str(tmp_path / "nope.pth"), "m2": str(tmp_path / "nah.pth")}
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        ttest_main(str(config_path))
