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


def test_train_script_writes_ttest_json_when_enabled(tmp_path):
    from scripts.train import main as train_main

    out_dir = str(tmp_path / "outputs")
    config_path = _write_config(tmp_path, out_dir)
    train_main(str(config_path))
    ttest_path = Path(out_dir) / "test_ttest.json"
    assert ttest_path.is_file()
    with open(ttest_path) as f:
        payload = json.load(f)
    assert payload["metric"] == "per_patch_miou"
    assert 0.0 <= payload["p_value"] <= 1.0
    assert payload["n_a"] >= 3 and payload["n_b"] >= 3


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
