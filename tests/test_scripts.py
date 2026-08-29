"""Smoke tests for the DARES train / evaluate command-line scripts."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import main as evaluate_main  # noqa: E402
from scripts.train import main as train_main  # noqa: E402

def make_h5(
    file_path: Path, num_patches: int = 6, height: int = 64, width: int = 64
) -> None:
    """Writes a tiny synthetic LZF HDF5 container."""
    file_path = Path(file_path)
    rng = np.random.default_rng(0)
    masks = rng.integers(0, 2, size=(num_patches, height, width)).astype(np.uint8)
    images = rng.random((num_patches, 4, height, width)).astype(np.float32)
    images[:, 0] = masks
    with h5py.File(file_path, "w") as f:
        f.create_dataset(
            "images", data=images, compression="lzf", chunks=(1, 4, height, width)
        )
        f.create_dataset(
            "masks", data=masks, compression="lzf", chunks=(1, height, width)
        )


def write_config(tmp_path: Path, method: str, out_dir: str) -> Path:
    """Writes a minimal DARES YAML config pointing at the synthetic data."""
    for domain in ("source", "target"):
        for split, num in (("train", 6), ("val", 3), ("test", 3)):
            make_h5(tmp_path / f"{domain}_{split}.h5", num_patches=num)

    config = {
        "data": {
            "source_dir": str(tmp_path),
            "target_dir": str(tmp_path),
            "batch_size": 2,
            "patch_size": 64,
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
            "method": method,
            "epochs": 1,
            "lr": 1e-4,
            "device": "cpu",
            "use_amp": False,
            "seed": 42,
        },
        "experiment": {
            "name": f"test_{method}",
            "version": 1,
            "output_dir": out_dir,
            "save_results": True,
        },
    }
    config_path = tmp_path / f"config_{method}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.mark.parametrize("method", ["source_only", "dares"])
def test_train_script_end_to_end(tmp_path, method):
    """train.py produces the checkpoint, history and test-metrics artifacts."""
    out_dir = str(tmp_path / "outputs" / method)
    config_path = write_config(tmp_path, method, out_dir)

    train_main(str(config_path))

    out = Path(out_dir)
    assert (out / "model_final.pth").is_file()
    assert (out / "history.json").is_file()
    assert (out / "test_metrics.json").is_file()

    with open(out / "test_metrics.json") as f:
        metrics = json.load(f)
    assert "target_test" in metrics and "source_test" in metrics
    assert metrics["target_test"]["mIoU"] >= 0.0


def test_evaluate_script_end_to_end(tmp_path):
    """evaluate.py loads a checkpoint and produces metrics + figures."""
    out_dir = str(tmp_path / "outputs" / "source_only")
    config_path = write_config(tmp_path, "source_only", out_dir)
    train_main(str(config_path))

    model_path = str(Path(out_dir) / "model_final.pth")
    eval_dir = str(tmp_path / "evaluation")
    evaluate_main(str(config_path), model_path, output_dir=eval_dir)

    out = Path(eval_dir)
    assert (out / "evaluation_metrics.json").is_file()
    assert (out / "target_test_confusion_matrix.png").is_file()
    assert (out / "source_test_confusion_matrix.png").is_file()
    assert (out / "target_test_predictions.png").is_file()

    with open(out / "evaluation_metrics.json") as f:
        metrics = json.load(f)
    assert "confusion_matrix" in metrics["target_test"]
    assert metrics["target_test"]["mIoU"] >= 0.0


def test_check_data_script(tmp_path, capsys):
    """check_data.py validates the six containers and reports per-split stats."""
    from scripts.check_data import main as check_main

    write_config(tmp_path, "source_only", str(tmp_path / "out"))

    check_main(str(tmp_path), str(tmp_path), batch_size=2)

    captured = capsys.readouterr().out
    assert "DARES HDF5 container validation" in captured
    for split_file in ("source_train.h5", "target_train.h5", "target_test.h5"):
        assert split_file in captured
    assert "forest ratio" in captured


METHODS = ["source_only", "advent", "dacs", "fda", "dares"]
LIME_LEVELS = {"low": "low", "medium": "medium", "high": "high"}
ARCH_COMBOS = [
    "resnet50_resunet",
    "resnet50_deeplabv3p",
    "convnext_tiny_deeplabv3p",
    "swin_t_resunet",
    "swin_t_deeplabv3p",
]


def test_config_matrix_complete_and_valid():
    """The three experiment folders parse and point at the right settings."""
    from dares.config import ExperimentConfig

    configs_root = ROOT / "configs"

    # 1. LIME_stress: convnext_tiny_resunet x 5 methods x 3 LIME levels.
    lime_dir = configs_root / "LIME_stress"
    for level, variant in LIME_LEVELS.items():
        for method in METHODS:
            path = lime_dir / level / f"{method}.yaml"
            assert path.is_file(), f"missing {path}"
            cfg = ExperimentConfig.from_yaml(path)
            assert cfg.model.backbone == "convnext_tiny"
            assert cfg.model.head == "resunet"
            assert cfg.data.target_variant == variant
            assert cfg.training.method == method
            assert cfg.training.epochs == 25
            assert cfg.training.warmup_epochs == 2
            assert cfg.experiment.output_dir == Path(
                f"/content/drive/MyDrive/DARES_experiments/LIME_stress/{level}/{method}/experiment_1"
            )

    # 2. architectures: source_only + dares, medium, 5 backbone-head combos.
    arch_dir = configs_root / "architectures"
    for combo in ARCH_COMBOS:
        backbone, head = combo.rsplit("_", 1)
        for method in ("source_only", "dares"):
            path = arch_dir / combo / f"{method}.yaml"
            assert path.is_file(), f"missing {path}"
            cfg = ExperimentConfig.from_yaml(path)
            assert cfg.model.backbone == backbone
            assert cfg.model.head == head
            assert cfg.data.target_variant == "medium"
            assert cfg.training.method == method
            assert cfg.experiment.output_dir == Path(
                f"/content/drive/MyDrive/DARES_experiments/architectures/{combo}/{method}/experiment_1"
            )

    # 3. ablation: convnext_tiny_resunet, medium, one DARES component off each.
    ablation_dir = configs_root / "ablation"
    expected_ablations = {
        "dares_no_anti_collapse": {"beta": 0.0, "repulsion_gamma": 0.5, "trust_region": True},
        "dares_no_repulsion": {"beta": 1.0, "repulsion_gamma": 0.0, "trust_region": True},
        "dares_no_trust_region": {"beta": 1.0, "repulsion_gamma": 0.5, "trust_region": False},
    }
    for name, params in expected_ablations.items():
        path = ablation_dir / f"{name}.yaml"
        assert path.is_file(), f"missing {path}"
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.model.backbone == "convnext_tiny"
        assert cfg.model.head == "resunet"
        assert cfg.data.target_variant == "medium"
        assert cfg.training.method == "dares"
        assert cfg.training.beta == params["beta"]
        assert cfg.training.repulsion_gamma == params["repulsion_gamma"]
        assert cfg.training.trust_region == params["trust_region"]
        assert cfg.experiment.output_dir == Path(
            f"/content/drive/MyDrive/DARES_experiments/ablation/{name}/experiment_1"
        )
