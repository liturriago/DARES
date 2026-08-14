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


BACKBONES = ["resnet50", "convnext_tiny", "swin_t"]
HEADS = ["resunet", "deeplabv3p"]
METHODS = ["source_only", "advent", "cycada", "cbst", "dares"]


def test_config_matrix_complete_and_valid():
    """One config folder per backbone-head pair, five methods each, all parse."""
    from dares.config import ExperimentConfig

    training_dir = ROOT / "configs" / "training"
    folders = sorted(p.name for p in training_dir.iterdir() if p.is_dir())
    expected_folders = sorted(f"{bb}_{hd}" for bb in BACKBONES for hd in HEADS)
    assert folders == expected_folders, "missing backbone-head folders"

    total = 0
    for folder in folders:
        backbone, head = folder.rsplit("_", 1)
        for method in METHODS:
            path = training_dir / folder / f"{method}.yaml"
            assert path.is_file(), f"missing {path}"
            cfg = ExperimentConfig.from_yaml(path)
            assert cfg.model.backbone == backbone
            assert cfg.model.head == head
            assert cfg.training.method == method
            assert cfg.training.epochs == 25
            expected_warmup = 2 if method == "dares" else 5
            assert cfg.training.warmup_epochs == expected_warmup
            assert cfg.experiment.output_dir == Path(
                f"outputs/{folder}/{method}/experiment_1"
            )
            total += 1
    assert total == len(BACKBONES) * len(HEADS) * len(METHODS) == 30
