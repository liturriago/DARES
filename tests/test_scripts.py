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


def test_infer_script_end_to_end(tmp_path):
    """infer.py loads a checkpoint on one image and writes all artifacts."""
    import torch

    from dares.config import ExperimentConfig
    from dares.models import build_model
    from scripts.infer import main as infer_main

    out_dir = str(tmp_path / "outputs" / "source_only")
    config_path = write_config(tmp_path, "source_only", out_dir)
    cfg = ExperimentConfig.from_yaml(str(config_path))
    model = build_model(cfg.model)
    model_path = tmp_path / "weights.pth"
    torch.save({"model": model.state_dict()}, model_path)

    infer_dir = str(tmp_path / "inference")
    infer_main(
        str(config_path),
        str(model_path),
        str(tmp_path / "source_test.h5"),
        index=1,
        output_dir=infer_dir,
        device="cpu",
    )

    out = Path(infer_dir)
    stem = "source_test_idx1"
    for name in (
        f"{stem}_prediction.png",
        f"{stem}_prediction.npy",
        f"{stem}_probability.npy",
        f"{stem}_probability.png",
        f"{stem}_groundtruth.png",
        f"{stem}_groundtruth.npy",
        f"{stem}_metrics.json",
        f"{stem}_overlay.png",
        f"{stem}_input.png",
        f"{stem}_nir.png",
        f"{stem}_nir.npy",
    ):
        assert (out / name).is_file(), name

    mask = np.load(out / f"{stem}_prediction.npy")
    assert mask.shape == (64, 64)
    assert set(np.unique(mask)) <= {0, 1}
    nir = np.load(out / f"{stem}_nir.npy")
    assert nir.shape == (64, 64) and nir.dtype == np.float32
    prob = np.load(out / f"{stem}_probability.npy")
    assert prob.shape == (64, 64)
    assert 0.0 <= float(prob.min()) and float(prob.max()) <= 1.0
    gt = np.load(out / f"{stem}_groundtruth.npy")
    with h5py.File(tmp_path / "source_test.h5", "r") as f:
        assert np.array_equal(gt, f["masks"][1])
    with open(out / f"{stem}_metrics.json") as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["mIoU"] <= 1.0

    # --all sweeps the whole container and writes the summary JSON.
    all_dir = str(tmp_path / "inference_all")
    infer_main(
        str(config_path),
        str(model_path),
        str(tmp_path / "source_test.h5"),
        all_patches=True,
        output_dir=all_dir,
        device="cpu",
    )
    all_out = Path(all_dir)
    for i in range(3):
        assert (all_out / f"source_test_idx{i}_prediction.npy").is_file()
    with open(all_out / "inference_summary.json") as f:
        summary = json.load(f)
    assert len(summary["patches"]) == 3
    assert "mIoU" in summary["mean"]

    # NumPy image input with fewer bands than the model: zero-padded, no GT.
    np_path = tmp_path / "scene.npy"
    np.save(np_path, np.random.rand(2, 64, 64).astype(np.float32))
    infer_main(
        str(config_path),
        str(model_path),
        str(np_path),
        output_dir=str(tmp_path / "inference_npy"),
        device="cpu",
    )
    assert (tmp_path / "inference_npy" / "scene_prediction.npy").is_file()
    assert not (tmp_path / "inference_npy" / "scene_groundtruth.npy").exists()


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


METHODS = ["source_only", "advent", "cbst", "dares"]
LIME_LEVELS = {"low": "low", "medium": "medium", "high": "high"}
ARCH_COMBOS = {
    "convnext_tiny_resunet": ("convnext_tiny", "resunet"),
    "convnext_tiny_deeplabv3p": ("convnext_tiny", "deeplabv3p"),
    "resnet50_deeplabv3p": ("resnet50", "deeplabv3p"),
    "swin_t_deeplabv3p": ("swin_t", "deeplabv3p"),
    "swin_t_resunet": ("swin_t", "resunet"),
}


def test_config_matrix_complete_and_valid():
    """The three experiment folders parse and point at the right settings."""
    from dares.config import ExperimentConfig

    configs_root = ROOT / "configs"

    # 1. LIME_stress: ResNet-50 + ResUNet x 4 methods x 3 LIME levels.
    lime_dir = configs_root / "LIME_stress"
    for level, variant in LIME_LEVELS.items():
        for method in METHODS:
            path = lime_dir / level / f"{method}.yaml"
            assert path.is_file(), f"missing {path}"
            cfg = ExperimentConfig.from_yaml(path)
            assert cfg.model.backbone == "resnet50"
            assert cfg.model.head == "resunet"
            assert cfg.data.target_variant == variant
            assert cfg.training.method == method
            assert cfg.training.epochs == 25
            assert cfg.training.warmup_epochs == 2
            assert cfg.experiment.output_dir == Path(
                f"outputs/LIME_stress/{level}/{method}/experiment_1"
            )

    # 2. architectures: source_only + dares, medium, 5 backbone-head combos.
    arch_dir = configs_root / "architectures"
    for combo, (backbone, head) in ARCH_COMBOS.items():
        for method in ("source_only", "dares"):
            path = arch_dir / combo / f"{method}.yaml"
            assert path.is_file(), f"missing {path}"
            cfg = ExperimentConfig.from_yaml(path)
            assert cfg.model.backbone == backbone
            assert cfg.model.head == head
            assert cfg.data.target_variant == "medium"
            assert cfg.training.method == method
            assert cfg.experiment.output_dir.parts[:2] == ("outputs", "architectures")
            assert cfg.experiment.output_dir.parts[-2:] == (method, "experiment_1")

    # 3. ablation: resnet50_resunet, medium, one DARES term off each.
    ablation_dir = configs_root / "ablation"
    # name -> (config field, off value): the single term switched off.
    expected_ablations = {
        "dares_no_align": ("lambda_align", 0.0),
        "dares_no_anti_collapse": ("beta", 0.0),
        "dares_no_repulsion": ("repulsion_gamma", 0.0),
        "dares_no_em": ("use_renyi_em", False),
    }
    for name, (field, off_value) in expected_ablations.items():
        path = ablation_dir / f"{name}.yaml"
        assert path.is_file(), f"missing {path}"
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.model.backbone == "resnet50"
        assert cfg.model.head == "resunet"
        assert cfg.data.target_variant == "medium"
        assert cfg.training.method == "dares"
        assert cfg.experiment.output_dir == Path(
            f"outputs/ablation/{name}/experiment_1"
        )
        assert getattr(cfg.training, field) == off_value
