"""
Command-line script to train any DARES unsupervised domain adaptation method.

The training method is selected by ``training.method`` in the YAML config
(``source_only``, ``advent``, ``cycada``, ``cbst`` or ``dares``) and can be
overridden with ``--method``.

Examples:
    python scripts/train.py --config configs/training/dares.yaml
    python scripts/train.py --config configs/training/dares.yaml --method dares
    python scripts/train.py --config configs/training/dares.yaml --device cpu
"""
import argparse
import json
from pathlib import Path

import torch

from dares.config import ExperimentConfig, TrainConfig
from dares.data.loader import DARESDataLoader
from dares.engines import build_engine
from dares.models import build_model
from dares.training.schedulers import build_scheduler
from dares.utils.evaluation import evaluate_segmentation, metrics_to_jsonable
from dares.utils.reproducibility import set_seed


def main(
    config_path: str,
    method: str | None = None,
    device: str | None = None,
) -> None:
    """Trains the method configured in the YAML and saves the best checkpoint.

    Args:
        config_path (str): Path to the YAML configuration file.
        method (str | None): Optional override for ``training.method``.
        device (str | None): Optional device override (``"cuda"`` / ``"cpu"``).
    """
    cfg = ExperimentConfig.from_yaml(config_path)
    if method is not None:
        cfg.training = TrainConfig(**cfg.training.model_dump(), method=method)

    device_name = cfg.training.device if torch.cuda.is_available() else "cpu"
    if device is not None:
        device_name = device
    device = torch.device(device_name)

    set_seed(cfg.training.seed)

    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    data_manager = DARESDataLoader(cfg.data)
    source_loaders = data_manager.get_source_loaders()
    target_loaders = data_manager.get_target_loaders()

    # 2. Model
    model = build_model(cfg.model)

    # 3. Engine (routes by cfg.training.method)
    engine = build_engine(
        cfg.training.method,
        model,
        source_loaders,
        target_loaders,
        cfg.training,
        device,
    )
    scheduler = build_scheduler(engine.optimizer, cfg.training)

    print(f"\n Starting {cfg.training.method.upper()} experiment on {device}...")
    trained_model = engine.fit(scheduler=scheduler)

    # 4. Save the best checkpoint + history
    checkpoint_path = output_dir / "model_final.pth"
    engine.save_checkpoint(checkpoint_path)
    history = dict(engine.history)
    history["best_miou"] = engine.best_miou
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best target-validation mIoU: {engine.best_miou:.4f}")
    print(f"Checkpoint saved at: {checkpoint_path}")

    # 5. Final test evaluation on both domains
    print("\n Generating final statistical reports on the test splits...")
    target_metrics = evaluate_segmentation(
        trained_model,
        target_loaders["test"],
        device,
        cfg.model.num_classes,
        engine.class_names,
        use_amp=cfg.training.use_amp,
        prefix="TARGET TEST",
    )
    source_metrics = evaluate_segmentation(
        trained_model,
        source_loaders["test"],
        device,
        cfg.model.num_classes,
        engine.class_names,
        use_amp=cfg.training.use_amp,
        prefix="SOURCE TEST",
    )
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(
            {
                "target_test": metrics_to_jsonable(target_metrics),
                "source_test": metrics_to_jsonable(source_metrics),
            },
            f,
            indent=2,
        )
    print(f"\n Experiment completed. Results saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DARES training script (source_only / advent / cycada / cbst / dares)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Override the training method (source_only, advent, cycada, cbst, dares)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device (cuda / cpu)",
    )
    args = parser.parse_args()
    main(args.config, args.method, args.device)
