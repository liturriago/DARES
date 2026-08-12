"""
Command-line script to evaluate a trained DARES model checkpoint.

Runs pixel-level inference on the source and target test splits, prints the
full metric report (IoU, DICE, precision, recall, mIoU, accuracy, MCC) and
saves JSON metrics plus confusion-matrix / prediction-overlay figures.

Examples:
    python scripts/evaluate.py --config configs/training/dares.yaml \
        --model outputs/dares/experiment_1/model_final.pth
    python scripts/evaluate.py --config configs/training/dares.yaml \
        --model outputs/dares/experiment_1/model_final.pth --output_dir outputs/dares/eval
"""
import argparse
import json
from pathlib import Path

import torch

from dares.config import ExperimentConfig
from dares.data.loader import DARESDataLoader
from dares.models import build_model
from dares.utils.evaluation import evaluate_segmentation, metrics_to_jsonable
from dares.utils.reproducibility import set_seed
from dares.utils.visualizer import SegmentationVisualizer


def main(
    config_path: str,
    model_path: str,
    output_dir: str | None = None,
) -> None:
    """Evaluates a trained checkpoint on the source and target test splits.

    Args:
        config_path (str): Path to the YAML configuration file.
        model_path (str): Path to the trained checkpoint (``model_final.pth``).
        output_dir (str | None): Optional override for ``experiment.output_dir``.
    """
    cfg = ExperimentConfig.from_yaml(config_path)
    if output_dir is not None:
        cfg.experiment.output_dir = Path(output_dir)

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    set_seed(cfg.training.seed)

    output_path = Path(cfg.experiment.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Data
    data_manager = DARESDataLoader(cfg.data)
    source_loaders = data_manager.get_source_loaders()
    target_loaders = data_manager.get_target_loaders()

    # 2. Model + weights
    model = build_model(cfg.model)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(device)

    class_names = ["non_forest", "forest"]
    for loader in (source_loaders["train"], target_loaders["train"]):
        classes = getattr(loader.dataset, "classes", None)
        if classes:
            class_names = list(classes)
            break

    # 3. Evaluation
    print(f"\n Evaluating {model_path}")
    source_metrics = evaluate_segmentation(
        model,
        source_loaders["test"],
        device,
        cfg.model.num_classes,
        class_names,
        use_amp=cfg.training.use_amp,
        prefix="SOURCE TEST",
    )
    target_metrics = evaluate_segmentation(
        model,
        target_loaders["test"],
        device,
        cfg.model.num_classes,
        class_names,
        use_amp=cfg.training.use_amp,
        prefix="TARGET TEST",
    )

    # 4. Artifacts
    with open(output_path / "evaluation_metrics.json", "w") as f:
        json.dump(
            {
                "source_test": metrics_to_jsonable(source_metrics),
                "target_test": metrics_to_jsonable(target_metrics),
            },
            f,
            indent=2,
        )

    viz = SegmentationVisualizer(output_path)
    viz.plot_confusion_matrix(
        target_metrics, class_names, "target_test_confusion_matrix.png"
    )
    viz.plot_confusion_matrix(
        source_metrics, class_names, "source_test_confusion_matrix.png"
    )
    viz.plot_prediction_overlay(
        model,
        target_loaders["test"],
        device,
        class_names,
        "target_test_predictions.png",
        use_amp=cfg.training.use_amp,
    )

    print(f"\n Evaluation complete. Results saved in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARES evaluation script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained .pth checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory defined in the YAML config",
    )
    args = parser.parse_args()
    main(args.config, args.model, args.output_dir)
