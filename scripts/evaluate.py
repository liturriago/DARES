"""
Command-line script to evaluate a trained DARES model checkpoint.

Runs pixel-level inference on the source and target test splits, prints the
full metric report (IoU, DICE, precision, recall, mIoU, accuracy, MCC) and
saves JSON metrics plus confusion-matrix / prediction-overlay figures.

Examples:
    python scripts/evaluate.py --config configs/LIME_stress/medium/dares.yaml \
        --model outputs/LIME_stress/medium/dares/experiment_1/model_final.pth
    python scripts/evaluate.py --config configs/LIME_stress/medium/dares.yaml \
        --model outputs/LIME_stress/medium/dares/experiment_1/model_final.pth --output_dir outputs/dares/eval
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
from dares.utils.ttest import (
    format_result,
    per_sample_miou_scores,
    run_ttest,
)
from dares.utils.visualizer import SegmentationVisualizer


def _load_model(cfg, model_path: str, device):
    model = build_model(cfg.model)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device)


def main(
    config_path: str,
    model_path: str,
    output_dir: str | None = None,
    compare_model: str | None = None,
    enable_ttest: bool | None = None,
) -> None:
    """Evaluates a trained checkpoint on the source and target test splits.

    Args:
        config_path (str): Path to the YAML configuration file.
        model_path (str): Path to the trained checkpoint (``model_final.pth``).
        output_dir (str | None): Optional override for ``experiment.output_dir``.
        compare_model (str | None): Optional second checkpoint; runs a paired
            per-patch mIoU t-test (same target-test patches) between the two.
        enable_ttest (bool | None): Force the domain-gap t-test on/off,
            overriding ``stats.enabled``.
    """
    cfg = ExperimentConfig.from_yaml(config_path)
    if output_dir is not None:
        cfg.experiment.output_dir = Path(output_dir)
    if enable_ttest is not None:
        cfg.stats.enabled = bool(enable_ttest)
    compare_path = compare_model or cfg.stats.compare_checkpoint

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
    model = _load_model(cfg, model_path, device)

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
        ignore_index=cfg.training.ignore_index,
    )
    target_metrics = evaluate_segmentation(
        model,
        target_loaders["test"],
        device,
        cfg.model.num_classes,
        class_names,
        use_amp=cfg.training.use_amp,
        prefix="TARGET TEST",
        ignore_index=cfg.training.ignore_index,
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

    # 5. Optional significance testing (per-patch mIoU).
    ttests: dict = {}
    if cfg.stats.enabled:
        print("\n Running per-patch mIoU t-test (source vs target)...")
        try:
            src_scores = per_sample_miou_scores(
                model,
                source_loaders["test"],
                device,
                cfg.model.num_classes,
                ignore_index=cfg.training.ignore_index,
                use_amp=cfg.training.use_amp,
            )
            tgt_scores = per_sample_miou_scores(
                model,
                target_loaders["test"],
                device,
                cfg.model.num_classes,
                ignore_index=cfg.training.ignore_index,
                use_amp=cfg.training.use_amp,
            )
            kind = cfg.stats.test if cfg.stats.test != "paired" else "welch"
            res = run_ttest(
                src_scores,
                tgt_scores,
                test=kind,
                alpha=cfg.stats.alpha,
                alternative=cfg.stats.alternative,
                min_samples=cfg.stats.min_samples,
            )
            print(format_result(res))
            ttests["source_vs_target"] = {
                "comparison": "source_test_vs_target_test",
                "metric": "per_patch_miou",
                **res.to_dict(),
            }
        except ValueError as exc:
            print(f"[t-test] skipped: {exc}")
    if compare_path is not None:
        print(f"\n Running paired t-test vs {compare_path} (target test)...")
        try:
            other = _load_model(cfg, compare_path, device)
            a = per_sample_miou_scores(
                model,
                target_loaders["test"],
                device,
                cfg.model.num_classes,
                ignore_index=cfg.training.ignore_index,
                use_amp=cfg.training.use_amp,
            )
            b = per_sample_miou_scores(
                other,
                target_loaders["test"],
                device,
                cfg.model.num_classes,
                ignore_index=cfg.training.ignore_index,
                use_amp=cfg.training.use_amp,
            )
            res = run_ttest(
                a,
                b,
                test="paired",
                alpha=cfg.stats.alpha,
                alternative=cfg.stats.alternative,
                min_samples=cfg.stats.min_samples,
            )
            print(format_result(res))
            ttests["model_vs_compare"] = {
                "comparison": "model_vs_compare_on_target_test",
                "metric": "per_patch_miou",
                **res.to_dict(),
            }
        except ValueError as exc:
            print(f"[t-test] paired skipped: {exc}")
    if ttests:
        with open(output_path / "evaluation_ttest.json", "w") as f:
            json.dump(ttests, f, indent=2)

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
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Second checkpoint for a paired per-patch mIoU t-test",
    )
    parser.add_argument(
        "--ttest",
        dest="enable_ttest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force the source-vs-target t-test on/off (overrides stats.enabled)",
    )
    args = parser.parse_args()
    main(args.config, args.model, args.output_dir, args.compare, args.enable_ttest)
