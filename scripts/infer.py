"""
Command-line script to run single-image inference with a trained DARES model.

Loads the weights from a checkpoint into the configured segmentation model,
runs dense inference on one image (or every patch of an HDF5 container with
``--all``) and saves the prediction artifacts: the input image (RGB composite
PNG) and its NIR band (PNG + NPY), the hard mask (PNG + NPY), the
forest-probability heatmap (PNG + NPY), the ground-truth mask when the
container carries one (PNG + NPY), a qualitative overlay figure and a
per-patch metrics JSON.

Supported image inputs:
    * HDF5 container (``.h5``): reads the patch selected by ``--index`` from
      the ``images`` dataset (and its ``masks`` counterpart, when present).
    * NumPy array (``.npy`` / ``.npz``): a ``(C, H, W)`` float array.
    * Standard image (``.png`` / ``.jpg``): loaded with Pillow; missing
      bands are zero-padded up to the model channel count.

Examples:
    python scripts/infer.py --config configs/LIME_stress/medium/dares.yaml \
        --model outputs/LIME_stress/medium/dares/experiment_1/model_final.pth \
        --image /path/to/target_test.h5 --index 3
    python scripts/infer.py --config configs/LIME_stress/medium/dares.yaml \
        --model model_final.pth --image target_test.h5 --all \
        --output_dir outputs/infer
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from torch.amp import autocast

from dares.config import ExperimentConfig
from dares.data.h5_dataset import HDF5Dataset
from dares.data.transforms import SegmentationTransforms
from dares.models import build_model


def load_image(
    image_path: str, index: int, in_channels: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Reads one ``(C, H, W)`` float32 image (and its GT mask, if any).

    Args:
        image_path (str): Path to the ``.h5`` / ``.npy`` / ``.npz`` /
            ``.png`` / ``.jpg`` input.
        index (int): Patch index when the input is an HDF5 container.
        in_channels (int): Channel count the model expects; thinner inputs
            are zero-padded.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: ``(image, ground_truth)`` where
        ``image`` is ``(in_channels, H, W)`` float32 and ``ground_truth`` is
        ``(H, W)`` uint8 for HDF5 containers exposing ``masks`` (else ``None``).

    Raises:
        ValueError: If the input has more bands than the model expects or an
            unsupported shape.
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    ground_truth: np.ndarray | None = None

    if suffix == ".h5":
        with h5py.File(path, "r") as f:
            array = np.asarray(f["images"][index], dtype=np.float32)
            if "masks" in f:
                ground_truth = np.asarray(f["masks"][index], dtype=np.uint8)
    elif suffix == ".npy":
        array = np.load(path).astype(np.float32)
    elif suffix == ".npz":
        with np.load(path) as z:
            array = np.asarray(z[z.files[0]], dtype=np.float32)
    elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        pil = Image.open(path).convert("RGB")
        array = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image extension: {suffix!r}")

    if array.ndim != 3:
        raise ValueError(f"Expected a (C, H, W) image, got shape {array.shape}")
    if array.shape[0] > in_channels:
        raise ValueError(
            f"Image has {array.shape[0]} bands but the model expects "
            f"{in_channels}; check --config model.in_channels."
        )
    if array.shape[0] < in_channels:
        pad = np.zeros(
            (in_channels - array.shape[0], *array.shape[1:]), dtype=np.float32
        )
        array = np.concatenate([array, pad], axis=0)

    return np.nan_to_num(array, nan=0.0, posinf=10000.0, neginf=-10000.0), ground_truth


def _rgb_composite(img: torch.Tensor) -> np.ndarray:
    """Percentile-stretches the first three bands into a displayable RGB."""
    arr = img[:3].numpy()
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)


def patch_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, float]:
    """Computes per-class IoU, DICE (positive class) and mIoU for one patch.

    Args:
        prediction (np.ndarray): Hard class mask ``(H, W)``.
        ground_truth (np.ndarray): Label mask ``(H, W)`` (``ignore_index``
            pixels are excluded).
        num_classes (int): Number of classes.
        ignore_index (int): Label to exclude from the computation.

    Returns:
        dict[str, float]: ``iou_<class>`` per class, ``dice_forest`` and
        ``mIoU``.
    """
    valid = ground_truth != ignore_index
    pred = np.where(valid, prediction, -1)
    gt = np.where(valid, ground_truth, -1)
    out: dict[str, float] = {}
    ious: list[float] = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        union = int((p | g).sum())
        inter = int((p & g).sum())
        iou = inter / union if union > 0 else float("nan")
        name = HDF5Dataset.classes[c] if c < len(HDF5Dataset.classes) else f"class_{c}"
        out[f"iou_{name}"] = iou
        if union > 0:
            ious.append(iou)
        if name == "forest":
            out["dice_forest"] = 2.0 * inter / max(int(p.sum() + g.sum()), 1)
    out["mIoU"] = float(np.mean(ious)) if ious else float("nan")
    return out


def save_artifacts(
    image: torch.Tensor,
    prob_forest: torch.Tensor,
    prediction: torch.Tensor,
    ground_truth: np.ndarray | None,
    metrics: dict[str, float] | None,
    class_name: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    """Writes the prediction / GT / probability artifacts and the overlay.

    Args:
        image (torch.Tensor): Normalized input ``(C, H, W)`` (for display).
        prob_forest (torch.Tensor): Positive-class probability ``(H, W)``.
        prediction (torch.Tensor): Hard class mask ``(H, W)`` (int64).
        ground_truth (np.ndarray | None): GT mask ``(H, W)`` when available.
        metrics (dict[str, float] | None): Per-patch metrics for the titles
            and the JSON dump.
        class_name (str): Name of the positive class (overlay title).
        output_dir (Path): Destination directory.
        stem (str): File-name stem (input name + patch index).

    Returns:
        list[Path]: The saved artifact paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    pred_np = prediction.numpy().astype(np.uint8)
    prob_np = prob_forest.numpy().astype(np.float32)

    mask_path = output_dir / f"{stem}_prediction.png"
    Image.fromarray(pred_np, mode="L").save(mask_path)
    paths.append(mask_path)

    npy_pred = output_dir / f"{stem}_prediction.npy"
    np.save(npy_pred, pred_np)
    npy_prob = output_dir / f"{stem}_probability.npy"
    np.save(npy_prob, prob_np)
    paths.extend([npy_pred, npy_prob])

    if ground_truth is not None:
        gt_png = output_dir / f"{stem}_groundtruth.png"
        Image.fromarray(ground_truth, mode="L").save(gt_png)
        gt_npy = output_dir / f"{stem}_groundtruth.npy"
        np.save(gt_npy, ground_truth)
        paths.extend([gt_png, gt_npy])

    if metrics is not None:
        metrics_path = output_dir / f"{stem}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        paths.append(metrics_path)

    prob_path = output_dir / f"{stem}_probability.png"
    plt.imsave(prob_path, prob_np, cmap="viridis")
    paths.append(prob_path)

    overlay_path = output_dir / f"{stem}_overlay.png"
    rgb = _rgb_composite(image)

    input_path = output_dir / f"{stem}_input.png"
    Image.fromarray((rgb * 255).round().astype(np.uint8), mode="RGB").save(input_path)
    paths.append(input_path)

    if image.shape[0] > 3:
        nir = image[3].numpy()
        lo, hi = np.percentile(nir, 2), np.percentile(nir, 98)
        nir_stretched = np.clip((nir - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        nir_png = output_dir / f"{stem}_nir.png"
        Image.fromarray((nir_stretched * 255).round().astype(np.uint8), mode="L").save(
            nir_png
        )
        nir_npy = output_dir / f"{stem}_nir.npy"
        np.save(nir_npy, nir.astype(np.float32))
        paths.extend([nir_png, nir_npy])

    forest = prediction.numpy() == 1
    blended = rgb.copy()
    blended[forest] = 0.55 * blended[forest] + 0.45 * np.array([0.0, 0.8, 0.2])

    panels: list[tuple[np.ndarray, str, float | None, float | None]] = [
        (rgb, "Input (B2, B3, B4)", None, None),
        (pred_np, "Prediction", 0.0, 1.0),
        (prob_np, f"P({class_name})", 0.0, 1.0),
        (
            np.clip(blended, 0.0, 1.0),
            f"Overlay | {class_name}: {100.0 * forest.mean():.2f}%",
            None,
            None,
        ),
    ]
    cmaps = ["viridis", "Greens", "viridis", "viridis"]
    if ground_truth is not None:
        panels.insert(1, (ground_truth, "Ground truth", 0.0, 1.0))
        cmaps.insert(1, "Greens")
    if metrics is not None:
        arr, title, vmin, vmax = panels[-1]
        panels[-1] = (arr, f"{title} | mIoU {metrics['mIoU']:.4f}", vmin, vmax)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, (arr, title, vmin, vmax), cmap in zip(np.atleast_1d(axes), panels, cmaps):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(overlay_path)

    return paths


def main(
    config_path: str,
    model_path: str,
    image_path: str,
    index: int = 0,
    all_patches: bool = False,
    output_dir: str | None = None,
    device: str | None = None,
) -> None:
    """Runs inference with a trained checkpoint on one image or one container.

    Args:
        config_path (str): Path to the YAML configuration file (provides the
            model architecture and the data normalization statistics).
        model_path (str): Path to the trained checkpoint (``model_final.pth``).
        image_path (str): Path to the input image / container.
        index (int): Patch index for HDF5 inputs (ignored with ``all_patches``).
        all_patches (bool): Run every patch of an HDF5 container.
        output_dir (str | None): Destination folder; defaults to
            ``experiment.output_dir/inference``.
        device (str | None): Optional device override (``cuda`` / ``cpu``).
    """
    cfg = ExperimentConfig.from_yaml(config_path)

    device_name = cfg.training.device if torch.cuda.is_available() else "cpu"
    if device is not None:
        device_name = device
    device_obj = torch.device(device_name)

    # 1. Model + weights
    model = build_model(cfg.model)
    state = torch.load(model_path, map_location=device_obj, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(device_obj).eval()

    # 2. Patch selection
    path = Path(image_path)
    if all_patches:
        if path.suffix.lower() != ".h5":
            raise ValueError("--all requires an HDF5 container input")
        with h5py.File(path, "r") as f:
            indices = range(int(f["images"].shape[0]))
    else:
        indices = [index]

    out = Path(output_dir) if output_dir else Path(cfg.experiment.output_dir) / "inference"
    transform = SegmentationTransforms(
        train=False, mean=cfg.data.mean, std=cfg.data.std
    )()
    class_names = HDF5Dataset.classes
    use_amp = bool(cfg.training.use_amp and device_obj.type == "cuda")
    summary: list[dict[str, object]] = []

    # 3. Dense inference per patch
    for i in indices:
        array, ground_truth = load_image(str(path), i, cfg.model.in_channels)
        tensor, _ = transform(array, None)
        batch = tensor.unsqueeze(0).to(device_obj)

        with torch.no_grad(), autocast(device_type=device_obj.type, enabled=use_amp):
            logits = model(batch, mode="class")
        probs = F.softmax(logits.float(), dim=1)[0]
        prediction = torch.argmax(probs, dim=0)
        prob_forest = probs[1] if probs.shape[0] > 1 else probs[0]

        metrics = None
        if ground_truth is not None:
            metrics = patch_metrics(
                prediction.cpu().numpy(), ground_truth, cfg.model.num_classes
            )

        stem = f"{path.stem}_idx{i}" if path.suffix.lower() == ".h5" else path.stem
        saved = save_artifacts(
            tensor,
            prob_forest.cpu(),
            prediction.cpu(),
            ground_truth,
            metrics,
            class_names[1],
            out,
            stem,
        )

        forest_pct = 100.0 * float((prediction == 1).float().mean())
        line = (
            f"  idx {i}: input {tuple(array.shape)} | forest {forest_pct:.2f}%"
        )
        if metrics is not None:
            line += f" | mIoU {metrics['mIoU']:.4f}"
            summary.append({"index": i, **metrics})
        print(line)
        for p in saved:
            print(f"    saved {p.name}")

    if summary:
        mean_metrics = {
            k: float(np.nanmean([s[k] for s in summary]))
            for k in summary[0]
            if k != "index"
        }
        with open(out / "inference_summary.json", "w") as f:
            json.dump({"patches": summary, "mean": mean_metrics}, f, indent=2)
        print(f"\n Mean over {len(summary)} patches: "
              + " | ".join(f"{k} {v:.4f}" for k, v in mean_metrics.items()))
    print(f"\n Artifacts in: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DARES single-image inference script"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained .pth checkpoint"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input (.h5/.npy/.npz/.png/.jpg)"
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Patch index for HDF5 inputs"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Infer every patch of an HDF5 container",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Destination folder (defaults to <experiment.output_dir>/inference)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (cuda / cpu)"
    )
    args = parser.parse_args()
    main(
        args.config,
        args.model,
        args.image,
        args.index,
        args.all,
        args.output_dir,
        args.device,
    )
