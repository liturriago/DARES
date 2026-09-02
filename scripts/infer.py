"""
Command-line script to run single-image inference with a trained DARES model.

Loads the weights from a checkpoint into the configured segmentation model,
runs dense inference on ONE image and saves the prediction artifacts: the
hard mask (PNG + NPY), the forest-probability heatmap and a qualitative
overlay figure.

Supported image inputs:
    * HDF5 container (``.h5``): reads the patch selected by ``--index`` from
      the ``images`` dataset (the same containers used for training).
    * NumPy array (``.npy`` / ``.npz``): a ``(C, H, W)`` float array.
    * Standard image (``.png`` / ``.jpg``): loaded with Pillow; missing
      bands are zero-padded up to the model channel count.

Examples:
    python scripts/infer.py --config configs/LIME_stress/medium/dares.yaml \
        --model outputs/LIME_stress/medium/dares/experiment_1/model_final.pth \
        --image /path/to/target_test.h5 --index 3
    python scripts/infer.py --config configs/LIME_stress/medium/dares.yaml \
        --model model_final.pth --image patch.npy --output_dir outputs/infer
"""
import argparse
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


def load_image(image_path: str, index: int, in_channels: int) -> np.ndarray:
    """Reads one ``(C, H, W)`` float32 image from the supported containers.

    Args:
        image_path (str): Path to the ``.h5`` / ``.npy`` / ``.npz`` /
            ``.png`` / ``.jpg`` input.
        index (int): Patch index when the input is an HDF5 container.
        in_channels (int): Channel count the model expects; thinner inputs
            are zero-padded.

    Returns:
        np.ndarray: The sanitized image of shape ``(in_channels, H, W)``.

    Raises:
        ValueError: If the input has more bands than the model expects or an
            unsupported shape.
    """
    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix == ".h5":
        with h5py.File(path, "r") as f:
            array = np.asarray(f["images"][index], dtype=np.float32)
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

    return np.nan_to_num(array, nan=0.0, posinf=10000.0, neginf=-10000.0)


def _rgb_composite(img: torch.Tensor) -> np.ndarray:
    """Percentile-stretches the first three bands into a displayable RGB."""
    arr = img[:3].numpy()
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)


def save_artifacts(
    image: torch.Tensor,
    prob_forest: torch.Tensor,
    prediction: torch.Tensor,
    class_name: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    """Writes the prediction mask, probability map and overlay figure.

    Args:
        image (torch.Tensor): Normalized input ``(C, H, W)`` (for display).
        prob_forest (torch.Tensor): Positive-class probability ``(H, W)``.
        prediction (torch.Tensor): Hard class mask ``(H, W)`` (int64).
        class_name (str): Name of the positive class (overlay title).
        output_dir (Path): Destination directory.
        stem (str): File-name stem derived from the input image.

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

    prob_path = output_dir / f"{stem}_probability.png"
    plt.imsave(prob_path, prob_np, cmap="viridis")
    paths.append(prob_path)

    overlay_path = output_dir / f"{stem}_overlay.png"
    rgb = _rgb_composite(image)
    forest = prediction.numpy() == 1
    blended = rgb.copy()
    blended[forest] = 0.55 * blended[forest] + 0.45 * np.array([0.0, 0.8, 0.2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input (B2, B3, B4)")
    axes[1].imshow(prob_np, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"P({class_name})")
    axes[2].imshow(np.clip(blended, 0.0, 1.0))
    axes[2].set_title(
        f"Prediction | {class_name}: {100.0 * forest.mean():.2f}%"
    )
    for ax in axes:
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
    output_dir: str | None = None,
    device: str | None = None,
) -> None:
    """Runs inference with a trained checkpoint on a single image.

    Args:
        config_path (str): Path to the YAML configuration file (provides the
            model architecture and the data normalization statistics).
        model_path (str): Path to the trained checkpoint (``model_final.pth``).
        image_path (str): Path to the input image / container.
        index (int): Patch index for HDF5 inputs.
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

    # 2. Image -> normalized tensor
    array = load_image(image_path, index, cfg.model.in_channels)
    transform = SegmentationTransforms(
        train=False, mean=cfg.data.mean, std=cfg.data.std
    )()
    tensor, _ = transform(array, None)
    batch = tensor.unsqueeze(0).to(device_obj)

    # 3. Dense inference
    use_amp = bool(cfg.training.use_amp and device_obj.type == "cuda")
    with torch.no_grad(), autocast(device_type=device_obj.type, enabled=use_amp):
        logits = model(batch, mode="class")
    probs = F.softmax(logits.float(), dim=1)[0]
    prediction = torch.argmax(probs, dim=0)
    prob_forest = probs[1] if probs.shape[0] > 1 else probs[0]

    # 4. Artifacts
    out = Path(output_dir) if output_dir else Path(cfg.experiment.output_dir) / "inference"
    class_names = HDF5Dataset.classes
    saved = save_artifacts(
        tensor, prob_forest.cpu(), prediction.cpu(), class_names[1], out, Path(image_path).stem
    )

    forest_pct = 100.0 * float((prediction == 1).float().mean())
    print(f"\n Inference on {image_path} (device={device_obj})")
    print(f"  Input shape: {tuple(array.shape)} | forest: {forest_pct:.2f}%")
    for path in saved:
        print(f"  Saved: {path}")


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
        args.output_dir,
        args.device,
    )
