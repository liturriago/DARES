"""Minimal plotting utilities for the DARES evaluation pipeline."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch.amp import autocast
from typing import Any


class SegmentationVisualizer:
    """Saves confusion-matrix heatmaps and qualitative prediction overlays.

    Args:
        output_dir (str | Path): Directory where figures are saved.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        metrics: dict[str, Any],
        class_names: list[str],
        filename: str = "confusion_matrix.png",
    ) -> Path:
        """Saves a normalized confusion-matrix heatmap.

        Args:
            metrics (dict[str, Any]): Output of ``MetricTracker.compute_full_metrics``.
            class_names (list[str]): Human-readable class names.
            filename (str): Output file name.

        Returns:
            Path: Path of the saved figure.
        """
        conf = np.asarray(metrics["confusion_matrix"], dtype=float)
        cm_norm = conf / (conf.sum(axis=1, keepdims=True) + 1e-8)
        thresh = cm_norm.max() / 2 if cm_norm.size else 0.5

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        im = ax.imshow(cm_norm, cmap="Blues")
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j,
                    i,
                    f"{conf[i, j]:.0f}\n({cm_norm[i, j]:.1%})",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=9,
                )
        ax.set_title(
            f"Confusion matrix | mIoU {metrics['mIoU']:.4f} | "
            f"DICE {metrics['mean_dice']:.4f}"
        )
        fig.colorbar(im, ax=ax)
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    @torch.no_grad()
    def plot_prediction_overlay(
        self,
        model: torch.nn.Module,
        loader: Any,
        device: torch.device,
        class_names: list[str],
        filename: str = "predictions.png",
        num_samples: int = 3,
        use_amp: bool = False,
    ) -> Path:
        """Saves a qualitative (input, ground truth, prediction) overlay grid.

        Args:
            model (nn.Module): The segmentation model.
            loader (DataLoader): Labeled loader (first samples are taken).
            device (torch.device): Computing device.
            class_names (list[str]): Human-readable class names (unused, kept
                for interface symmetry).
            filename (str): Output file name.
            num_samples (int): Number of samples to display.
            use_amp (bool): Whether to run inference under AMP.

        Returns:
            Path: Path of the saved figure.

        Raises:
            ValueError: If the loader yields no labeled samples.
        """
        collected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            if labels is None:
                continue
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(imgs.to(device), mode="class")
            preds = torch.argmax(logits, dim=1).cpu()
            for k in range(imgs.shape[0]):
                collected.append((imgs[k], labels[k], preds[k]))
                if len(collected) >= num_samples:
                    break
            if len(collected) >= num_samples:
                break
        if not collected:
            raise ValueError("plot_prediction_overlay found no labeled samples")

        n = len(collected)
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1:
            axes = axes[None, :]
        for row, (img, label, pred) in enumerate(collected):
            axes[row, 0].imshow(self._rgb_composite(img))
            axes[row, 0].set_title("Input (B2, B3, B4)")
            axes[row, 1].imshow(label.numpy(), cmap="Greens", vmin=0, vmax=1)
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].imshow(pred.numpy(), cmap="Greens", vmin=0, vmax=1)
            axes[row, 2].set_title("Prediction")
            for ax in axes[row]:
                ax.axis("off")
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _rgb_composite(img: torch.Tensor) -> np.ndarray:
        """Percentile-stretches the first three channels into a displayable RGB."""
        arr = img[:3].numpy()
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        arr = (arr - lo) / max(hi - lo, 1e-6)
        return np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)
