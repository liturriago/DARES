"""Pixel-level metrics for semantic segmentation: confusion matrix, IoU,
DICE, overall accuracy, and the generalized Matthews correlation coefficient
(MCC) for binary Forest / Non-Forest segmentation."""

import math
from typing import Any

import torch
from torch import Tensor

_EPS: float = 1e-8


class MetricTracker:
    """Static helpers for dense per-pixel segmentation metrics.

    All inputs are pixel-level integer class-index tensors (e.g. ``(B, H, W)``
    or ``(N,)``); ``preds`` is expected to come from ``argmax`` of the logits.
    Every per-class metric is guarded against division by zero so that missing
    classes yield ``0.0`` (never NaN).
    """

    @staticmethod
    def compute_confusion_matrix(
        preds: Tensor,
        labels: Tensor,
        num_classes: int,
        ignore_index: int | None = None,
    ) -> Tensor:
        """Computes the multi-class confusion matrix.

        Args:
            preds (Tensor): Predicted class indices of any shape, e.g.
                ``(B, H, W)`` or ``(N,)``.
            labels (Tensor): Ground-truth class indices, same shape as
                ``preds``.
            num_classes (int): Number of classes (``C``).
            ignore_index (int | None): Optional label value excluded from the
                matrix (e.g. ``255`` for masked water / NoData pixels). When
                set, every pixel whose label matches is dropped entirely
                (prediction and label).

        Returns:
            Tensor: Confusion matrix ``conf[c_actual, c_pred]`` of shape
                ``(C, C)`` on ``preds.device``.

        Raises:
            ValueError: If ``num_classes < 1`` or the number of predicted and
                label elements differ.
        """
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
        preds_flat = preds.reshape(-1).long()
        labels_flat = labels.reshape(-1).long()
        if preds_flat.numel() != labels_flat.numel():
            raise ValueError(
                "preds and labels must have the same number of elements, "
                f"got {preds_flat.numel()} vs {labels_flat.numel()}."
            )
        if ignore_index is not None:
            valid = labels_flat != ignore_index
            preds_flat = preds_flat[valid]
            labels_flat = labels_flat[valid]
        indices = labels_flat * num_classes + preds_flat
        return torch.bincount(
            indices, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

    @staticmethod
    def _per_class_counts(conf: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decomposes a confusion matrix into per-class TP / FP / FN / TN."""
        total = conf.sum()
        tp = conf.diag()
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        tn = total - tp - fp - fn
        return tp, fp, fn, tn

    @staticmethod
    def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
        """Element-wise division returning ``0.0`` where the denominator is zero."""
        safe = denominator.clamp_min(_EPS)
        result = numerator / safe
        return torch.where(denominator == 0, torch.zeros_like(result), result)

    @staticmethod
    def compute_iou(
        preds: Tensor,
        labels: Tensor,
        num_classes: int,
        ignore_index: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Computes per-class IoU and the macro mean IoU.

        Args:
            preds (Tensor): Predicted class indices.
            labels (Tensor): Ground-truth class indices.
            num_classes (int): Number of classes.
            ignore_index (int | None): Optional label value excluded from the
                confusion matrix (e.g. ``255``).

        Returns:
            tuple[Tensor, Tensor]: Per-class IoU ``(C,)`` and scalar mIoU
                (mean over all ``C`` classes).
        """
        conf = MetricTracker.compute_confusion_matrix(
            preds, labels, num_classes, ignore_index=ignore_index
        )
        tp, fp, fn, _ = MetricTracker._per_class_counts(conf)
        iou = MetricTracker._safe_divide(tp, tp + fp + fn)
        return iou, iou.mean()

    @staticmethod
    def compute_dice(
        preds: Tensor,
        labels: Tensor,
        num_classes: int,
        ignore_index: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Computes per-class DICE and the macro mean.

        Args:
            preds (Tensor): Predicted class indices.
            labels (Tensor): Ground-truth class indices.
            num_classes (int): Number of classes.
            ignore_index (int | None): Optional label value excluded from the
                confusion matrix (e.g. ``255``).

        Returns:
            tuple[Tensor, Tensor]: Per-class DICE ``(C,)`` and scalar mean.
        """
        conf = MetricTracker.compute_confusion_matrix(
            preds, labels, num_classes, ignore_index=ignore_index
        )
        tp, fp, fn, _ = MetricTracker._per_class_counts(conf)
        dice = MetricTracker._safe_divide(2 * tp, 2 * tp + fp + fn)
        return dice, dice.mean()

    @staticmethod
    def compute_accuracy(
        preds: Tensor, labels: Tensor, ignore_index: int | None = None
    ) -> float:
        """Computes the overall pixel accuracy ``trace(conf) / total``.

        Args:
            preds (Tensor): Predicted class indices.
            labels (Tensor): Ground-truth class indices.
            ignore_index (int | None): Optional label value excluded from the
                computation (e.g. ``255``).

        Returns:
            float: Fraction of correctly predicted valid pixels in ``[0, 1]``.
        """
        if ignore_index is not None:
            preds_flat = preds.reshape(-1).long()
            labels_flat = labels.reshape(-1).long()
            valid = labels_flat != ignore_index
            preds_flat = preds_flat[valid]
            labels_flat = labels_flat[valid]
            preds, labels = preds_flat, labels_flat
        if labels.numel() == 0:
            return 0.0
        num_classes = int(max(int(preds.max()), int(labels.max()))) + 1
        conf = MetricTracker.compute_confusion_matrix(preds, labels, num_classes)
        return float(torch.trace(conf)) / float(conf.sum())

    @staticmethod
    def compute_mcc(
        preds: Tensor,
        labels: Tensor,
        num_classes: int,
        ignore_index: int | None = None,
    ) -> float:
        """Computes the generalized (multi-class) Matthews correlation coefficient.

        Uses ``mcc = (s * c - row @ col) /
        sqrt((s^2 - row @ row) * (s^2 - col @ col))`` with ``s`` the total
        count, ``c`` the trace, and ``row``/``col`` the per-class row and
        column sums of the confusion matrix. This reduces to the standard
        binary formula for ``C = 2``. Returns ``0.0`` when the denominator is
        not positive (degenerate case).

        Args:
            preds (Tensor): Predicted class indices.
            labels (Tensor): Ground-truth class indices.
            num_classes (int): Number of classes.
            ignore_index (int | None): Optional label value excluded from the
                confusion matrix (e.g. ``255``).

        Returns:
            float: MCC in ``[-1, 1]``, ``0.0`` for degenerate cases.
        """
        conf = MetricTracker.compute_confusion_matrix(
            preds, labels, num_classes, ignore_index=ignore_index
        ).to(torch.float64)
        row = conf.sum(dim=1)
        col = conf.sum(dim=0)
        s = float(conf.sum())
        c = float(torch.trace(conf))
        numerator = s * c - float(row @ col)
        denom_sq = (s * s - float(row @ row)) * (s * s - float(col @ col))
        if denom_sq <= 0.0:
            return 0.0
        mcc = numerator / math.sqrt(denom_sq)
        return float(min(max(mcc, -1.0), 1.0))

    @staticmethod
    def compute_precision_recall(
        preds: Tensor,
        labels: Tensor,
        num_classes: int,
        ignore_index: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Computes per-class precision and recall.

        Args:
            preds (Tensor): Predicted class indices.
            labels (Tensor): Ground-truth class indices.
            num_classes (int): Number of classes.
            ignore_index (int | None): Optional label value excluded from the
                confusion matrix (e.g. ``255``).

        Returns:
            tuple[Tensor, Tensor]: Per-class precision ``(C,)`` and per-class
                recall ``(C,)``.
        """
        conf = MetricTracker.compute_confusion_matrix(
            preds, labels, num_classes, ignore_index=ignore_index
        )
        tp, fp, fn, _ = MetricTracker._per_class_counts(conf)
        precision = MetricTracker._safe_divide(tp, tp + fp)
        recall = MetricTracker._safe_divide(tp, tp + fn)
        return precision, recall

    @staticmethod
    def compute_full_metrics(
        preds: Tensor,
        labels: Tensor,
        num_classes: int,
        ignore_index: int | None = None,
    ) -> dict[str, Any]:
        """Computes the full set of segmentation metrics in one pass.

        Args:
            preds (Tensor): Predicted class indices.
            labels (Tensor): Ground-truth class indices.
            num_classes (int): Number of classes.
            ignore_index (int | None): Optional label value excluded from all
                metrics (e.g. ``255`` for masked water / NoData pixels).

        Returns:
            dict[str, Any]: Dictionary with keys ``confusion_matrix``,
                ``per_class`` (containing ``iou``, ``dice``, ``precision``,
                ``recall``, ``accuracy`` and ``support``), ``mIoU``,
                ``mean_dice``, ``overall_acc`` and ``mcc``.
        """
        conf = MetricTracker.compute_confusion_matrix(
            preds, labels, num_classes, ignore_index=ignore_index
        )
        tp, fp, fn, tn = MetricTracker._per_class_counts(conf)
        total = float(conf.sum())

        iou = MetricTracker._safe_divide(tp, tp + fp + fn)
        dice = MetricTracker._safe_divide(2 * tp, 2 * tp + fp + fn)
        precision = MetricTracker._safe_divide(tp, tp + fp)
        recall = MetricTracker._safe_divide(tp, tp + fn)
        per_class_acc = MetricTracker._safe_divide(
            tp + tn, torch.full((num_classes,), float(total))
        )
        support = conf.sum(dim=1)

        return {
            "confusion_matrix": conf,
            "per_class": {
                "iou": iou.tolist(),
                "dice": dice.tolist(),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "accuracy": per_class_acc.tolist(),
                "support": support.tolist(),
            },
            "mIoU": float(iou.mean()),
            "mean_dice": float(dice.mean()),
            "overall_acc": float(torch.trace(conf)) / total,
            "mcc": MetricTracker.compute_mcc(
                preds, labels, num_classes, ignore_index=ignore_index
            ),
        }

    @staticmethod
    def print_summary(
        prefix: str, metrics: dict[str, Any], class_names: list[str]
    ) -> None:
        """Prints a formatted report of segmentation metrics to stdout.

        Args:
            prefix (str): Section heading, e.g. ``"SOURCE VAL"``.
            metrics (dict[str, Any]): Output of ``compute_full_metrics``.
            class_names (list[str]): Human-readable class names.
        """
        per = metrics["per_class"]
        header = (
            f"{'Class':<18}{'IoU':>10}{'DICE':>10}"
            f"{'Prec':>10}{'Recall':>10}{'Support':>10}"
        )
        width = len(header)
        print("=" * width)
        print(f"{prefix}")
        print("=" * width)
        print(header)
        print("-" * width)
        for name, iou, dice, prec, rec, support in zip(
            class_names,
            per["iou"],
            per["dice"],
            per["precision"],
            per["recall"],
            per["support"],
        ):
            print(
                f"{name:<18}{iou:>10.4f}{dice:>10.4f}"
                f"{prec:>10.4f}{rec:>10.4f}{support:>10d}"
            )
        print("-" * width)
        print(
            f"{'mean':<18}{metrics['mIoU']:>10.4f}"
            f"{metrics['mean_dice']:>10.4f}"
        )
        print("=" * width)
