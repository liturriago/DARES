"""Shared model evaluation helper for the DARES pipeline."""

from typing import Any

import torch
from torch.amp import autocast

from dares.utils.metrics import MetricTracker


@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    use_amp: bool = False,
    prefix: str = "Evaluation",
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Runs pixel-level inference and computes the full metric set.

    Args:
        model (nn.Module): The segmentation model (``mode='class'``).
        loader (DataLoader): Labeled loader to evaluate.
        device (torch.device): Computing device.
        num_classes (int): Number of output classes.
        class_names (list[str]): Human-readable class names.
        use_amp (bool): Whether to run inference under automatic mixed precision.
        prefix (str): Label for the printed report.
        ignore_index (int): Label value excluded from the metrics (``255`` for
            water / NoData pixels per the DARES dataset contract).

    Returns:
        dict[str, Any]: The full metrics dict from
            ``MetricTracker.compute_full_metrics`` (confusion matrix, per-class
            IoU/DICE/precision/recall, mIoU, mean DICE, overall accuracy, MCC).
    """
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1]
        if labels is None:
            continue
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs, mode="class")
        all_preds.append(torch.argmax(logits, dim=1).reshape(-1).cpu())
        all_labels.append(labels.reshape(-1).long())

    if not all_preds:
        raise ValueError(
            "evaluate_segmentation received no labeled batches; "
            "the loader has no masks to evaluate against."
        )

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrics = MetricTracker.compute_full_metrics(
        preds, labels, num_classes, ignore_index=ignore_index
    )
    MetricTracker.print_summary(prefix, metrics, class_names)
    return metrics


def metrics_to_jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Converts a metrics dict (containing a torch tensor) to JSON-serializable
    primitives.

    Args:
        metrics (dict[str, Any]): Output of ``MetricTracker.compute_full_metrics``.

    Returns:
        dict[str, Any]: The same dict with the confusion matrix converted to a
            nested Python list.
    """
    out = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
    out["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    return out
