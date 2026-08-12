"""Tests for the DARES metrics module (confusion matrix, IoU, DICE, MCC)."""

import math

import pytest
import torch

from dares.utils.metrics import MetricTracker


def test_perfect_predictions():
    """Perfect predictions yield IoU = DICE = OA = MCC = 1.0."""
    labels = torch.tensor([0] * 50 + [1] * 50, dtype=torch.long)
    preds = labels.clone()
    iou, miou = MetricTracker.compute_iou(preds, labels, 2)
    dice, mdice = MetricTracker.compute_dice(preds, labels, 2)
    assert torch.allclose(iou, torch.ones(2))
    assert torch.allclose(dice, torch.ones(2))
    assert miou.item() == pytest.approx(1.0)
    assert mdice.item() == pytest.approx(1.0)
    assert MetricTracker.compute_accuracy(preds, labels) == pytest.approx(1.0)
    assert MetricTracker.compute_mcc(preds, labels, 2) == pytest.approx(1.0)


def test_hand_computed_confusion_matrix():
    """Hand-verified numerics for confusion matrix [[50, 10], [5, 35]]."""
    labels = torch.tensor([0] * 60 + [1] * 40, dtype=torch.long)
    preds = torch.tensor([0] * 50 + [1] * 10 + [0] * 5 + [1] * 35, dtype=torch.long)
    conf = MetricTracker.compute_confusion_matrix(preds, labels, 2)
    assert conf.tolist() == [[50, 10], [5, 35]]

    iou, miou = MetricTracker.compute_iou(preds, labels, 2)
    dice, mdice = MetricTracker.compute_dice(preds, labels, 2)
    prec, rec = MetricTracker.compute_precision_recall(preds, labels, 2)

    assert iou.tolist() == pytest.approx([50 / 65, 35 / 50], abs=1e-4)
    assert miou.item() == pytest.approx(0.73462, abs=1e-4)
    assert dice.tolist() == pytest.approx([100 / 115, 70 / 85], abs=1e-4)
    assert mdice.item() == pytest.approx(0.84655, abs=1e-4)
    assert prec.tolist() == pytest.approx([50 / 55, 35 / 45], abs=1e-4)
    assert rec.tolist() == pytest.approx([50 / 60, 35 / 40], abs=1e-4)
    assert MetricTracker.compute_accuracy(preds, labels) == pytest.approx(0.85)
    assert MetricTracker.compute_mcc(preds, labels, 2) == pytest.approx(
        0.69752, abs=1e-4
    )


def test_compute_full_metrics_structure():
    """Full metric dictionary matches the hand-verified reference values."""
    labels = torch.tensor([0] * 60 + [1] * 40, dtype=torch.long)
    preds = torch.tensor([0] * 50 + [1] * 10 + [0] * 5 + [1] * 35, dtype=torch.long)
    metrics = MetricTracker.compute_full_metrics(preds, labels, 2)
    assert metrics["confusion_matrix"].tolist() == [[50, 10], [5, 35]]
    assert metrics["per_class"]["support"] == [60, 40]
    assert metrics["per_class"]["accuracy"] == pytest.approx(
        [85 / 100, 85 / 100], abs=1e-4
    )
    assert metrics["mIoU"] == pytest.approx(0.73462, abs=1e-4)
    assert metrics["mean_dice"] == pytest.approx(0.84655, abs=1e-4)
    assert metrics["overall_acc"] == pytest.approx(0.85)
    assert metrics["mcc"] == pytest.approx(0.69752, abs=1e-4)


def test_missing_class_is_not_nan():
    """A class absent from both tensors reports 0.0 (never NaN)."""
    labels = torch.zeros(100, dtype=torch.long)
    preds = torch.zeros(100, dtype=torch.long)
    iou, miou = MetricTracker.compute_iou(preds, labels, 2)
    dice, _ = MetricTracker.compute_dice(preds, labels, 2)
    assert math.isfinite(iou[1].item())
    assert iou[1].item() == 0.0
    assert dice[1].item() == 0.0
    assert miou.item() == pytest.approx(0.5)
    assert MetricTracker.compute_accuracy(preds, labels) == pytest.approx(1.0)
    assert math.isfinite(MetricTracker.compute_mcc(preds, labels, 2))


def test_all_wrong_binary():
    """Every prediction disagrees with its label (TP=0, FP=1, FN=1, TN=0)."""
    labels = torch.tensor([0] * 50 + [1] * 50, dtype=torch.long)
    preds = torch.tensor([1] * 50 + [0] * 50, dtype=torch.long)
    conf = MetricTracker.compute_confusion_matrix(preds, labels, 2)
    assert conf.tolist() == [[0, 50], [50, 0]]
    iou, _ = MetricTracker.compute_iou(preds, labels, 2)
    assert iou.tolist() == [0.0, 0.0]
    assert MetricTracker.compute_accuracy(preds, labels) == pytest.approx(0.0)
    assert MetricTracker.compute_mcc(preds, labels, 2) == pytest.approx(-1.0, abs=1e-4)


def test_multiclass_sanity():
    """Random C=3 tensors keep every metric in its valid range."""
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (4, 8, 8), dtype=torch.long)
    preds = torch.randint(0, 3, (4, 8, 8), dtype=torch.long)
    iou, miou = MetricTracker.compute_iou(preds, labels, 3)
    dice, _ = MetricTracker.compute_dice(preds, labels, 3)
    assert bool(((iou >= 0) & (iou <= 1)).all())
    assert bool(((dice >= 0) & (dice <= 1)).all())
    assert math.isfinite(miou.item())
    mcc = MetricTracker.compute_mcc(preds, labels, 3)
    assert -1.0 <= mcc <= 1.0


def test_batched_shape_matches_flat():
    """(B, H, W) inputs produce the same metrics as their flat versions."""
    torch.manual_seed(1)
    preds = torch.randint(0, 2, (3, 6, 6), dtype=torch.long)
    labels = torch.randint(0, 2, (3, 6, 6), dtype=torch.long)
    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)

    iou_a, miou_a = MetricTracker.compute_iou(preds, labels, 2)
    iou_b, miou_b = MetricTracker.compute_iou(preds_flat, labels_flat, 2)
    conf_a = MetricTracker.compute_confusion_matrix(preds, labels, 2)
    conf_b = MetricTracker.compute_confusion_matrix(preds_flat, labels_flat, 2)
    assert torch.equal(conf_a, conf_b)
    assert torch.allclose(iou_a, iou_b)
    assert miou_a.item() == pytest.approx(miou_b.item())


def test_validation_errors():
    """Mismatched sizes and an invalid num_classes raise ValueError."""
    with pytest.raises(ValueError):
        MetricTracker.compute_confusion_matrix(
            torch.tensor([0, 1]), torch.tensor([0, 1, 0]), 2
        )
    with pytest.raises(ValueError):
        MetricTracker.compute_confusion_matrix(
            torch.tensor([0, 1]), torch.tensor([0, 1]), 0
        )


def test_print_summary(capsys):
    """print_summary renders a report without errors."""
    labels = torch.tensor([0] * 60 + [1] * 40, dtype=torch.long)
    preds = torch.tensor([0] * 50 + [1] * 10 + [0] * 5 + [1] * 35, dtype=torch.long)
    metrics = MetricTracker.compute_full_metrics(preds, labels, 2)
    MetricTracker.print_summary("SOURCE VAL", metrics, ["Forest", "Non-Forest"])
    out = capsys.readouterr().out
    assert "Forest" in out
    assert "Non-Forest" in out
    assert "mIoU" in out
    assert "MCC" in out


def test_device_consistency():
    """The confusion matrix stays on the input device (CPU)."""
    preds = torch.tensor([0, 1, 0], dtype=torch.long)
    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    conf = MetricTracker.compute_confusion_matrix(preds, labels, 2)
    assert conf.device == preds.device
    assert conf.device.type == "cpu"
