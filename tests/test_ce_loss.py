"""Tests for the pixel-wise cross-entropy loss and its ignore_index handling.

The DARES dataset encodes water / wetland pixels as ``255`` in the masks
(see ``Docs/data.md``), so ``SegCrossEntropyLoss`` must ignore them by
default instead of treating them as a class index.
"""

import torch

from dares.losses.ce import SegCrossEntropyLoss


def test_ce_defaults_to_ignore_index_255():
    """The default ignore_index matches the dataset's water sentinel."""
    loss = SegCrossEntropyLoss()
    assert loss.ignore_index == 255


def test_masked_pixels_do_not_contribute():
    """Pixels labeled 255 are excluded from the loss average."""
    logits = torch.tensor(
        [
            [
                [0.0, 10.0, 0.0, 0.0],
                [10.0, 0.0, 5.0, 5.0],
            ]
        ]
    )  # (1, 2, 4) two valid + two water pixels
    masks = torch.tensor([[0, 1, 255, 255]])  # (1, 4)

    loss = SegCrossEntropyLoss(ignore_index=255)
    value = loss(logits, masks)

    # Softmax CE on the two valid pixels (class 0 then class 1).
    p_valid = torch.softmax(logits[:, :, :2], dim=1)
    expected = -(p_valid[0, 0, 0].log() + p_valid[0, 1, 1].log()) / 2.0
    assert value.item() == expected.item()


def test_maskless_batches_yield_zero():
    """Unlabeled batches (mask None) produce a zero loss."""
    logits = torch.randn(2, 2, 8, 8)
    loss = SegCrossEntropyLoss()
    value = loss(logits, None)
    assert value.item() == 0.0