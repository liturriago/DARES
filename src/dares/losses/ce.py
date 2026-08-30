"""Pixel-wise cross-entropy loss for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegCrossEntropyLoss(nn.Module):
    """Pixel-wise cross-entropy between logits and class-index masks.

    Accepts ``(B, C, H, W)`` logits with ``(B, H, W)`` int64 masks. Unlabeled
    batches (``masks is None``) yield a zero loss, so the same module can be
    shared by labeled source and unlabeled target pipelines.

    Args:
        ignore_index (int): Class index ignored in the loss. The DARES dataset
            encodes water / wetland pixels as ``255`` (see ``Docs/data.md``),
            so the default matches the container convention.
    """

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, masks: torch.Tensor | None
    ) -> torch.Tensor:
        """Computes the pixel-wise cross-entropy.

        Args:
            logits (torch.Tensor): Logits of shape ``(B, C, H, W)``.
            masks (torch.Tensor | None): Class-index masks of shape
                ``(B, H, W)``, or ``None`` for unlabeled batches.

        Returns:
            torch.Tensor: Scalar loss; ``0.0`` (on ``logits.device``) when
                ``masks`` is ``None``.
        """
        if masks is None:
            return torch.tensor(0.0, device=logits.device)
        return F.cross_entropy(logits, masks.long(), ignore_index=self.ignore_index)
