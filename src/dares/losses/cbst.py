"""Class-balanced self-training (CBST) pseudo-labeling and self-training loss.

CBST (Zou et al., 2018) is a self-training scheme for unsupervised domain
adaptation. The model is trained with supervised cross-entropy on the labeled
source domain and a masked cross-entropy on the unlabeled target domain. The
target pseudo-labels are refreshed every iteration and selected per class with
a balanced top-ratio policy: for each class, the ``topk_ratio``-most-confident
pixels (above ``threshold``) are kept, so frequent classes cannot dominate the
pseudo-label pool.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBSTPseudoLabeling(nn.Module):
    """Generates class-balanced pseudo-labels and per-pixel selection weights.

    Parameters
    ----------
    num_classes : int
        Number of semantic classes.
    topk_ratio : float
        Fraction of the highest-confidence target pixels kept per class
        (default ``0.5``).
    threshold : float
        Minimum softmax confidence a pixel needs to be a pseudo-label
        candidate (default ``0.9``).
    """

    def __init__(
        self,
        num_classes: int,
        topk_ratio: float = 0.5,
        threshold: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.topk_ratio = topk_ratio
        self.threshold = threshold

    def forward(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes class-balanced pseudo-labels and their selection weights.

        Parameters
        ----------
        logits : torch.Tensor
            Target logits of shape ``(B, C, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(pseudo_labels, weights)``; ``pseudo_labels`` has shape
            ``(B, H, W)`` with the argmax class index per pixel, and
            ``weights`` has shape ``(B, H, W)`` and is ``1.0`` only at the
            selected pixels (per class, the top ``topk_ratio`` confident
            candidates above ``threshold``) and ``0.0`` elsewhere.
        """
        probs = F.softmax(logits.float(), dim=1)
        pseudo = torch.argmax(probs, dim=1)
        conf = probs.max(dim=1).values

        weights = torch.zeros_like(
            pseudo, dtype=torch.float32, device=logits.device
        )

        for c in range(self.num_classes):
            mask = pseudo == c
            candidates = mask.nonzero(as_tuple=False)  # (N, 3) -> (b, h, w)
            candidate_conf = conf[mask]  # (N,)
            if candidates.shape[0] == 0:
                continue

            keep = candidate_conf > self.threshold
            kept = candidates[keep]
            kept_conf = candidate_conf[keep]
            n_kept = int(kept.shape[0])
            if n_kept == 0:
                continue

            k = min(int(math.ceil(self.topk_ratio * n_kept)), n_kept)
            k = max(1, k)
            if k >= n_kept:
                selected = kept
            else:
                _, topk_idx = torch.topk(kept_conf, k)
                selected = kept[topk_idx]
            weights[selected[:, 0], selected[:, 1], selected[:, 2]] = 1.0

        return pseudo, weights


class CBSTSelfTrainingLoss(nn.Module):
    """Masked pixel-wise cross-entropy against the target pseudo-labels.

    The loss is the per-pixel cross-entropy between the target logits and the
    pseudo-labels, masked by the class-balanced selection weights and averaged
    only over the selected pixels.

    Parameters
    ----------
    eps : float
        Small constant added to the weight sum to guard against division by
        zero when no pixel is selected (default ``1e-8``).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the masked self-training loss.

        Parameters
        ----------
        logits : torch.Tensor
            Target logits of shape ``(B, C, H, W)``.
        pseudo_labels : torch.Tensor
            Pseudo-labels of shape ``(B, H, W)``.
        weights : torch.Tensor
            Per-pixel selection mask of shape ``(B, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar masked mean loss; ``0.0`` when no pixel is selected.
        """
        per_pixel = (
            F.cross_entropy(logits, pseudo_labels, reduction="none") * weights
        )
        return per_pixel.sum() / (weights.sum() + self.eps)
