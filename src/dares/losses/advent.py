"""ADVENT losses: entropy-map computation and explicit entropy minimization.

Implements the entropy-based alignment losses of Tsai et al. (2019),
"Learning to Adapt Structured Output Space for Semantic Segmentation"
(ADVENT): a per-pixel Shannon entropy map of the softmax predictions that
feeds a domain discriminator, plus an explicit entropy-minimization term on
the unlabeled target domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS: float = 1e-8


def entropy_map(logits: torch.Tensor) -> torch.Tensor:
    """Computes the per-pixel Shannon entropy map from class logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw class logits of shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Per-pixel entropy map of shape ``(B, 1, H, W)`` in natural units
        (nats), where ``1`` denotes maximal uncertainty and ``0`` perfect
        confidence.
    """
    p = F.softmax(logits, dim=1)
    ent = -(p * (p + _EPS).log()).sum(dim=1, keepdim=True)
    return ent


class EntropyLoss(nn.Module):
    """Mean Shannon entropy of the softmax prediction map.

    Used by the ADVENT engine to explicitly minimize the prediction entropy on
    the unlabeled target domain, pushing the model towards confident and
    unambiguous decisions.
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes the mean per-pixel entropy over the batch.

        Parameters
        ----------
        logits : torch.Tensor
            Raw class logits of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar entropy loss, the mean of :func:`entropy_map` over the
            batch and spatial dimensions.
        """
        return entropy_map(logits).mean()
