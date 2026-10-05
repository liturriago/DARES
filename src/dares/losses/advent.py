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


def entropy_map(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Computes the per-pixel Shannon entropy map from class logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw class logits of shape ``(B, C, H, W)``.
    mask : torch.Tensor | None
        Optional label map of shape ``(B, H, W)``. Pixels equal to
        ``ignore_index`` are zeroed so they do not drive the domain
        discriminator.
    ignore_index : int
        Label value treated as ignore when ``mask`` is given.

    Returns
    -------
    torch.Tensor
        Per-pixel entropy map of shape ``(B, 1, H, W)`` in natural units
        (nats), where ``1`` denotes maximal uncertainty and ``0`` perfect
        confidence.
    """
    p = F.softmax(logits.float(), dim=1)
    ent = -(p * (p + _EPS).log()).sum(dim=1, keepdim=True)
    if mask is not None:
        m = mask.long()
        if tuple(m.shape[-2:]) != tuple(ent.shape[-2:]):
            m = F.interpolate(m.unsqueeze(1).float(), size=ent.shape[-2:], mode="nearest").squeeze(1).long()
        valid = (m != int(ignore_index)).to(ent.dtype).unsqueeze(1)
        ent = ent * valid
    return ent


class EntropyLoss(nn.Module):
    """Mean Shannon entropy of the softmax prediction map.

    Used by the ADVENT engine to explicitly minimize the prediction entropy on
    the unlabeled target domain, pushing the model towards confident and
    unambiguous decisions.
    """

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Computes the mean per-pixel entropy over the batch.

        Parameters
        ----------
        logits : torch.Tensor
            Raw class logits of shape ``(B, C, H, W)``.
        mask : torch.Tensor | None
            Optional label map of shape ``(B, H, W)``; ``ignore_index``
            pixels are excluded from the mean.

        Returns
        -------
        torch.Tensor
            Scalar entropy loss, the mean of :func:`entropy_map` over the
            batch and spatial dimensions (valid pixels only when ``mask``).
        """
        if mask is None:
            return entropy_map(logits).mean()
        ent = entropy_map(logits, mask, self.ignore_index)
        m = mask.long()
        if tuple(m.shape[-2:]) != tuple(ent.shape[-2:]):
            m = F.interpolate(m.unsqueeze(1).float(), size=ent.shape[-2:], mode="nearest").squeeze(1).long()
        valid = (m != self.ignore_index).to(ent.dtype)
        denom = valid.sum().clamp_min(1.0)
        return ent.sum() / denom
