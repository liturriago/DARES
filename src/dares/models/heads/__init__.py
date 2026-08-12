"""
Segmentation heads / decoders for DARES.

Each head is a ``torch.nn.Module`` that consumes the multi-scale feature dict
produced by a DARES backbone (keys ``"S2"`` ... ``"S32"``) and returns a tuple
``(features, logits)``:

* ``features`` ->  dense feature map ``(B, D, H, W)`` at the **full input
  resolution**; ``D`` is the embedding dimensionality used by the DARES
  alpha-Renyi alignment loss (Gram matrix computation).
* ``logits``   ->  pixel-wise class logits ``(B, C, H, W)`` at the **full input
  resolution**.

Heads are constructed with ``(backbone_out_channels: dict[str, int], config)``.
"""

from dares.models.heads.deeplabv3p import DeepLabV3PHead
from dares.models.heads.registry import build_head
from dares.models.heads.resunet import ResUNetHead

__all__ = ["ResUNetHead", "DeepLabV3PHead", "build_head"]
