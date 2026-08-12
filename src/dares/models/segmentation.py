"""Composition of a backbone encoder and a segmentation head decoder."""

import torch
import torch.nn as nn

from dares.config import ModelConfig
from dares.models.backbones.registry import build_backbone
from dares.models.heads.registry import build_head


class SegmentationModel(nn.Module):
    """Full segmentation model combining a backbone encoder and a head decoder.

    The model maps an input tensor ``x`` of shape ``(B, C_in, H, W)`` to class
    logits, dense features, or both, depending on the forward ``mode``.

    Args:
        config (ModelConfig): Model configuration providing the backbone and
            head names plus all architecture hyperparameters.

    Attributes:
        backbone (nn.Module): Multi-scale backbone encoder producing the
            ``"S2"`` ... ``"S32"`` feature dict.
        head (nn.Module): Dense decoder mapping the features to a tuple
            ``(features_fullres, logits)`` at the full input resolution.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = build_backbone(config.backbone, config)
        self.head = build_head(config.head, self.backbone.out_channels, config)

    @property
    def out_channels(self) -> dict[str, int]:
        """dict[str, int]: Number of output channels per backbone feature level."""
        return self.backbone.out_channels

    @property
    def feature_dim(self) -> int:
        """int: Dimension of the dense feature map returned in 'feature' mode."""
        return self.head.feature_dim

    def forward(
        self, x: torch.Tensor, mode: str = "class"
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Runs the backbone and head with the requested output mode.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C_in, H, W)`` with
                ``H`` and ``W`` multiples of 32.
            mode (str): Output mode, one of ``"class"``, ``"feature"`` or
                ``"both"`` (default ``"class"``).

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Class logits of
                shape ``(B, C, H, W)`` for ``"class"``, dense features of shape
                ``(B, D, H, W)`` for ``"feature"``, or a ``(features, logits)``
                tuple for ``"both"``.

        Raises:
            ValueError: If ``mode`` is not one of the supported modes.
        """
        features, logits = self.head(self.backbone(x))
        if mode == "class":
            return logits
        if mode == "feature":
            return features
        if mode == "both":
            return features, logits
        raise ValueError(
            f"Unknown mode {mode!r}; expected 'class', 'feature' or 'both'"
        )

    def freeze_backbone(self) -> None:
        """Disables gradient computation for all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Re-enables gradient computation for all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_model(config: ModelConfig) -> SegmentationModel:
    """Builds a complete segmentation model from a model configuration.

    Args:
        config (ModelConfig): Model configuration.

    Returns:
        SegmentationModel: The composed backbone + head segmentation model.
    """
    return SegmentationModel(config)
