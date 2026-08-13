"""Collation of (image, mask) batches that tolerates unlabeled samples."""

import torch

from dares.data.h5_dataset import Sample


def dares_collate(
    batch: list[Sample],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Collates a batch of ``(image, mask)`` pairs into tensors.

    All masks must be present or all masks must be ``None``; mixed batches
    indicate an inconsistent split configuration and are rejected.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor | None]]): Batch of
            ``(image, mask)`` samples from an ``HDF5Dataset``.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: Stacked ``images`` of shape
            ``(B, C, H, W)`` and stacked ``masks`` of shape ``(B, H, W)``, or
            ``None`` if every sample is unlabeled.

    Raises:
        ValueError: If the batch mixes labeled and unlabeled samples.
    """
    images = torch.stack([sample[0] for sample in batch])
    masks = [sample[1] for sample in batch]
    if all(mask is not None for mask in masks):
        masks = torch.stack(masks)
    elif all(mask is None for mask in masks):
        masks = None
    else:
        raise ValueError(
            "dares_collate received a mixed batch of labeled and unlabeled "
            "samples; configure each split with a consistent `labeled` flag"
        )
    return images, masks
