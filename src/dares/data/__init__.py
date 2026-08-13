"""
Data loading package for the DARES framework.

Every split is stored as an HDF5 (``.h5``) container with LZF compression:

* ``images``  ->  ``(N, 4, H, W)`` float32 surface reflectance (B2, B3, B4, B8).
* ``masks``   ->  ``(N, H, W)`` uint8 ground truth (0 = Non-Forest, 1 = Forest);
  absent for the unlabeled target training container.

The package provides:

* ``HDF5Dataset``  ->  lazy, worker-safe ``torch.utils.data.Dataset`` reading a
  single HDF5 container.
* ``SegmentationTransforms``  ->  pair-wise (image, mask) augmentations and
  per-channel normalization.
* ``dares_collate``  ->  collate fn that tolerates unlabeled (mask-less)
  samples.
* ``DARESDataLoader``  ->  manager that builds source/target split loaders from
  the ``source_*.h5`` / ``target_*.h5`` files in each domain directory.

Datasets yield ``(image, mask)`` pairs where ``image`` is ``(4, H, W)`` float32
and ``mask`` is ``(H, W)`` int64 (or ``None`` for unlabeled splits), matching
the model contract in ``dares.models``.
"""
from dares.data.collate import dares_collate
from dares.data.h5_dataset import HDF5Dataset
from dares.data.loader import DARESDataLoader
from dares.data.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotate90,
    RandomVerticalFlip,
    SegmentationTransforms,
)

__all__ = [
    "HDF5Dataset",
    "SegmentationTransforms",
    "Compose",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotate90",
    "dares_collate",
    "DARESDataLoader",
]
