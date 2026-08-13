"""Lazy, picklable PyTorch Dataset over a single HDF5 container.

On Windows, DataLoader workers are spawned, so every worker receives a pickled
copy of the dataset. An open ``h5py.File`` handle cannot be pickled, therefore
the handle is opened on first access and dropped during pickling so each
worker re-opens the file in its own process.
"""

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch

from dares.data.transforms import Transform

Sample = tuple[torch.Tensor, torch.Tensor | None]


class HDF5Dataset(torch.utils.data.Dataset):
    """PyTorch dataset reading patches from a single HDF5 container.

    The container is expected to hold an ``images`` dataset of shape
    ``(N, C, H, W)`` (float32) and, for labeled splits, a ``masks`` dataset of
    shape ``(N, H, W)`` (uint8). Images and masks are returned as ``(C, H, W)``
    and ``(H, W)`` tensors respectively, or ``mask`` is ``None`` for unlabeled
    splits.

    Args:
        file_path (str | Path): Path to the HDF5 container.
        transform (Callable | None): Optional ``(image, mask) -> (image, mask)``
            transform applied to every sample.
        image_key (str): Name of the images dataset (default ``"images"``).
        mask_key (str): Name of the masks dataset (default ``"masks"``).
        labeled (bool): Whether the split is labeled. When ``False`` the mask
            dataset is never read and every sample yields ``mask=None``.

    Attributes:
        file_path (Path): Path to the HDF5 container.
        image_key (str): Images dataset name.
        mask_key (str): Masks dataset name.
        labeled (bool): Whether the split is treated as labeled.
        has_labels (bool): Whether the split exposes labels (== ``labeled``).
        num_patches (int): Number of patches in the container.
        patch_size (int): Spatial height of the patches (e.g. 224).
        classes (list[str]): ``["non_forest", "forest"]`` (Class 0 then 1).

    Raises:
        KeyError: If ``labeled`` is ``True`` and the mask dataset is missing
            from the container.
    """

    classes: list[str] = ["non_forest", "forest"]

    def __init__(
        self,
        file_path: str | Path,
        transform: Callable | None = None,
        image_key: str = "images",
        mask_key: str = "masks",
        labeled: bool = True,
    ) -> None:
        self.file_path = Path(file_path)
        self.image_key = image_key
        self.mask_key = mask_key
        self.labeled = labeled
        self.has_labels = labeled
        self.transform: Callable | None = transform
        self._file: h5py.File | None = None
        self._dataset_length: int | None = None
        self._patch_size: int | None = None

    def __getstate__(self) -> dict:
        """Returns the picklable state with the open HDF5 handle dropped."""
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restores the state ensuring no stale HDF5 handle survives."""
        self.__dict__.update(state)
        self._file = None

    def _ensure_open(self) -> h5py.File:
        """Returns the lazily-opened read-only HDF5 handle."""
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __len__(self) -> int:
        """Returns the number of patches in the container."""
        if self._dataset_length is None:
            file = self._ensure_open()
            self._dataset_length = int(file[self.image_key].shape[0])
        return self._dataset_length

    @property
    def num_patches(self) -> int:
        """int: Number of patches in the container."""
        return len(self)

    @property
    def patch_size(self) -> int:
        """int: Spatial size of the patches read from the HDF5 shape."""
        if self._patch_size is None:
            file = self._ensure_open()
            self._patch_size = int(file[self.image_key].shape[2])
        return self._patch_size

    def __getitem__(self, index: int) -> Sample:
        """Loads and transforms a single patch.

        Args:
            index (int): Patch index in the container.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: The ``(image, mask)``
                pair; ``image`` is ``(C, H, W)`` float32 and ``mask`` is
                ``(H, W)`` int64 (or ``None`` for unlabeled splits).

        Raises:
            KeyError: If ``labeled`` is ``True`` and the mask dataset does not
                exist in the container.
        """
        file = self._ensure_open()
        image = np.asarray(file[self.image_key][index], dtype=np.float32)
        if self.labeled:
            if self.mask_key not in file:
                raise KeyError(
                    f"mask key {self.mask_key!r} not found in {self.file_path}; "
                    "labeled=True requires a mask dataset (misconfigured split)"
                )
            mask = np.asarray(file[self.mask_key][index], dtype=np.uint8)
        else:
            mask = None
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask
