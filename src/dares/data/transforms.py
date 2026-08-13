"""Pair-wise (image, mask) transforms for the DARES data pipeline.

Every transform operates on an ``(image, mask)`` tuple where ``mask`` may be
``None`` for unlabeled splits. All transforms MUST pass ``mask=None`` through
unchanged so the unlabeled target training container is never interpreted.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch

Transform = Callable[
    [np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]


class Compose:
    """Chains a sequence of transforms applied to (image, mask) pairs.

    Args:
        transforms (list[Callable]): Transforms applied in order, each taking
            and returning an ``(image, mask)`` tuple.
    """

    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Applies each transform in sequence.

        Args:
            image (np.ndarray | torch.Tensor): Input image of shape
                ``(C, H, W)``.
            mask (np.ndarray | torch.Tensor | None): Input mask of shape
                ``(H, W)`` or ``None`` for unlabeled samples.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: The transformed
                ``(image, mask)`` pair.
        """
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class ToTensor:
    """Converts (image, mask) pairs to torch tensors.

    Images become ``torch.float32`` tensors of shape ``(C, H, W)`` and masks
    become ``torch.int64`` tensors of shape ``(H, W)`` (or stay ``None``).
    """

    def __call__(
        self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Converts the inputs to torch tensors.

        Args:
            image (np.ndarray | torch.Tensor): Input image of shape
                ``(C, H, W)``.
            mask (np.ndarray | torch.Tensor | None): Input mask of shape
                ``(H, W)`` or ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: The ``(image, mask)``
                pair as tensors.
        """
        image = torch.as_tensor(image, dtype=torch.float32)
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.int64)
        return image, mask


class Normalize:
    """Per-channel image normalization.

    Args:
        mean (Sequence[float]): Per-channel mean of length ``C``.
        std (Sequence[float]): Per-channel std of length ``C``; a zero entry
            is replaced by ``1.0`` to avoid division by zero.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32))
        std_t = torch.as_tensor(np.asarray(std, dtype=np.float32))
        std_t = torch.where(std_t == 0.0, torch.ones_like(std_t), std_t)
        self.mean = mean_t.view(-1, 1, 1)
        self.std = std_t.view(-1, 1, 1)

    def __call__(
        self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalizes the image channels; the mask is passed through unchanged.

        Args:
            image (np.ndarray | torch.Tensor): Input image of shape
                ``(C, H, W)``.
            mask (np.ndarray | torch.Tensor | None): Input mask or ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: The normalized
                ``(image, mask)`` pair.
        """
        image = torch.as_tensor(image, dtype=torch.float32)
        image = (image - self.mean) / self.std
        return image, mask


class RandomHorizontalFlip:
    """Randomly flips the (image, mask) pair horizontally.

    Args:
        p (float): Probability of applying the flip.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flips image and mask across the width axis with probability ``p``.

        Args:
            image (np.ndarray | torch.Tensor): Input image of shape
                ``(C, H, W)``.
            mask (np.ndarray | torch.Tensor | None): Input mask of shape
                ``(H, W)`` or ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: The flipped
                ``(image, mask)`` pair.
        """
        image = torch.as_tensor(image)
        if torch.rand(1).item() < self.p:
            image = torch.flip(image, dims=(-1,))
            if mask is not None:
                mask = torch.flip(torch.as_tensor(mask), dims=(-1,))
        return image, mask


class RandomVerticalFlip:
    """Randomly flips the (image, mask) pair vertically.

    Args:
        p (float): Probability of applying the flip.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flips image and mask across the height axis with probability ``p``.

        Args:
            image (np.ndarray | torch.Tensor): Input image of shape
                ``(C, H, W)``.
            mask (np.ndarray | torch.Tensor | None): Input mask of shape
                ``(H, W)`` or ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: The flipped
                ``(image, mask)`` pair.
        """
        image = torch.as_tensor(image)
        if torch.rand(1).item() < self.p:
            image = torch.flip(image, dims=(-2,))
            if mask is not None:
                mask = torch.flip(torch.as_tensor(mask), dims=(-2,))
        return image, mask


class RandomRotate90:
    """Randomly rotates the (image, mask) pair by a multiple of 90 degrees.

    Args:
        p (float): Probability of applying the rotation.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Rotates image and mask by a random k in {1, 2, 3} with probability ``p``.

        Args:
            image (np.ndarray | torch.Tensor): Input image of shape
                ``(C, H, W)``.
            mask (np.ndarray | torch.Tensor | None): Input mask of shape
                ``(H, W)`` or ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: The rotated
                ``(image, mask)`` pair.
        """
        image = torch.as_tensor(image)
        if torch.rand(1).item() < self.p:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k=k, dims=(-2, -1))
            if mask is not None:
                mask = torch.rot90(torch.as_tensor(mask), k=k, dims=(-2, -1))
        return image, mask


class SegmentationTransforms:
    """Builder of the train / inference (image, mask) transform pipelines.

    Args:
        train (bool): Whether the object defaults to the training pipeline.
        mean (Sequence[float]): Per-channel normalization mean.
        std (Sequence[float]): Per-channel normalization std.

    Attributes:
        train_transform (Compose): Augmented pipeline (rotations, flips,
            ToTensor, Normalize).
        inference_transform (Compose): ToTensor + Normalize pipeline.
    """

    def __init__(self, train: bool, mean: Sequence[float], std: Sequence[float]) -> None:
        self.train = train
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.train_transform = Compose(
            [
                RandomRotate90(),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                Normalize(self.mean, self.std),
            ]
        )
        self.inference_transform = Compose(
            [
                ToTensor(),
                Normalize(self.mean, self.std),
            ]
        )

    def __call__(self, train: bool | None = None) -> Compose:
        """Returns the requested transform pipeline.

        Args:
            train (bool | None): Whether to return the training (augmented)
                pipeline; defaults to the value passed at construction time.

        Returns:
            Compose: The selected ``(image, mask)`` transform pipeline.
        """
        use_train = self.train if train is None else train
        return self.train_transform if use_train else self.inference_transform
