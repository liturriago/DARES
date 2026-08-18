"""Builds source / target DataLoaders from the DARES HDF5 containers.

Convenience re-exports: ``HDF5Dataset``, ``dares_collate`` and
``SegmentationTransforms`` are also available from this module.
"""

from pathlib import Path

from torch.utils.data import DataLoader

from dares.config import DataConfig
from dares.data.collate import dares_collate
from dares.data.h5_dataset import HDF5Dataset
from dares.data.transforms import SegmentationTransforms

SPLIT_FILENAMES: dict[str, str] = {
    "train": "{}_train.h5",
    "validation": "{}_val.h5",
    "test": "{}_test.h5",
}

_TARGET_VARIANT_SUFFIX: dict[str, str] = {
    "original": "",
    "low": "_lime_low",
    "medium": "_lime_med",
    "high": "_lime_high",
}

_TARGET_SPLIT_FRAGMENT: dict[str, str] = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

__all__ = [
    "DARESDataLoader",
    "HDF5Dataset",
    "dares_collate",
    "SegmentationTransforms",
    "container_filename",
]


def container_filename(
    domain: str, split: str, target_variant: str = "original"
) -> str:
    """Resolves the HDF5 container filename for a domain and split.

    Source containers always use the base ``source_{train,val,test}.h5``
    naming. Target containers depend on the selected LIME degradation tier:
    ``"original"`` maps to ``target_{train,val,test}.h5`` while the perturbed
    tiers map to ``target_{train,val,test}_lime_{low,med,high}.h5`` (see
    ``Docs/data.md`` section 6.1).

    Args:
        domain (str): ``"source"`` or ``"target"``.
        split (str): ``"train"``, ``"validation"`` or ``"test"``.
        target_variant (str): Target degradation tier (``"original"``,
            ``"low"``, ``"medium"`` or ``"high"``).

    Returns:
        str: The HDF5 container filename for the split.

    Raises:
        ValueError: If ``split`` is not one of the known splits or
            ``target_variant`` is unknown.
    """
    if split not in _TARGET_SPLIT_FRAGMENT:
        raise ValueError(f"unknown split {split!r}; expected train/validation/test")
    if target_variant not in _TARGET_VARIANT_SUFFIX:
        raise ValueError(
            f"unknown target_variant {target_variant!r}; "
            "expected original/low/medium/high"
        )
    if domain == "source":
        return SPLIT_FILENAMES[split].format(domain)
    return (
        f"target_{_TARGET_SPLIT_FRAGMENT[split]}"
        f"{_TARGET_VARIANT_SUFFIX[target_variant]}.h5"
    )


class DARESDataLoader:
    """Manager that builds source / target split loaders from a DataConfig.

    Args:
        config (DataConfig): Data configuration providing the domain
            directories, batch size, worker count, normalization statistics
            and augmentation flag.

    Attributes:
        config (DataConfig): The stored data configuration.
        transforms (SegmentationTransforms): The train / inference transform
            pipelines.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.transforms = SegmentationTransforms(
            train=config.use_augmentation,
            mean=config.mean,
            std=config.std,
        )

    def _build_loader(
        self,
        file_path: Path,
        split: str,
        labeled: bool,
        shuffle: bool,
        augment: bool,
    ) -> DataLoader:
        """Builds a single DataLoader for one split container.

        Args:
            file_path (Path): Path to the HDF5 container.
            split (str): Split name; ``"train"`` splits drop the last partial
                batch to keep batch sizes consistent for the Gram-matrix loss.
            labeled (bool): Whether the split is labeled.
            shuffle (bool): Whether to shuffle the samples each epoch.
            augment (bool): Whether to apply the training (augmented) pipeline.

        Returns:
            DataLoader: The configured DataLoader.
        """
        transform = self.transforms(train=augment)
        dataset = HDF5Dataset(
            file_path,
            transform=transform,
            labeled=labeled,
        )
        worker_kwargs = {}
        if self.config.num_workers > 0:
            worker_kwargs = {"prefetch_factor": 2, "persistent_workers": True}
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=dares_collate,
            drop_last=(split == "train"),
            **worker_kwargs,
        )

    def get_split_loaders(
        self,
        domain: str,
        labeled: bool = True,
        splits: tuple[str, ...] = ("train", "validation", "test"),
    ) -> dict[str, DataLoader]:
        """Builds loaders for the requested splits of one domain.

        The domain directory is resolved via ``config.<domain>_dir`` and split
        files via ``SPLIT_FILENAMES``. Only the ``train`` split is shuffled and
        augmented (augmentation applies to the source domain only).

        Args:
            domain (str): Domain name, ``"source"`` or ``"target"``.
            labeled (bool): Whether the built loaders expose masks.
            splits (tuple[str, ...]): Splits to build (default all three).

        Returns:
            dict[str, DataLoader]: ``{split: DataLoader}`` for each requested
                split.

        Raises:
            ValueError: If ``domain`` is not ``"source"`` or ``"target"``.
            FileNotFoundError: If a split container does not exist in the
                domain directory.
        """
        if domain not in {"source", "target"}:
            raise ValueError(
                f"unknown domain {domain!r}; expected 'source' or 'target'"
            )
        domain_dir = self.config.resolved_dir(f"{domain}_dir")
        loaders: dict[str, DataLoader] = {}
        for split in splits:
            file_path = Path(domain_dir) / container_filename(
                domain, split, self.config.target_variant
            )
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"split container not found at {file_path}"
                )
            loaders[split] = self._build_loader(
                file_path,
                split,
                labeled=labeled,
                shuffle=(split == "train"),
                augment=(domain == "source" and split == "train" and self.config.use_augmentation),
            )
        return loaders

    def get_source_loaders(self) -> dict[str, DataLoader]:
        """Builds the fully labeled source loaders.

        Returns:
            dict[str, DataLoader]: ``{"train", "validation", "test"}`` loaders
                with masks and training-time augmentation on ``train``.
        """
        return self.get_split_loaders(domain="source", labeled=True)

    def get_target_loaders(self) -> dict[str, DataLoader]:
        """Builds the target loaders with an unlabeled training split.

        Returns:
            dict[str, DataLoader]: ``{"train", "validation", "test"}`` loaders;
                ``train`` is unlabeled (masks never used) and deterministic
                (no augmentation), ``validation`` and ``test`` are labeled.
        """
        loaders: dict[str, DataLoader] = {}
        for split in ("train", "validation", "test"):
            loaders[split] = self.get_split_loaders(
                domain="target",
                labeled=(split != "train"),
                splits=(split,),
            )[split]
        return loaders
