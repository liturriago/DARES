"""Tests for the DARES data loading package (dataset, transforms, collate, loader)."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dares.config import DataConfig
from dares.data import (
    DARESDataLoader,
    HDF5Dataset,
    Normalize,
    RandomHorizontalFlip,
    RandomRotate90,
    RandomVerticalFlip,
    SegmentationTransforms,
    dares_collate,
)


def make_h5(
    file_path: Path,
    num_patches: int = 4,
    height: int = 64,
    width: int = 64,
    include_masks: bool = True,
) -> None:
    """Writes a tiny synthetic HDF5 container (images + optional masks)."""
    file_path = Path(file_path)
    with h5py.File(file_path, "w") as f:
        images = np.random.rand(num_patches, 4, height, width).astype(np.float32)
        f.create_dataset(
            "images",
            data=images,
            compression="lzf",
            chunks=(1, 4, height, width),
        )
        if include_masks:
            masks = np.random.randint(0, 2, size=(num_patches, height, width)).astype(
                np.uint8
            )
            f.create_dataset(
                "masks",
                data=masks,
                compression="lzf",
                chunks=(1, height, width),
            )


def test_h5_dataset_labeled(tmp_path):
    """Labeled dataset returns (image, mask) with the documented dtypes/shapes."""
    file_path = tmp_path / "source_train.h5"
    make_h5(file_path, num_patches=4, height=64, width=64)

    transforms = SegmentationTransforms(train=False, mean=(0, 0, 0, 0), std=(1, 1, 1, 1))
    dataset = HDF5Dataset(file_path, transform=transforms.inference_transform, labeled=True)

    assert len(dataset) == 4
    assert dataset.num_patches == 4
    assert dataset.patch_size == 64
    assert dataset.has_labels is True
    assert dataset.classes == ["non_forest", "forest"]

    image, mask = dataset[0]
    assert image.shape == (4, 64, 64)
    assert image.dtype == torch.float32
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.int64
    assert set(torch.unique(mask).tolist()) <= {0, 1}


def test_h5_dataset_unlabeled(tmp_path):
    """Unlabeled dataset never reads the mask key and yields mask=None."""
    file_path = tmp_path / "target_train.h5"
    make_h5(file_path)

    dataset = HDF5Dataset(file_path, labeled=False)

    assert dataset.has_labels is False
    for index in range(len(dataset)):
        image, mask = dataset[index]
        assert image.shape == (4, 64, 64)
        assert mask is None


def test_h5_dataset_labeled_missing_masks_raises(tmp_path):
    """labeled=True on a mask-less container raises KeyError."""
    file_path = tmp_path / "target_train.h5"
    make_h5(file_path, include_masks=False)

    dataset = HDF5Dataset(file_path, labeled=True)

    with pytest.raises(KeyError):
        dataset[0]


def test_transform_pipeline_geometry_and_shapes(tmp_path):
    """A full epoch of augmented __getitem__ keeps shapes and image/mask geometry."""
    height = width = 64
    num_patches = 8
    file_path = tmp_path / "source_train.h5"
    with h5py.File(file_path, "w") as f:
        images = np.zeros((num_patches, 4, height, width), dtype=np.float32)
        masks = np.random.randint(0, 2, size=(num_patches, height, width)).astype(
            np.uint8
        )
        for i in range(num_patches):
            images[i, 0] = masks[i]
            images[i, 1:] = 0.5
        f.create_dataset(
            "images",
            data=images,
            compression="lzf",
            chunks=(1, 4, height, width),
        )
        f.create_dataset(
            "masks",
            data=masks,
            compression="lzf",
            chunks=(1, height, width),
        )

    transforms = SegmentationTransforms(train=True, mean=(0, 0, 0, 0), std=(1, 1, 1, 1))
    dataset = HDF5Dataset(file_path, transform=transforms.train_transform, labeled=True)

    for i in range(len(dataset)):
        image, mask = dataset[i]
        assert image.shape == (4, height, width)
        assert image.dtype == torch.float32
        assert mask.shape == (height, width)
        assert mask.dtype == torch.int64
        torch.testing.assert_close(image[0], mask.float(), atol=0, rtol=0)


def test_random_flips_known_mapping():
    """Flips/rotations reproduce the known pixel mapping on a crafted tensor."""
    height = width = 6
    grid = torch.arange(height * width, dtype=torch.float32).view(height, width)
    image = grid.unsqueeze(0).repeat(4, 1, 1)
    mask = grid.clone()

    vflip = RandomVerticalFlip(p=1.0)
    flipped_image, flipped_mask = vflip(image, mask)
    assert torch.equal(flipped_image[0], torch.flip(grid, dims=(0,)))
    assert torch.equal(flipped_mask, torch.flip(grid, dims=(0,)))

    hflip = RandomHorizontalFlip(p=1.0)
    flipped_image, flipped_mask = hflip(image, mask)
    assert torch.equal(flipped_image[0], torch.flip(grid, dims=(1,)))
    assert torch.equal(flipped_mask, torch.flip(grid, dims=(1,)))

    rot = RandomRotate90(p=1.0)
    rotated_image, rotated_mask = rot(image, mask)
    candidates = [torch.rot90(grid, k=k, dims=(-2, -1)) for k in (1, 2, 3)]
    assert any(torch.equal(rotated_image[0], candidate) for candidate in candidates)
    assert torch.equal(rotated_image[0], rotated_mask)


def test_normalize():
    """Normalize applies (image - mean) / std per channel with a std==0 guard."""
    image = torch.zeros(4, 2, 2)
    for c in range(4):
        image[c] = float(c) * 2.0 + 1.0
    mean = [1.0, 2.0, 3.0, 4.0]
    std = [2.0, 2.0, 4.0, 0.0]

    normalized, mask = Normalize(mean=mean, std=std)(image, None)

    expected = torch.zeros(4, 2, 2)
    for c in range(4):
        expected[c] = (float(c) * 2.0 + 1.0 - mean[c]) / (std[c] if std[c] != 0 else 1.0)
    torch.testing.assert_close(normalized, expected)
    assert mask is None


def test_dares_collate():
    """dares_collate stacks labeled batches, passes None through, rejects mixed."""
    image = torch.randn(4, 8, 8)
    mask = torch.randint(0, 2, (8, 8))

    images, masks = dares_collate([(image, mask), (image, mask)])
    assert images.shape == (2, 4, 8, 8)
    assert masks.shape == (2, 8, 8)
    assert masks.dtype == torch.int64

    images, masks = dares_collate([(image, None), (image, None)])
    assert images.shape == (2, 4, 8, 8)
    assert masks is None

    with pytest.raises(ValueError):
        dares_collate([(image, mask), (image, None)])


def _make_six_containers(tmp_path: Path) -> None:
    for domain in ("source", "target"):
        for filename in (f"{domain}_train.h5", f"{domain}_val.h5", f"{domain}_test.h5"):
            make_h5(tmp_path / filename, num_patches=4, height=64, width=64)


def test_dares_loader_source_and_target(tmp_path):
    """DARESDataLoader builds source (labeled) and target (unlabeled train) batches."""
    _make_six_containers(tmp_path)
    config = DataConfig(
        source_dir=tmp_path,
        target_dir=tmp_path,
        batch_size=2,
        num_workers=0,
    )
    loader = DARESDataLoader(config)

    source_loaders = loader.get_source_loaders()
    assert set(source_loaders) == {"train", "validation", "test"}
    images, masks = next(iter(source_loaders["train"]))
    assert images.shape == (2, 4, 64, 64)
    assert masks.shape == (2, 64, 64)
    assert masks.dtype == torch.int64

    target_loaders = loader.get_target_loaders()
    assert set(target_loaders) == {"train", "validation", "test"}
    images, masks = next(iter(target_loaders["train"]))
    assert images.shape == (2, 4, 64, 64)
    assert masks is None
    images, masks = next(iter(target_loaders["validation"]))
    assert images.shape == (2, 4, 64, 64)
    assert masks.shape == (2, 64, 64)


def test_dares_loader_missing_file_raises(tmp_path):
    """A missing split container raises FileNotFoundError with the resolved path."""
    config = DataConfig(
        source_dir=tmp_path,
        target_dir=tmp_path,
        batch_size=2,
        num_workers=0,
    )
    loader = DARESDataLoader(config)

    with pytest.raises(FileNotFoundError) as exc_info:
        loader.get_source_loaders()
    assert "source_train.h5" in str(exc_info.value)


def test_dares_loader_unknown_domain_raises(tmp_path):
    """An unknown domain name raises ValueError."""
    config = DataConfig(
        source_dir=tmp_path,
        target_dir=tmp_path,
        batch_size=2,
        num_workers=0,
    )
    loader = DARESDataLoader(config)

    with pytest.raises(ValueError):
        loader.get_split_loaders(domain="nope")


def test_data_config_resolved_dir(tmp_path):
    """resolved_dir joins relative dirs to data_root and leaves absolute paths."""
    root = tmp_path / "input"
    relative = DataConfig(
        source_dir="source", target_dir="target", data_root=root
    )
    assert relative.resolved_dir("source_dir") == root / "source"
    assert relative.resolved_dir("target_dir") == root / "target"

    kaggle = DataConfig(
        source_dir="/kaggle/input/dares-data/source",
        target_dir="/kaggle/input/dares-data/target",
        data_root=root,
    )
    assert kaggle.resolved_dir("source_dir") == Path("/kaggle/input/dares-data/source")
    assert kaggle.resolved_dir("target_dir") == Path("/kaggle/input/dares-data/target")

    no_root = DataConfig(source_dir="source", target_dir="target")
    assert no_root.resolved_dir("source_dir") == Path("source")


def test_dares_loader_resolves_data_root(tmp_path):
    """DARESDataLoader resolves relative dirs against DataConfig.data_root."""
    root = tmp_path / "kaggle_input"
    for domain in ("source", "target"):
        (root / domain).mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            make_h5(root / domain / f"{domain}_{split}.h5", num_patches=2)
    config = DataConfig(
        source_dir="source",
        target_dir="target",
        data_root=root,
        batch_size=2,
        num_workers=0,
    )
    loader = DARESDataLoader(config)
    source_loaders = loader.get_source_loaders()
    images, masks = next(iter(source_loaders["train"]))
    assert images.shape == (2, 4, 64, 64)
    assert masks.shape == (2, 64, 64)
