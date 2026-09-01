"""Tests for the CLAN (category-level adversarial) engine and its losses."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines.clan import CLANTrainer
from dares.losses.clan import (
    CLANDiscriminator,
    clan_adversarial_loss,
    clan_discriminator_loss,
    masked_class_slices,
    target_pseudo_slices,
)
from dares.models import build_model


def make_h5(
    file_path: Path,
    num_patches: int = 4,
    height: int = 64,
    width: int = 64,
    structured: bool = False,
) -> None:
    """Writes a tiny synthetic HDF5 container."""
    file_path = Path(file_path)
    with h5py.File(file_path, "w") as f:
        masks = np.random.randint(0, 2, size=(num_patches, height, width)).astype(
            np.uint8
        )
        images = np.random.rand(num_patches, 4, height, width).astype(np.float32)
        if structured:
            images[:, 0] = masks
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


@pytest.fixture(scope="module")
def loaders(tmp_path_factory):
    """Builds the six HDF5 containers and the paired source / target loaders."""
    tmp_path = tmp_path_factory.mktemp("clan_data")
    for domain, structured in (("source", True), ("target", False)):
        for split, num in (("train", 8), ("val", 4), ("test", 4)):
            make_h5(
                tmp_path / f"{domain}_{split}.h5",
                num_patches=num,
                structured=structured,
            )
    config = DataConfig(
        source_dir=tmp_path,
        target_dir=tmp_path,
        batch_size=2,
        num_workers=0,
    )
    loader = DARESDataLoader(config)
    return loader.get_source_loaders(), loader.get_target_loaders()


@pytest.fixture(scope="module")
def make_model(loaders):
    """Factory returning a fresh, freshly-initialized segmentation model."""

    def _make() -> torch.nn.Module:
        return build_model(
            ModelConfig(
                backbone="resnet50",
                head="resunet",
                in_channels=4,
                num_classes=2,
                pretrained=False,
            )
        )

    return _make


def _make_engine(model: torch.nn.Module, loaders) -> CLANTrainer:
    """Builds a CLAN engine with the standard CPU test configuration."""
    source_loaders, target_loaders = loaders
    config = TrainConfig(
        method="clan",
        epochs=1,
        lr=1e-4,
        lambda_clan=0.1,
        clan_threshold=0.5,
        lr_d=1e-5,
        device="cpu",
        use_amp=False,
        seed=42,
    )
    return CLANTrainer(
        model, source_loaders, target_loaders, config, torch.device("cpu")
    )


def test_masked_class_slices_single_label():
    """A fully-annotated sample yields one masked slice per present class."""
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 0] = 2.0  # class 0 wins everywhere
    masks = torch.zeros(1, 8, 8, dtype=torch.long)
    slices, labels = masked_class_slices(logits, masks)
    assert labels.tolist() == [0]
    assert slices.shape[1] == 1  # single class channel
    torch.testing.assert_close(
        slices[0, 0], F.softmax(logits.float(), dim=1)[0, 0], atol=1e-6, rtol=1e-6
    )


def test_masked_class_slices_excludes_ignore():
    """ignore_index pixels are excluded from the generated class masks."""
    logits = torch.zeros(2, 2, 8, 8)
    masks = torch.randint(0, 2, (2, 8, 8), dtype=torch.long)
    masks[:, 0, 0] = 255
    slices, labels = masked_class_slices(logits, masks)
    assert slices.shape[0] > 0
    # Every slice label is a valid semantic class (never the ignore index).
    assert set(labels.tolist()) <= {0, 1}


def test_target_pseudo_slices_shape():
    """target_pseudo_slices returns slices, pseudo-labels and a pseudo map."""
    logits = torch.randn(2, 2, 16, 16)
    slices, labels, pseudo = target_pseudo_slices(logits, threshold=0.5)
    assert pseudo.shape == (2, 16, 16)
    assert slices.shape[0] == labels.shape[0]
    assert slices.shape[1] == 1


def test_discriminator_output_dimension():
    """CLANDiscriminator outputs (B, num_classes + 1) logits."""
    disc = CLANDiscriminator(num_classes=2)
    x = torch.randn(4, 1, 16, 16)
    y = disc(x)
    assert y.shape == (4, 3)


def test_train_epoch_finite_metrics(make_model, loaders):
    """train_epoch returns finite scalars for every reported metric."""
    engine = _make_engine(make_model(), loaders)
    metrics = engine.train_epoch()
    assert set(metrics) == {
        "loss_total",
        "loss_ce",
        "loss_disc",
        "loss_adv_clan",
        "epoch_time",
    }
    for value in metrics.values():
        assert np.isfinite(value)
    assert metrics["epoch_time"] >= 0.0


def test_clan_adversarial_loss_zero_without_slices():
    """An empty target slice set yields a zero adversarial loss."""
    d_t = torch.empty(0, 3)
    labels_t = torch.empty(0, dtype=torch.long)
    loss = clan_adversarial_loss(d_t, labels_t)
    assert loss.item() == 0.0


def test_clan_discriminator_loss_uses_target_label():
    """Target slices are labeled with the extra num_classes category."""
    d_src = torch.randn(4, 3)
    labels_src = torch.tensor([0, 1, 0, 1])
    d_tgt = torch.randn(4, 3)
    labels_tgt = torch.tensor([0, 1, 0, 1])
    loss = clan_discriminator_loss(
        d_src, labels_src, d_tgt, labels_tgt, num_classes=2, lambda_out=1.0
    )
    assert loss.ndim == 0
    assert np.isfinite(loss.item())


def test_fit_returns_model_with_history(make_model, loaders):
    """fit returns an nn.Module, tracks best mIoU and the loss history."""
    engine = _make_engine(make_model(), loaders)
    result = engine.fit()
    assert isinstance(result, torch.nn.Module)
    assert engine.best_miou > 0.0
    for key in ("loss_total", "loss_ce", "loss_disc", "loss_adv_clan"):
        assert key in engine.history
        assert len(engine.history[key]) == 1
        assert np.isfinite(engine.history[key][0])


def test_discriminator_learns(make_model, loaders):
    """The discriminator receives gradient updates across training epochs."""
    engine = _make_engine(make_model(), loaders)
    before = {
        name: param.detach().clone()
        for name, param in engine.discriminator.named_parameters()
    }
    for _ in range(3):
        metrics = engine.train_epoch()
    after = {
        name: param.detach()
        for name, param in engine.discriminator.named_parameters()
    }

    moved = any(
        not torch.equal(before[name], after[name]) for name in before
    )
    assert moved, "discriminator parameters did not receive updates"

    for key in ("loss_adv_clan", "loss_disc", "loss_total", "loss_ce"):
        assert key in metrics
        assert np.isfinite(metrics[key])
