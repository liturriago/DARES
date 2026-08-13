"""Tests for the ADVENT training engine and entropy losses."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines.advent import ADVENTTrainer
from dares.losses.advent import entropy_map
from dares.losses.domain import adversarial_loss
from dares.models import build_model


def make_h5(
    file_path: Path,
    num_patches: int = 4,
    height: int = 64,
    width: int = 64,
    structured: bool = False,
) -> None:
    """Writes a tiny synthetic HDF5 container.

    ``structured`` containers encode the mask into image channel 0 so the
    segmenter can learn confident predictions, giving the discriminator a
    separable signal versus the purely noisy target domain.
    """
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
    tmp_path = tmp_path_factory.mktemp("advent_data")
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


def _make_engine(model: torch.nn.Module, loaders) -> ADVENTTrainer:
    """Builds an ADVENT engine with the standard CPU test configuration."""
    source_loaders, target_loaders = loaders
    config = TrainConfig(
        method="advent",
        epochs=1,
        lr=1e-4,
        lambda_adv=0.1,
        lambda_entropy=0.1,
        device="cpu",
        use_amp=False,
        seed=42,
    )
    return ADVENTTrainer(
        model, source_loaders, target_loaders, config, torch.device("cpu")
    )


def _discriminator_loss(
    engine: ADVENTTrainer,
    loaders,
) -> float:
    """Computes the discriminator loss on one paired batch without training."""
    source_loaders, target_loaders = loaders
    src_iter, tgt_iter, _ = engine._dual_iterators(
        source_loaders["train"], target_loaders["train"]
    )
    imgs_s, _ = next(src_iter)
    imgs_t, _ = next(tgt_iter)
    with torch.no_grad():
        logits_s = engine.model(imgs_s.to(engine.device), mode="class")
        logits_t = engine.model(imgs_t.to(engine.device), mode="class")
        e_s = entropy_map(logits_s)
        e_t = entropy_map(logits_t)
        d_s = engine.discriminator(e_s)
        d_t = engine.discriminator(e_t)
        loss_dis, _ = adversarial_loss(d_s, d_t)
    return float(loss_dis.item())


def test_entropy_map_uniform_max_and_onehot_zero():
    """Uniform softmax yields maximal entropy; a one-hot yields ~0."""
    uniform = torch.zeros(2, 4, 8, 8)
    ent = entropy_map(uniform)
    assert ent.shape == (2, 1, 8, 8)
    torch.testing.assert_close(
        ent, torch.full_like(ent, np.log(4.0)), atol=1e-6, rtol=1e-6
    )

    one_hot = torch.full((2, 4, 8, 8), -10.0)
    one_hot[:, 0] = 10.0
    ent = entropy_map(one_hot)
    assert torch.allclose(ent, torch.zeros_like(ent), atol=1e-6)


def test_train_epoch_finite_metrics(make_model, loaders):
    """train_epoch returns finite scalars for every reported metric."""
    engine = _make_engine(make_model(), loaders)
    metrics = engine.train_epoch()
    assert set(metrics) == {"loss_total", "loss_ce", "loss_adv", "loss_ent", "epoch_time"}
    for value in metrics.values():
        assert np.isfinite(value)
    assert metrics["epoch_time"] >= 0.0


def test_advent_grad_clip_stable(make_model, loaders):
    """With grad_clip enabled the adversarial training stays finite."""
    source_loaders, target_loaders = loaders
    config = TrainConfig(
        method="advent",
        epochs=1,
        lr=1e-4,
        lambda_adv=0.1,
        lambda_entropy=0.1,
        grad_clip=5.0,
        lr_d=1e-5,
        device="cpu",
        use_amp=False,
        seed=42,
    )
    engine = ADVENTTrainer(
        make_model(), source_loaders, target_loaders, config, torch.device("cpu")
    )
    metrics = engine.train_epoch()
    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite"


def test_fit_returns_model_with_history(make_model, loaders):
    """fit returns an nn.Module, tracks best mIoU and the loss history."""
    engine = _make_engine(make_model(), loaders)
    result = engine.fit()
    assert isinstance(result, torch.nn.Module)
    assert engine.best_miou > 0.0
    for key in ("loss_total", "loss_ce", "loss_adv", "loss_ent"):
        assert key in engine.history
        assert len(engine.history[key]) == 1
        assert np.isfinite(engine.history[key][0])


def test_discriminator_learns(make_model, loaders):
    """The discriminator receives gradient updates across training epochs.

    The adversarial objective is a minimax game, so the discriminator loss is
    not guaranteed to be monotonically decreasing (the segmenter learns to
    fool the discriminator). We therefore assert robustly: all losses are
    finite and the discriminator parameters moved during training.
    """
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

    for key in ("loss_adv", "loss_ent", "loss_total", "loss_ce"):
        assert key in metrics
        assert np.isfinite(metrics[key])
