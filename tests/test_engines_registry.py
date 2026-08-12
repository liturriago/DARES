"""Registry + cross-engine contract tests for the DARES training engines.

Validates that every engine exposes the same design pattern: identical
constructor signature, subclass of ``BaseTrainer``, ``train_epoch`` returning
a metrics dict with ``epoch_time``, and ``fit`` returning the best model.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dares.config import DataConfig, ModelConfig, TrainConfig
from dares.data import DARESDataLoader
from dares.engines import build_engine
from dares.engines.registry import ENGINES
from dares.models import build_model
from dares.training.base_trainer import BaseTrainer

METHODS = sorted(ENGINES)


def make_h5(file_path: Path, num_patches: int = 6, height: int = 64, width: int = 64) -> None:
    """Writes a tiny synthetic LZF HDF5 container."""
    file_path = Path(file_path)
    rng = np.random.default_rng(0)
    masks = rng.integers(0, 2, size=(num_patches, height, width)).astype(np.uint8)
    images = rng.random((num_patches, 4, height, width)).astype(np.float32)
    images[:, 0] = masks
    with h5py.File(file_path, "w") as f:
        f.create_dataset("images", data=images, compression="lzf", chunks=(1, 4, height, width))
        f.create_dataset("masks", data=masks, compression="lzf", chunks=(1, height, width))


@pytest.fixture(scope="module")
def loaders(tmp_path_factory):
    """Builds the six HDF5 containers and the paired source / target loaders."""
    tmp_path = tmp_path_factory.mktemp("engines_data")
    for domain in ("source", "target"):
        for split, num in (("train", 6), ("val", 3), ("test", 3)):
            make_h5(tmp_path / f"{domain}_{split}.h5", num_patches=num)
    config = DataConfig(source_dir=tmp_path, target_dir=tmp_path, batch_size=2, num_workers=0)
    loader = DARESDataLoader(config)
    return loader.get_source_loaders(), loader.get_target_loaders()


def _make_model():
    return build_model(
        ModelConfig(backbone="resnet50", head="resunet", in_channels=4, num_classes=2, pretrained=False)
    )


def test_registry_unknown_method():
    """Unknown method names raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown training method"):
        build_engine(
            "nope",
            None,
            {},
            {},
            TrainConfig(method="source_only", epochs=1, device="cpu", use_amp=False),
            torch.device("cpu"),
        )


def test_registry_mapping():
    """Every registered method maps to a BaseTrainer subclass."""
    for name in METHODS:
        assert name in {"source_only", "advent", "cycada", "cbst", "dares"}
        engine_cls = ENGINES[name]
        assert issubclass(engine_cls, BaseTrainer)


@pytest.mark.parametrize("method", METHODS)
def test_engine_common_contract(method, loaders):
    """All engines share the same interface: signature, train_epoch, fit."""
    source_loaders, target_loaders = loaders
    model = _make_model()
    config = TrainConfig(method=method, epochs=1, lr=1e-4, device="cpu", use_amp=False, seed=42)
    device = torch.device("cpu")

    engine = build_engine(method, model, source_loaders, target_loaders, config, device)
    assert engine.__class__ is ENGINES[method]
    assert isinstance(engine, BaseTrainer)

    metrics = engine.train_epoch()
    assert "epoch_time" in metrics
    for key, value in metrics.items():
        if key == "epoch_time":
            assert value >= 0.0
        else:
            assert np.isfinite(value)

    result = engine.fit()
    assert isinstance(result, torch.nn.Module)
    assert engine.best_miou > 0.0
