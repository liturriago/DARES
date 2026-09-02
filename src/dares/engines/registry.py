"""Engine factory: builds a training engine from its method name."""

from typing import Any

from dares.config import TrainConfig

from dares.engines.advent import ADVENTTrainer
from dares.engines.cbst import CBSTTrainer
from dares.engines.clan import CLANTrainer
from dares.engines.dacs import DACSTrainer
from dares.engines.dares import DARESTrainer
from dares.engines.dares_v2 import DARESV2Trainer
from dares.engines.fda import FDATrainer
from dares.engines.source_only import SourceOnlyTrainer

ENGINES: dict[str, type] = {
    "source_only": SourceOnlyTrainer,
    "advent": ADVENTTrainer,
    "cbst": CBSTTrainer,
    "clan": CLANTrainer,
    "dacs": DACSTrainer,
    "fda": FDATrainer,
    "dares": DARESTrainer,
    "dares_v2": DARESV2Trainer,
}


def build_engine(
    name: str,
    model: Any,
    source_loaders: dict[str, Any],
    target_loaders: dict[str, Any],
    config: TrainConfig,
    device: Any,
) -> Any:
    """Builds a training engine by method name.

    Every engine subclasses ``dares.training.base_trainer.BaseTrainer`` and is
    constructed with the same signature ``(model, source_loaders,
    target_loaders, config, device)``.

    Args:
        name (str): Method name, one of ``"source_only"``, ``"advent"``,
            ``"cbst"``, ``"clan"``, ``"dacs"``, ``"fda"``, ``"dares"`` or
            ``"dares_v2"``.
        model (nn.Module): The segmentation model.
        source_loaders (dict[str, DataLoader]): Labeled source loaders.
        target_loaders (dict[str, DataLoader]): Target loaders (train is
            unlabeled).
        config (TrainConfig): Training configuration.
        device (torch.device): Computing device.

    Returns:
        Any: The instantiated training engine.

    Raises:
        ValueError: If ``name`` is not a registered method.
    """
    engine_cls = ENGINES.get(name)
    if engine_cls is None:
        raise ValueError(
            f"Unknown training method {name!r}; "
            f"expected one of {sorted(ENGINES)}"
        )
    return engine_cls(model, source_loaders, target_loaders, config, device)
