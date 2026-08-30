from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
import yaml


class DataConfig(BaseModel):
    """Configuration for data loading and transformation."""

    source_dir: Path
    target_dir: Path
    target_variant: Literal["original", "low", "medium", "high"] = "original"
    data_root: Optional[Path] = None
    batch_size: int = Field(default=8, gt=0)
    patch_size: int = Field(default=224, gt=0)
    num_workers: int = Field(default=4, ge=0)
    mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    std: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    use_augmentation: bool = True

    def resolved_dir(self, field: Literal["source_dir", "target_dir"]) -> Path:
        """Resolves a directory field against data_root when it is relative."""
        path = Path(getattr(self, field))
        text = str(path)
        if self.data_root is None or path.is_absolute() or text.startswith(("/", "\\")):
            return path
        return Path(self.data_root) / path


class ModelConfig(BaseModel):
    """Configuration for the segmentation model architecture."""

    backbone: Literal[
        "resnet50",
        "convnext_tiny",
        "convnext_base",
        "convnext_large",
        "swin_t",
        "swin_s",
        "swin_b",
    ] = "resnet50"
    head: Literal["resunet", "deeplabv3p"] = "resunet"
    in_channels: int = Field(default=4, gt=0)
    num_classes: int = Field(default=2, gt=0)
    pretrained: bool = True
    dropout_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    resunet_channels: list[int] = Field(default_factory=lambda: [512, 256, 128, 64, 32])
    deeplab_aspp_channels: int = Field(default=256, gt=0)
    deeplab_low_level_channels: int = Field(default=48, gt=0)


class TrainConfig(BaseModel):
    """Configuration for training and Domain Adaptation."""

    epochs: int = Field(default=25, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    gamma: float = Field(default=0.94, gt=0.0, le=1.0)

    # Learning rate schedules
    lr_schedule: bool = False
    schedule_alpha: float = Field(default=20.0, gt=0.0)
    schedule_beta: float = Field(default=0.75, gt=0.0)

    use_amp: bool = True
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    seed: int = 42

    # Warm-up
    warmup_epochs: Optional[int] = None
    warmup_steps: int = Field(default=1000, ge=0)
    ramp_steps: int = Field(default=4000, ge=1)
    ramp_delta: float = Field(default=10.0, gt=0.0)

    # DARES Sampling
    quota: int = Field(default=256, gt=0)
    min_samples: int = Field(default=8, gt=0)
    normalize_features: bool = False
    weight_cross: bool = False
    sigma_min: float = Field(default=1e-3, gt=0.0)
    sigma_max: float = Field(default=1e3, gt=0.0)

    # DARES Loss Safeguards
    lambda_max: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.0, ge=0.0)
    repulsion_gamma: float = Field(default=0.5, ge=0.0)
    eta_floor: float = Field(default=1.0, ge=0.0)
    entropy_gap: float = Field(default=0.25, ge=0.0)
    repulsion_margin: float = Field(default=0.2, ge=0.0)

    # Gradient Control & Alignment Form (CORREGIDOS A VALORES ESTABLES)
    trust_region: bool = True
    grad_ratio: float = Field(default=0.8, gt=0.0)
    ema_decay: float = Field(default=0.9, gt=0.0, le=1.0)
    align_form: Literal["ce", "mi"] = "mi"

    # Rényi-EM dense regularization
    use_renyi_em: bool = True
    lambda_em: float = Field(default=0.05, ge=0.0)
    em_pool: bool = False
    em_pool_kernel: int = Field(default=3, gt=0)

    # Method selection
    method: Literal["source_only", "advent", "dacs", "fda", "dares"] = "dares"

    # ADVENT
    lambda_adv: float = Field(default=0.1, ge=0.0)
    lambda_entropy: float = Field(default=0.1, ge=0.0)

    # DACS
    dacs_threshold: float = Field(default=0.968, gt=0.0, le=1.0)
    dacs_mix_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    dacs_color_jitter: bool = True
    dacs_brightness: float = Field(default=0.5, ge=0.0)
    dacs_contrast: float = Field(default=0.5, ge=0.0)
    dacs_saturation: float = Field(default=0.5, ge=0.0)
    dacs_hue: float = Field(default=0.25, ge=0.0)
    dacs_blur: bool = True
    dacs_blur_kernel: int = Field(default=5, gt=0)
    dacs_blur_sigma: tuple[float, float] = (0.1, 2.0)

    # FDA
    fda_beta: float = Field(default=0.09, gt=0.0, le=0.5)
    fda_lambda_entropy: float = Field(default=0.005, ge=0.0)
    fda_eta: float = Field(default=2.0, gt=0.0)

    # Gradient clipping & secondary optimizer
    lr_d: Optional[float] = Field(default=None, gt=0)
    grad_clip: Optional[float] = Field(default=None, ge=0.0)


class ExperimentMetadata(BaseModel):
    """Metadata for experiment tracking and outputs."""

    name: str = "dares_experiment"
    version: int = Field(default=1, ge=0)
    output_dir: Path = Path("outputs/experiment_1")
    save_results: bool = False


class ExperimentConfig(BaseModel):
    """Global schema that unites all configurations."""

    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    experiment: ExperimentMetadata

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Loads and validates the configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if "experiment" in config_dict and "output_dir" in config_dict["experiment"]:
            out_dir = str(config_dict["experiment"]["output_dir"]).replace("\\", "/")
            if out_dir.startswith("/outputs"):
                config_dict["experiment"]["output_dir"] = out_dir.lstrip("/")

        return cls(**config_dict)