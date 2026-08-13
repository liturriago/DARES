from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class DataConfig(BaseModel):
    """Configuration for data loading and transformation.

    Attributes:
        source_dir (Path): Directory containing the source domain HDF5 containers
            (source_train.h5, source_val.h5, source_test.h5). May be an absolute
            path (e.g. a Kaggle mount ``/kaggle/input/<dataset>/source``) or a
            path relative to ``data_root``.
        target_dir (Path): Directory containing the target domain HDF5 containers
            (target_train.h5, target_val.h5, target_test.h5).
        data_root (Path | None): Optional root used to resolve relative
            ``source_dir`` / ``target_dir`` (portable between local and Kaggle).
        batch_size (int): Number of patches per batch.
        patch_size (int): Spatial size of the (square) patches loaded from disk.
        num_workers (int): Number of subprocesses used for data loading.
        mean (tuple[float, ...]): Per-channel normalization mean (length == in_channels).
        std (tuple[float, ...]): Per-channel normalization std (length == in_channels).
        use_augmentation (bool): Whether to apply spatial augmentations to the
            source training set.
    """
    source_dir: Path
    target_dir: Path
    data_root: Path | None = None
    batch_size: int = Field(default=8, gt=0)
    patch_size: int = Field(default=224, gt=0)
    num_workers: int = Field(default=4, ge=0)
    mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    std: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    use_augmentation: bool = True

    def resolved_dir(self, field: Literal["source_dir", "target_dir"]) -> Path:
        """Resolves a directory field against ``data_root`` when it is relative.

        Absolute paths (e.g. ``/kaggle/input/<dataset>/source``) are returned
        unchanged; relative paths are joined under ``data_root`` (falling back
        to the path itself when ``data_root`` is ``None``).

        Args:
            field (Literal): The ``DataConfig`` field to resolve, ``"source_dir"``
                or ``"target_dir"``.

        Returns:
            Path: The resolved directory path.
        """
        path = Path(getattr(self, field))
        text = str(path)
        if self.data_root is None or path.is_absolute() or text.startswith(("/", "\\")):
            return path
        return Path(self.data_root) / path


class ModelConfig(BaseModel):
    """Configuration for the segmentation model architecture.

    Attributes:
        backbone (Literal): Name of the encoder / feature extractor.
        head (Literal): Name of the segmentation decoder / head.
        in_channels (int): Number of input channels (4 = B2, B3, B4, B8).
        num_classes (int): Number of output classes (2 = Forest / Non-Forest).
        pretrained (bool): Whether to use ImageNet-pretrained backbone weights.
        dropout_rate (float): Dropout probability in the classification head.
        resunet_channels (list[int]): Channel plan for the ResUNet decoder,
            ordered from the deepest level (S32) to the shallowest (S2).
        deeplab_aspp_channels (int): Internal channel width of the DeepLabV3+ ASPP.
        deeplab_low_level_channels (int): Channel width of the low-level feature
            projection in DeepLabV3+.
        feature_dim (int): Dimension of the dense feature map returned in
            'feature' mode (used for Gram matrix computation).
    """
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
    resunet_channels: list[int] = Field(
        default_factory=lambda: [512, 256, 128, 64, 32]
    )
    deeplab_aspp_channels: int = Field(default=256, gt=0)
    deeplab_low_level_channels: int = Field(default=48, gt=0)


class TrainConfig(BaseModel):
    """Configuration for training and Domain Adaptation.

    Attributes:
        epochs (int): Total number of training epochs.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        gamma (float): Exponential learning rate decay factor.
        lr_schedule (bool): Enable the dynamic CREDA learning rate schedule
            (eta(p) = eta0 * (1 + alpha*p)^(-beta)).
        schedule_alpha (float): alpha hyperparameter of the LR schedule.
        schedule_beta (float): beta hyperparameter of the LR schedule.
        schedule_delta (float): delta hyperparameter of the alignment weight ramp.
        use_amp (bool): Whether to use Automatic Mixed Precision.
        device (Literal): Computing device (cpu, cuda, mps).
        seed (int): Random seed for reproducibility.
        lambda_renyi (float): Weight of the DARES alpha-Renyi alignment term.
        tau (float): Pseudo-label confidence threshold for the sampling operator.
        n_max (int): Maximum number of samples per class per mini-batch (N_max).
        sigma (float | Literal["auto"]): Kernel bandwidth; "auto" uses the
            median heuristic.
        alpha (int): Order of the Renyi entropy (alpha = 2).
        warmup_epochs (int | None): Epochs with alignment disabled (lambda_renyi = 0).
    """
    epochs: int = Field(default=45, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    gamma: float = Field(default=0.94, gt=0.0, le=1.0)

    lr_schedule: bool = True
    schedule_alpha: float = Field(default=20.0, gt=0.0)
    schedule_beta: float = Field(default=0.75, gt=0.0)
    schedule_delta: float = Field(default=20.0, gt=0.0)

    use_amp: bool = True
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    seed: int = 42

    # DARES alignment hyperparameters
    lambda_renyi: float = Field(default=0.1, ge=0.0)
    tau: float = Field(default=0.85, gt=0.0, le=1.0)
    n_max: int = Field(default=1024, gt=0)
    sigma: float | Literal["auto"] = "auto"
    alpha: int = Field(default=2, ge=1)
    warmup_epochs: int | None = None

    # Method selection (drives build_engine)
    method: Literal["source_only", "advent", "cycada", "cbst", "dares"] = "dares"

    # ADVENT / adversarial alignment
    lambda_adv: float = Field(default=0.1, ge=0.0)
    lambda_entropy: float = Field(default=0.1, ge=0.0)

    # CyCADA
    lambda_pixel: float = Field(default=1.0, ge=0.0)
    lambda_feat: float = Field(default=1.0, ge=0.0)
    lambda_cycle: float = Field(default=1.0, ge=0.0)
    lambda_identity: float = Field(default=0.1, ge=0.0)

    # CBST
    lambda_self: float = Field(default=1.0, ge=0.0)
    pseudo_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    pseudo_topk_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    n_self_training_rounds: int = Field(default=1, gt=0)

    # Optimizer learning-rate overrides
    lr_d: float | None = Field(default=None, gt=0)
    lr_g: float | None = Field(default=None, gt=0)


class ExperimentMetadata(BaseModel):
    """Metadata for experiment tracking and outputs.

    Attributes:
        name (str): Unique name for the experiment.
        version (int): Iteration number of the experiment.
        output_dir (Path): Directory to save logs, models, and visualizations.
        save_results (bool): Whether to save the results of the experiment.
    """
    name: str = "dares_experiment"
    version: int = Field(default=1, ge=0)
    output_dir: Path = Path("outputs/experiment_1")
    save_results: bool = False


class ExperimentConfig(BaseModel):
    """Global schema that unites all configurations.

    Attributes:
        data (DataConfig): Data-related settings.
        model (ModelConfig): Architecture-related settings.
        training (TrainConfig): Training-related settings.
        experiment (ExperimentMetadata): Metadata for tracking.
    """
    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    experiment: ExperimentMetadata

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Loads and validates the configuration from a YAML file.

        Args:
            yaml_path (str | Path): Path to the YAML configuration file.

        Returns:
            ExperimentConfig: Validated configuration object.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Clean output_dir to prevent absolute paths starting with /outputs
        if "experiment" in config_dict and "output_dir" in config_dict["experiment"]:
            out_dir = str(config_dict["experiment"]["output_dir"]).replace("\\", "/")
            if out_dir.startswith("/outputs"):
                config_dict["experiment"]["output_dir"] = out_dir.lstrip("/")

        return cls(**config_dict)
