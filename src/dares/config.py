from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class DataConfig(BaseModel):
    """Configuration for data loading and transformation.

    Attributes:
        source_dir (Path): Directory containing the source domain HDF5 containers
            (source_train.h5, source_val.h5, source_test.h5). May be an absolute
            path (e.g. a Kaggle mount ``/kaggle/input/<dataset>/Source``) or a
            path relative to ``data_root``.
        target_dir (Path): Directory containing the target domain HDF5 containers.
            Points at a single variant folder (e.g. ``Target_Original``,
            ``Target_Low``, ``Target_Medium`` or ``Target_High``); the container
            filenames inside are selected by ``target_variant``.
        target_variant (Literal): Target degradation tier. ``"original"`` reads
            ``target_{split}.h5``; the LIME tiers read
            ``target_{split}_lime_{low,med,high}.h5`` respectively.
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
    target_variant: Literal["original", "low", "medium", "high"] = "original"
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
        lr_schedule (bool): Enable the dynamic DARES learning rate schedule
            (eta(p) = eta0 * (1 + alpha*p)^(-beta)).
        schedule_alpha (float): alpha hyperparameter of the LR schedule.
        schedule_beta (float): beta hyperparameter of the LR schedule.
        use_amp (bool): Whether to use Automatic Mixed Precision.
        device (Literal): Computing device (cpu, cuda, mps).
        seed (int): Random seed for reproducibility.
        warmup_epochs (int | None): Epochs with alignment disabled.
            Kept for backward compatibility with the epoch-based schedule; the
            DARESLoss engine ignores it in favor of its per-step ``warmup_steps``.
        quota (int): DARESLoss per-class pixel sampling quota (M).
        min_samples (int): DARESLoss minimum pixels per class (tau) to include a class.
        lambda_max (float): DARESLoss peak alignment weight.
        beta (float): DARESLoss weight of the anti-collapse term.
        repulsion_gamma (float): DARESLoss weight of the inter-class repulsion term.
        eta_floor (float): DARESLoss absolute H2 (bits) floor for source classes.
        entropy_gap (float): DARESLoss target-vs-source entropy gap (bits).
        repulsion_margin (float): DARESLoss margin m (bits) of the repulsion hinge.
        warmup_steps (int): DARESLoss steps with lambda_eff = 0 (source-only warm-up).
        ramp_steps (int): DARESLoss sigmoid ramp length (steps) after warm-up.
        ramp_delta (float): DARESLoss sigmoid steepness.
        grad_ratio (float): DARESLoss grad-norm cap rho (max ||g_aux||/||g_seg||).
        trust_region (bool): DARESLoss enable the GradNorm-lite trust region
            (gradient-ratio cap). When ``False`` the alignment weight follows
            only the sigmoid ramp (``lambda_eff = lambda_max * s(t)``).
        ema_decay (float): DARESLoss EMA decay for gradient norms.
        method (Literal): UDA method driving ``build_engine`` (``"source_only"``,
            ``"advent"``, ``"dacs"``, ``"fda"`` or ``"dares"``).
        lambda_adv (float): ADVENT adversarial alignment weight.
        lambda_entropy (float): ADVENT entropy minimization weight.
        dacs_threshold (float): DACS pseudo-label confidence threshold.
        dacs_mix_ratio (float): DACS fraction of source classes pasted in ClassMix.
        dacs_color_jitter (bool): DACS enable color jitter (source and target).
        dacs_brightness / dacs_contrast / dacs_saturation / dacs_hue (float):
            DACS color-jitter magnitudes.
        dacs_blur (bool): DACS enable Gaussian blur on the target batch.
        dacs_blur_kernel (int): DACS Gaussian blur kernel size.
        dacs_blur_sigma (tuple[float, float]): DACS Gaussian blur sigma range.
        fda_beta (float): FDA spectral-swap fraction (low-frequency amplitude).
        fda_lambda_entropy (float): FDA entropy-regularization weight.
        fda_eta (float): FDA Charbonnier entropy exponent.
    """
    epochs: int = Field(default=45, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    gamma: float = Field(default=0.94, gt=0.0, le=1.0)

    lr_schedule: bool = True
    schedule_alpha: float = Field(default=20.0, gt=0.0)
    schedule_beta: float = Field(default=0.75, gt=0.0)

    use_amp: bool = True
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    seed: int = 42

    # Epoch-based warm-up (backward compatible; DARESLoss uses warmup_steps)
    warmup_epochs: int | None = None

    # DARES alignment hyperparameters (DARESLoss)
    quota: int = Field(default=256, gt=0)
    min_samples: int = Field(default=8, gt=0)
    lambda_max: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.0, ge=0.0)
    repulsion_gamma: float = Field(default=0.5, ge=0.0)
    eta_floor: float = Field(default=1.0, ge=0.0)
    entropy_gap: float = Field(default=0.25, ge=0.0)
    repulsion_margin: float = Field(default=0.2, ge=0.0)
    warmup_steps: int = Field(default=1000, ge=0)
    ramp_steps: int = Field(default=4000, ge=1)
    ramp_delta: float = Field(default=10.0, gt=0.0)
    grad_ratio: float = Field(default=0.8, gt=0.0)
    trust_region: bool = True
    ema_decay: float = Field(default=0.9, gt=0.0, le=1.0)

    # Method selection (drives build_engine)
    method: Literal["source_only", "advent", "dacs", "fda", "dares"] = "dares"

    # ADVENT / adversarial alignment
    lambda_adv: float = Field(default=0.1, ge=0.0)
    lambda_entropy: float = Field(default=0.1, ge=0.0)

    # DACS (cross-domain mixed sampling)
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

    # FDA (Fourier Domain Adaptation)
    fda_beta: float = Field(default=0.09, gt=0.0, le=0.5)
    fda_lambda_entropy: float = Field(default=0.005, ge=0.0)
    fda_eta: float = Field(default=2.0, gt=0.0)

    # Optimizer learning-rate override (discriminator methods e.g. ADVENT)
    lr_d: float | None = Field(default=None, gt=0)

    # Gradient clipping (global max norm) applied to each optimizer step.
    # Mainly stabilizes adversarial methods (ADVENT); None disables.
    grad_clip: float | None = Field(default=None, ge=0.0)


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
