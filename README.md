![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

# DARES: Domain Adaptation via α-Rényi Entropy for Semantic Segmentation

Unsupervised Domain Adaptation (UDA) framework for **dense semantic segmentation** applied to tropical rainforest **deforestation mapping**.

DARES adapts a segmentation model trained on **Sentinel-2** imagery over the Brazilian Amazon (São Félix do Xingu, PRODES) to **Landsat-8** imagery over the Colombian Amazon (San José del Guaviare / Caquetá forest reserves), under simultaneous **geographic** and **cross-sensor** domain shift. The method extends the Conditional Rényi α-Entropy Domain Adaptation (CREDA) framework to dense prediction tasks via a novel **Spatially-Stratified, Confidence-Guided Sampling operator** (Φ_c) that makes Gram-matrix computation tractable at pixel level.

## Features

- **Seven UDA methods** under one shared training interface: Source-Only baseline, ADVENT, CBST, CLAN, DACS, FDA and **DARES** (the proposed α-Rényi alignment).
- **Pluggable architecture matrix:** backbones (ResNet50, ConvNeXt-Tiny, Swin-T) × segmentation heads (ResUNet decoder, DeepLabV3+ ASPP).
- **DARES core loss (DARESLoss):** hardened class-conditional order-2 Rényi alignment over Gaussian Gram matrices with median-heuristic kernel bandwidth, per-pixel Rényi-2 confidence weighting, anti-collapse spectral entropy floors (`H2 ≥ η`), margin-hinged inter-class target repulsion, a per-step GradNorm-lite trust region on the deepest encoder block, and a dense Rényi-EM regularization term (`L_em`) on confident target predictions.
- **CREDA dynamic alignment schedule:** the Rényi weight follows the per-step sigmoid ramp `λ_eff = λ_max·s(t)·min(1, ρ·ĝ_seg/ĝ_aux)` (CREDA + GradNorm-lite EMA), so the alignment ramps in smoothly and can never dominate the supervised gradient.
- **HDF5 data pipeline** (`float32` 4-band patches, LZF compression) with lazy, worker-safe dataset loading.
- **Segmentation metrics:** per-class and mean IoU, DICE (F1), Overall Accuracy, MCC, plus per-class precision/recall.
- **Config-driven experiments:** every method/architecture combination is a YAML file; single CLI entry points for train and evaluate.
- **Automatic Mixed Precision** (AMP), deterministic seeding, tqdm progress and best-checkpoint tracking on target-validation mIoU.

## Methods

| Method | Type | Objective |
| --- | --- | --- |
| `source_only` | Baseline | Supervised CE on source only (adaptation gap reference). |
| `advent` | Adversarial | Entropy-map domain discriminator + target entropy minimization. |
| `cbst` | Self-training | Class-balanced self-training: per-class top-ratio pseudo-label selection + masked CE on target. |
| `clan` | Adversarial (output space) | Category-level adversarial output-space adaptation: a multi-class discriminator takes a single masked class channel (a "slice") of the prediction and classifies source vs target; target slices are pushed back towards their pseudo-label categories (CAA-Net / CLAN). |
| `fda` | Spectral alignment | Fourier low-frequency amplitude swap + Charbonnier entropy minimization. |
| `dares` | Info-theoretic | `L = L_seg + λ_eff·(L_align + β·L_ac + γ·L_rep) + λ_em·L_em` — hardened DARES class-conditional Rényi alignment with anti-collapse entropy floors, margin-hinged target repulsion, a per-step gradient trust region `λ_eff = λ_max·s(t)·min(1, ρ·ĝ_seg/ĝ_aux)` and dense Rényi-EM regularization. |

## Installation

```bash
git clone https://github.com/liturriago/DARES.git
cd DARES
python -m venv .venv && source .venv/bin/activate   # or `conda create -n dares python=3.11`
pip install -e .[dev]                                # dev extras: pytest, black, isort, mypy
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0 and torchvision. On Kaggle, `pip install -e .` inside a GPU notebook is enough.

## Dataset

The pre-processed dataset is published as a public Kaggle dataset, organized as
one folder per target degradation variant:

```
/kaggle/input/datasets/lucasiturriago/dares-amazon-uda/
├── Source/              # source_train.h5 | source_val.h5 | source_test.h5
├── Target_Original/     # target_train.h5  | target_val.h5  | target_test.h5
├── Target_Low/          # target_*_lime_low.h5     (LIME severity pool {0.1, 0.2})
├── Target_Medium/       # target_*_lime_med.h5     (LIME severity pool {0.3, 0.4, 0.5})
└── Target_High/         # target_*_lime_high.h5    (LIME severity pool {0.6, 0.7})
```

Every container stores `(N, 4, 224, 224)` float32 images and `(N, 224, 224)`
uint8 masks. Water (`WorldCover 80`) and wetland (`WorldCover 90`) pixels are
encoded as `255` with `ignore_index=255` in the loss and excluded from all
metrics. The target LIME variants simulate atmospheric attenuation and
illumination non-homogeneity at increasing severity (controlled radiometric
covariate shift). See [`Docs/data.md`](Docs/data.md) for the full
data-engineering specification (band configuration, water masking,
class-balance filtering, sliding-window extraction).

## Configuration

Experiments are fully described by a YAML file, organized in three folders:

```
configs/
├── LIME_stress/            # ResNet-50 + ResUNet × 7 methods × {low, medium, high} LIME tiers
│   ├── low/
│   ├── medium/
│   └── high/
├── architectures/          # source_only + dares × 5 backbone–head combos (medium)
│   ├── resnet50_resunet/
│   ├── resnet50_deeplabv3p/
│   ├── convnext_tiny_deeplabv3p/
│   ├── swin_t_resunet/
│   └── swin_t_deeplabv3p/
└── ablation/               # DARES with one component disabled (medium, ResNet-50 + ResUNet)
    ├── dares_no_align.yaml
    ├── dares_no_anti_collapse.yaml
    ├── dares_no_repulsion.yaml
    └── dares_no_em.yaml
```

`configs/LIME_stress/medium/dares.yaml` is the **DARES reference (headline) setup**. All other DARES configs (LIME tiers and architectures) share its exact training block and change only their target tier or model architecture; the ablation configs reuse it with exactly one safeguard switched off (see the comments in each file).

Example (`configs/LIME_stress/medium/dares.yaml`):

```yaml
data:
  source_dir: "/kaggle/input/datasets/lucasiturriago/dares-amazon-uda/Source"
  target_dir: "/kaggle/input/datasets/lucasiturriago/dares-amazon-uda/Target_Medium"
  target_variant: "medium"       # original | low | medium | high (selects target container naming)
  batch_size: 8
  patch_size: 224
  num_workers: 4
  mean: [0.0, 0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0, 1.0]
  use_augmentation: true

model:
  backbone: "resnet50"
  head: "resunet"
  in_channels: 4          # B2, B3, B4, B8
  num_classes: 2          # non_forest / forest
  pretrained: true
  dropout_rate: 0.1

training:
  method: "dares"
  epochs: 25
  lr: 0.0001
  weight_decay: 0.00001
  gamma: 0.94             # per-epoch multiplicative LR decay (DARESScheduler)
  use_amp: true
  device: "cuda"
  seed: 42
  warmup_epochs: 2

  quota: 256              # M — pixels sampled per class per batch
  min_samples: 8          # tau — class must have >= this many pixels
  lambda_max: 10.0        # peak alignment weight (DARES reference value)
  lambda_align: 1.0       # alignment term weight (L_align); 0 disables alignment only
  warmup_steps: 1000      # steps with lambda_eff = 0 (source-only warm-up)
  ramp_steps: 4000        # sigmoid ramp length (steps)
  ramp_delta: 10.0        # sigmoid steepness

  beta: 1.0               # anti-collapse term weight (L_ac)
  repulsion_gamma: 0.5    # inter-class repulsion term weight (L_rep)
  eta_floor: 1.0          # absolute H2 (bits) floor for source classes
  entropy_gap: 0.25       # target must stay within gap of source H2 (bits)
  repulsion_margin: 0.2   # hinge margin m (bits) for target repulsion

  trust_region: true      # GradNorm-lite trust region (set false to ablate)
  grad_ratio: 0.8         # rho — max ||g_aux||/||g_seg|| trust-region cap
  ema_decay: 0.9          # EMA decay for gradient norms

  use_renyi_em: true      # dense Rényi-EM regularization on target predictions
  lambda_em: 0.05         # EM weight (decoupled from the alignment weight)
  em_pool: false          # spatial pooling of the confidence map
  em_pool_kernel: 3

experiment:
  name: "resnet50_resunet_medium_dares"
  version: 1
  output_dir: "outputs/LIME_stress/medium/dares/experiment_1"
  save_results: true
```

## Usage

### Train (any method)

```bash
python scripts/train.py --config configs/LIME_stress/medium/dares.yaml
python scripts/train.py --config configs/LIME_stress/medium/dares.yaml --device cpu
```

The script builds the loaders, model and engine, runs `fit()`, saves the best checkpoint
(`model_final.pth`), the per-epoch `history.json`, and reports target/source test metrics
(`test_metrics.json`).

### Evaluate

```bash
python scripts/evaluate.py \
    --config configs/LIME_stress/medium/dares.yaml \
    --model outputs/LIME_stress/medium/dares/experiment_1/model_final.pth
```

Saves `evaluation_metrics.json`, confusion-matrix heatmaps and a qualitative
input/ground-truth/prediction overlay.

### Validate the HDF5 containers

```bash
python scripts/check_data.py --config configs/LIME_stress/medium/dares.yaml
```

Reports keys, shapes, dtypes, compression, per-split forest ratio (over valid,
non-water pixels; validates the [15%, 85%] balance protocol) and batches per
epoch. For a LIME tier, pass its container set explicitly:

```bash
python scripts/check_data.py \
    --source_dir .../Source --target_dir .../Target_High --target_variant high
```

### Stress-test variants (EXP-06…08)

Training a LIME degradation tier is just a different `target_dir` +
`target_variant` in the config:

```yaml
data:
  source_dir: "/kaggle/input/datasets/lucasiturriago/dares-amazon-uda/Source"
  target_dir: "/kaggle/input/datasets/lucasiturriago/dares-amazon-uda/Target_High"
  target_variant: high
```

### Kaggle notebook

```python
!git clone https://github.com/liturriago/DARES.git /kaggle/working/DARES
%cd /kaggle/working/DARES
!pip install -e .
!python scripts/train.py --config configs/LIME_stress/medium/dares.yaml
```

Use a GPU accelerator (T4 is enough for `batch_size=8` at `224×224`).

## Project structure

```
src/dares/
├── config.py            # Pydantic configs (data / model / training / experiment)
├── data/                # HDF5Dataset, pair transforms, collate, DARESDataLoader
├── models/              # backbones (resnet50, convnext, swin) + heads (resunet, deeplabv3p) + SegmentationModel
├── losses/              # CE, domain (discriminator), advent, cbst, clan, fda, dares_loss (DARES)
├── engines/             # source_only, advent, cbst, clan, dacs, fda, dares trainers + registry
├── training/            # BaseTrainer (AMP, evaluation, checkpointing, fit loop)
└── utils/               # metrics (mIoU/DICE/OA/MCC), evaluation, visualizer, reproducibility
scripts/                 # train.py, evaluate.py, check_data.py
configs/                 # LIME_stress / architectures / ablation experiment folders
tests/                   # pytest suite (models, data, metrics, engines, scripts)
```

## Testing

```bash
pytest tests/
```

## Citation

If you use this framework, please cite the DARES paper:

```bibtex
@article{iturriago2026dares,
  title   = {DARES: Domain Adaptation via $\alpha$-R\'enyi Entropy for Semantic Segmentation},
  author  = {Iturriago Salas, Lucas Miguel and Collazos Huertas, Diego Fabi\'an and
             Alvarez Meza, Andres Marino and Lozada Das Dores, Angel Jose},
  year    = {2026}
}
```

The CREDA foundation is described in:

```bibtex
@article{perez2025creda,
  title   = {Conditional Domain Adaptation with $\alpha$-R\'enyi Entropy Regularization
             and Noise-Aware Label Weighting},
  author  = {P\'erez-Rosero, Diego Armando and \'Alvarez-Meza, Andr\'es Marino and
             Castellanos-Dominguez, German},
  journal = {Mathematics},
  volume  = {13},
  number  = {16},
  pages   = {2602},
  year    = {2025},
  doi     = {10.3390/math13162602}
}
```

The category-level output-space adversarial baseline (CLAN / CAA-Net) is described in:

```bibtex
@article{ruan2019clan,
  title   = {Category-Level Adversaries for Semantic Domain Adaptation},
  author  = {Ruan, Congcong and Wang, Wei and Hu, Haifeng and Chen, Dihu},
  journal = {IEEE Access},
  volume  = {7},
  pages   = {83198--83208},
  year    = {2019},
  doi     = {10.1109/ACCESS.2019.2921030}
}
```

## License

This project is licensed under the MIT License.
