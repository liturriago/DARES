![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

# DARES: Domain Adaptation via α-Rényi Entropy for Semantic Segmentation

Unsupervised Domain Adaptation (UDA) framework for **dense semantic segmentation** applied to tropical rainforest **deforestation mapping**.

DARES adapts a segmentation model trained on **Sentinel-2** imagery over the Brazilian Amazon (São Félix do Xingu, PRODES) to **Landsat-8** imagery over the Colombian Amazon (San José del Guaviare / Caquetá forest reserves), under simultaneous **geographic** and **cross-sensor** domain shift. The method extends the Conditional Rényi α-Entropy Domain Adaptation (CREDA) framework [1] to dense prediction tasks via a novel **Spatially-Stratified, Confidence-Guided Sampling operator** (Φ_c) that makes Gram-matrix computation tractable at pixel level.

## Features

- **Five UDA methods** under one shared training interface: Source-Only baseline, ADVENT, CyCADA, CBST and **DARES** (the proposed α-Rényi alignment).
- **Pluggable architecture matrix:** backbones (ResNet50, ConvNeXt-Tiny, Swin-T) × segmentation heads (ResUNet decoder, DeepLabV3+ ASPP).
- **DARES core loss:** matrix-based order-2 Rényi mutual information `Ĩ₂` over class-conditional Gaussian Gram matrices, with median-heuristic kernel bandwidth and entropy-based pseudo-label confidence weighting.
- **CREDA dynamic alignment schedule:** the Rényi weight follows `λ(p) = λ_max · tanh(δp/2)` (CREDA Eq. 29) with `p` the relative training progress, so the alignment ramps in smoothly instead of jumping in at full strength.
- **HDF5 data pipeline** (`float32` 4-band patches, LZF compression) with lazy, worker-safe dataset loading.
- **Segmentation metrics:** per-class and mean IoU, DICE (F1), Overall Accuracy, MCC, plus per-class precision/recall.
- **Config-driven experiments:** every method/architecture combination is a YAML file; single CLI entry points for train and evaluate.
- **Automatic Mixed Precision** (AMP), deterministic seeding, tqdm progress and best-checkpoint tracking on target-validation mIoU.

## Methods

| Method | Type | Objective |
| --- | --- | --- |
| `source_only` | Baseline | Supervised CE on source only (adaptation gap reference). |
| `advent` | Adversarial | Entropy-map domain discriminator + target entropy minimization. |
| `cycada` | Adversarial | Cycle-consistent pixel translation + feature adversarial alignment. |
| `cbst` | Self-training | Class-balanced pseudo-label selection with masked CE. |
| `dares` | Info-theoretic | `L = L_CE(D_s) − λ(p) Σ_c Ĩ₂(K_s^c; K̃_t^c)` — class-conditional Rényi alignment with the Φ_c sampling operator and a CREDA dynamic weight ramp `λ(p) = λ_max·tanh(δp/2)`. |

## Installation

```bash
git clone https://github.com/liturriago/DARES.git
cd DARES
python -m venv .venv && source .venv/bin/activate   # or `conda create -n dares python=3.11`
pip install -e .[dev]                                # dev extras: pytest, black, isort, mypy
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0 and torchvision. On Kaggle, `pip install -e .` inside a GPU notebook is enough.

## Dataset

The pre-processed dataset is published as a public Kaggle dataset (all six containers in one folder):

```
/kaggle/input/datasets/lucasiturriago/dares-amazon-deforestation-uda/
```

| Container | Domain | Split | Patches | Image | Mask |
| --- | --- | --- | --- | --- | --- |
| `source_train.h5` | Source (Brazil) | Train | 3,944 | `(N, 4, 224, 224)` f32 | `(N, 224, 224)` u8 |
| `source_val.h5` | Source (Brazil) | Validation | 1,108 | … | … |
| `source_test.h5` | Source (Brazil) | Test | 633 | … | … |
| `target_train.h5` | Target (Colombia) | Train | 3,711 | … | … |
| `target_val.h5` | Target (Colombia) | Validation | 894 | … | … |
| `target_test.h5` | Target (Colombia) | Test | 589 | … | … |

See [`Docs/data.md`](Docs/data.md) for the full data-engineering specification (band configuration, water masking, class-balance filtering, sliding-window extraction).

## Configuration

Experiments are fully described by a YAML file. The matrix is organized as one folder per backbone–head pair:

```
configs/training/
├── resnet50_resunet/        # dares.yaml, source_only.yaml, advent.yaml, cycada.yaml, cbst.yaml
├── resnet50_deeplabv3p/
├── convnext_tiny_resunet/
├── convnext_tiny_deeplabv3p/
├── swin_t_resunet/
└── swin_t_deeplabv3p/
```

Example (`configs/training/resnet50_resunet/dares.yaml`):

```yaml
data:
  source_dir: "/kaggle/input/datasets/lucasiturriago/dares-amazon-deforestation-uda"
  target_dir: "/kaggle/input/datasets/lucasiturriago/dares-amazon-deforestation-uda"
  batch_size: 8
  patch_size: 224
  num_workers: 4

model:
  backbone: "resnet50"
  head: "resunet"
  in_channels: 4          # B2, B3, B4, B8
  num_classes: 2          # non_forest / forest
  pretrained: true

training:
  method: "dares"
  epochs: 45
  lr: 0.0001
  device: "cuda"
  lambda_renyi: 0.2       # λ_max of the alignment weight ramp
  tau: 0.8                # Φ_c confidence threshold
  n_max: 1024             # samples per class per mini-batch
  sigma: "auto"           # median-heuristic kernel bandwidth
  grid_size: 16           # Φ_c spatial grid (per side)
  schedule_delta: 8       # CREDA ramp steepness (λ(p) = λ_max·tanh(δp/2)); 0 = constant λ

experiment:
  name: "resnet50_resunet_dares"
  output_dir: "outputs/resnet50_resunet/dares/experiment_1"
```

## Usage

### Train (any method)

```bash
python scripts/train.py --config configs/training/resnet50_resunet/dares.yaml
python scripts/train.py --config configs/training/resnet50_resunet/dares.yaml --device cpu
```

The script builds the loaders, model and engine, runs `fit()`, saves the best checkpoint
(`model_final.pth`), the per-epoch `history.json`, and reports target/source test metrics
(`test_metrics.json`).

### Evaluate

```bash
python scripts/evaluate.py \
    --config configs/training/resnet50_resunet/dares.yaml \
    --model outputs/resnet50_resunet/dares/experiment_1/model_final.pth
```

Saves `evaluation_metrics.json`, confusion-matrix heatmaps and a qualitative
input/ground-truth/prediction overlay.

### Validate the HDF5 containers

```bash
python scripts/check_data.py --config configs/training/resnet50_resunet/dares.yaml
```

Reports keys, shapes, dtypes, compression, per-split forest ratio (validates the [15%, 85%]
balance protocol) and batches per epoch.

### Kaggle notebook

```python
!git clone https://github.com/liturriago/DARES.git /kaggle/working/DARES
%cd /kaggle/working/DARES
!pip install -e .
!python scripts/train.py --config configs/training/resnet50_resunet/dares.yaml
```

Use a GPU accelerator (T4 is enough for `batch_size=8` at `224×224`).

## Project structure

```
src/dares/
├── config.py            # Pydantic configs (data / model / training / experiment)
├── data/                # HDF5Dataset, pair transforms, collate, DARESDataLoader
├── models/              # backbones (resnet50, convnext, swin) + heads (resunet, deeplabv3p) + SegmentationModel
├── losses/              # CE, domain (discriminator), advent, cycada, cbst, renyi (DARES)
├── engines/             # source_only, advent, cycada, cbst, dares trainers + registry
├── training/            # BaseTrainer (AMP, evaluation, checkpointing, fit loop)
└── utils/               # metrics (mIoU/DICE/OA/MCC), evaluation, visualizer, reproducibility
scripts/                 # train.py, evaluate.py, check_data.py
configs/training/        # one folder per backbone-head pair
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

## License

This project is licensed under the MIT License.
