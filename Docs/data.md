# TECHNICAL SPECIFICATION REPORT: DATASET CONSTRUCTION AND PREPROCESSING PROTOCOL FOR DARES

**Project Title:** Domain Adaptation via $\alpha$-Rényi Entropy for Semantic Segmentation (DARES)

**Target Application:** Cross-Sensor and Cross-Region Deforestation Mapping (Brazil Amazon $\rightarrow$ Colombian Forest Reserves)

**Document Version:** 3.1 (Comprehensive Data Engineering & Model Architecture Specification)

---

## 1. Executive Overview & Experimental Setup

This technical report documents the complete data engineering protocol designed to evaluate the **DARES** framework. The benchmark addresses an Unsupervised Domain Adaptation (UDA) challenge for dense semantic segmentation under simultaneous **geographic domain shift** and **cross-sensor covariate shift**:

* **Source Domain ($\mathcal{D}_s$):** Sentinel-2 MSI imagery over the active deforestation frontier of the Brazilian Amazon (São Félix do Xingu, Pará). Fully labeled with ground truth annotations ($Y_s$).
* **Target Domain ($\mathcal{D}_t$):** Landsat-8 OLI imagery over the Colombian Amazonian deforestation arc (San José del Guaviare / Caquetá). Unlabeled during adaptation training ($X_t$), with spatial block subsets reserved for validation ($Y_{t,\text{val}}$) and benchmark evaluation ($Y_{t,\text{test}}$).

---

## 2. Satellite Data Specifications & Acquisition Strategy

Both source and target acquisitions are temporally aligned to the calendar year **2021** to guarantee thematic synchronization with the ground truth reference dataset and eliminate seasonal/inter-annual vegetation discrepancies.

### 2.1 Source Domain ($\mathcal{D}_s$): Sentinel-2 MSI

* **Platform/Sensor:** Copernicus Sentinel-2 MultiSpectral Instrument (Harmonized Level-2A, Surface Reflectance).
* **Native Spatial Resolution:** 10 meters per pixel (Ground Sample Distance - GSD).
* **Spectral Configuration:** 4-channel multi-spectral tensor containing native 10m bands:
* Channel 0: `B2` (Blue, $\lambda \approx 490\text{ nm}$)
* Channel 1: `B3` (Green, $\lambda \approx 560\text{ nm}$)
* Channel 2: `B4` (Red, $\lambda \approx 665\text{ nm}$)
* Channel 3: `B8` (Near-Infrared / NIR, $\lambda \approx 842\text{ nm}$)


* **Temporal Aggregation:** Pixel-wise median composite across all acquisitions in 2021.
* **Spectral Scale Normalization:** Raw integer surface reflectance values ($> 10.0$) are normalized to the physical float interval $[0.0, 1.0]$ via $1 / 10000.0$ division, yielding a clean range of $[0.0071, 1.0000]$.

### 2.2 Target Domain ($\mathcal{D}_t$): Landsat-8 OLI

* **Platform/Sensor:** USGS Landsat-8 Operational Land Imager (Collection 2 Level-2, Surface Reflectance).
* **Native Spatial Resolution:** 30 meters per pixel (GSD).
* **Pixel-Level Cloud Masking Protocol:** To prevent empty image collections in cloud-dense tropical Amazonian regions, scene-level filtering (`CLOUD_COVER < 20`) is eschewed in favor of per-pixel bitwise quality assessment via `QA_PIXEL` (Bit 3 = Clouds, Bit 4 = Cloud Shadows):

$$\text{Mask}_{\text{Valid}} = (\text{QA} \ \& \ 2^3 == 0) \ \land \ (\text{QA} \ \& \ 2^4 == 0)$$


* **Radiometric Calibration:** Official USGS Collection 2 Level-2 surface reflectance scaling formula applied directly in GEE:

$$\text{Reflectance} = \text{DN} \times 0.0000275 - 0.2$$



Yielding an unclipped physical surface reflectance range of $[-0.0893, 0.6386]$ (where slight negative values represent atmospheric correction residual noise over dense shadows).
* **Spectral Harmonization:** Selected equivalent spectral channels and renamed to standardize the input shape:
* Channel 0: `SR_B2` $\rightarrow$ Renamed `B2` (Blue, $\lambda \approx 482\text{ nm}$)
* Channel 1: `SR_B3` $\rightarrow$ Renamed `B3` (Green, $\lambda \approx 561\text{ nm}$)
* Channel 2: `SR_B4` $\rightarrow$ Renamed `B4` (Red, $\lambda \approx 655\text{ nm}$)
* Channel 3: `SR_B5` $\rightarrow$ Renamed `B8` (NIR, $\lambda \approx 865\text{ nm}$)


* **Export Strategy:** Exported at native **30m spatial resolution** (`scale: 30`) to avoid GEE server-side projection evaluation artifacts. Resampling to 10m is executed locally during Python ingestion via `Resampling.cubic`.

---

## 3. Ground Truth Annotation, Taxonomical Binarization & Water Masking

Ground truth annotations for both domains are derived from the **ESA WorldCover 2021 (v200)** global land cover dataset, natively generated at 10m spatial resolution.

### 3.1 Binary Class Taxonomy Mapping

The multi-class land cover taxonomy is collapsed into a binary semantic segmentation task focused on deforestation monitoring:

* **Class 1 (Forest / Target Class):** ESA WorldCover Value `10` (`Tree cover`).
* **Class 0 (Non-Forest / Deforested / Anthropogenic):** ESA WorldCover Values `20` (`Shrubland`), `30` (`Grassland`), `40` (`Cropland`), `50` (`Built-up`), and `60` (`Bare / sparse vegetation`).

### 3.2 Strict Water Masking Protocol

* **Excluded Class:** ESA WorldCover Value `80` (`Permanent water bodies`).
* **Methodology:** Permanent water bodies are explicitly converted into `NaN` / `NoData` mask values in both ground truth and imagery rasters.
* **Theoretical Justification:** Water exhibits near-zero NIR reflectance, creating an extreme bimodal distribution within Class 0 if grouped with highly reflective bare soil or pastures. Excluding water prevents noisy bimodal feature distributions within the non-forest class, ensuring that the Reproducing Kernel Hilbert Space (RKHS) Gram matrices computed by DARES reflect true land-use change rather than hydrological signals.

---

## 4. Spatial Footprint & Resolution Standardization Rationale

Rather than degrading Sentinel-2 to 30m, Landsat-8 is resampled to 10m for four key reasons:

1. **Ground Truth Fidelity:** Preserves sharp categorical boundaries from the native 10m ESA WorldCover layer without inducing boundary blur or mixed-pixel artifacts in the supervision signal.
2. **Feature Representation Density:** At 10m/px, a $224 \times 224$ patch yields $50,176$ spatial feature vectors. This density is crucial for the spatially-stratified confidence-guided sampling operator ($\Phi_c$) to select up to $N_{\max} = 1024$ reliable feature vectors per class per mini-batch for Gram matrix evaluation.
3. **Physical Footprint Equivalence:** Ensures a $1:1$ physical spatial scale across domains:

$$\text{Patch Size: } 224 \times 224 \text{ pixels} \times 10\text{ m/pixel} = 2240\text{ m} \times 2240\text{ m} \quad (2.24\text{ km} \times 2.24\text{ km})$$


4. **Receptive Field Preservation:** Allows the convolutional layers of the ResUNet backbone to operate on identical physical ground sample distances (GSD) across both domains.

---

## 5. Geographic Regions of Interest (ROIs) & Spatial Block Partitioning

To prevent spatial autocorrelation and data leakage, domain partitioning is strictly geographic rather than random.

### 5.1 Geographic Boundaries

* **Source Region (Brazil):** São Félix do Xingu, Pará state ($\sim 10,000\text{ km}^2$ bounding box). Representative of large-scale, consolidated deforestation patterns ("fishbone" and industrial pasture conversions).
* *Coordinates:* `[[-52.30, -7.00], [-52.30, -6.10], [-51.40, -6.10], [-51.40, -7.00]]`


* **Target Region (Colombia):** San José del Guaviare / Caquetá ($\sim 10,000\text{ km}^2$ bounding box). Representative of active, highly fragmented frontier deforestation in the Amazonian-Andean transition.
* *Coordinates:* `[[-72.90, 2.10], [-72.90, 3.00], [-72.00, 3.00], [-72.00, 2.10]]`



### 5.2 Spatial Block Partitioning (Actual Sizes)

Each domain composite is partitioned into three non-overlapping geographic blocks along the Y-axis (rows). The original plan targeted a $60\% / 15\% / 25\%$ train / validation / test split; the **actual** proportions of the final consolidated containers (after water / NoData and class-balance filtering) are:

| Domain | Train | Validation | Test | Total |
| --- | --- | --- | --- | --- |
| Source (Brazil) | 69.4% (3,944) | 19.5% (1,108) | 11.1% (633) | 5,685 |
| Target (Colombia) | 71.4% (3,711) | 17.2% (894) | 11.3% (589) | 5,194 |

* **Target Train:** Unlabeled spatial block used exclusively during UDA optimization for calculating the $\alpha$-Rényi alignment loss ($\tilde{I}_2$). Although a mask tensor is physically stored in the container, it is deliberately ignored during adaptation (treated as unlabeled).
* **Target Validation:** Labeled spatial block used for early stopping, model selection, and hyperparameter tuning ($\lambda=0.1, \tau=0.85$).
* **Target Test:** Labeled spatial block held out entirely until final evaluation to report mIoU and F1-score metrics.

---

## 6. Local Python Preprocessing, On-The-Fly Resampling & Dataset Consolidation

To guarantee deterministic spatial alignment and numerical stability, a Python pipeline (`rasterio`, `numpy`, `h5py`) performs local cubic resampling, spectral scaling, NaN sanitization, and sliding-window extraction.

### 6.1 On-The-Fly Spatial Resampling Pipeline

* **Source Domain:** Sentinel-2 ($10023 \times 10019$ px) ingested natively at 10m/px.
* **Target Domain:** Landsat-8 ingested at native 30m ($3340 \times 3340$ px) and dynamically resampled locally to match the exact 10m Ground Truth spatial grid ($10020 \times 10020$ px) using `rasterio.enums.Resampling.cubic`:

$$\text{Shape}_{\text{Target\_10m}} = (4, \ H_{\text{GT\_10m}}, \ W_{\text{GT\_10m}})$$



### 6.2 Sliding Window Extraction & Quality Filtering

* **Patch Resolution:** $224 \times 224$ pixels ($\text{Channels} = 4$, $\text{Height} = 224$, $\text{Width} = 224$).
* **Stride Length:** $112$ pixels ($50\%$ spatial overlap across adjacent patches).
* **Water / NoData Filtering:** Discards patches with $> 10\%$ pixels classified as water or missing data.
* **Entropy & Class-Balance Filtering:** Discards homogeneous patches; retains patches only if valid forest ratio falls strictly within $[15\%, 85\%]$.

### 6.3 Final Dataset Consolidation Summary

Patches are compiled into six dedicated HDF5 (`.h5`) container files using **LZF compression** (`compression="lzf"`, chunked per patch):

| File Name | Domain | Split | Patches | Image Tensor Shape | Mask Tensor Shape |
| --- | --- | --- | --- | --- | --- |
| `source_train.h5` | Source (Brazil) | Train | 3,944 | `(3944, 4, 224, 224)` | `(3944, 224, 224)` |
| `source_val.h5` | Source (Brazil) | Validation | 1,108 | `(1108, 4, 224, 224)` | `(1108, 224, 224)` |
| `source_test.h5` | Source (Brazil) | Test | 633 | `(633, 4, 224, 224)` | `(633, 224, 224)` |
| `target_train.h5` | Target (Colombia) | Train | 3,711 | `(3711, 4, 224, 224)` | `(3711, 224, 224)` |
| `target_val.h5` | Target (Colombia) | Validation | 894 | `(894, 4, 224, 224)` | `(894, 224, 224)` |
| `target_test.h5` | Target (Colombia) | Test | 589 | `(589, 4, 224, 224)` | `(589, 224, 224)` |

All six containers are stored in a **single directory** of a public Kaggle dataset and loaded directly from disk with `h5py`:

```
/kaggle/input/datasets/lucasiturriago/dares-amazon-deforestation-uda/
```

The `source_*.h5` / `target_*.h5` naming convention is resolved by the DARES data loader (`dares.data.loader.DARESDataLoader`), which points both the source and target domain directories at this shared folder.

---

## 7. Architectural Design & Ablation Benchmark Matrix

To systematically evaluate the DARES adaptation operator ($\tilde{I}_2$) across different feature extraction capacities and receptive fields, the modular codebase supports a combinatorial matrix of feature backbones and segmentation decoders.

### 7.1 Feature Extractor Backbones (*Encoders*)

1. **ResNet (ResNet-34 / ResNet-50):** Classical convolutional baseline utilizing residual connections for multi-scale feature extraction.
2. **ConvNeXt (ConvNeXt-Tiny / Small):** Modernized pure-convolutional architecture incorporating $7 \times 7$ depthwise convolutions and inverted bottlenecks.
3. **Swin Transformer (Swin-T / Swin-S):** Hierarchical Vision Transformer using shifted window self-attention mechanisms to model long-range spatial context.

### 7.2 Segmentation Decoders (*Heads*)

1. **UNet Decoder:** Progressive upsampling decoder with skip connections to directly recover high-resolution spatial boundaries.
2. **DeepLabV3+ Head:** Decoder based on **Atrous Spatial Pyramid Pooling (ASPP)** and depthwise separable atrous convolutions, capturing multi-scale context without loss of spatial resolution.

---

## 8. Methodological Justifications for Parameter Choices

### 8.1 Patch Dimensions ($224 \times 224$) and Stride ($112$)

* **Backbone Compatibility:** Aligns directly with standard ImageNet-pretrained encoders, preserving receptive field properties.
* **Spatial Scale:** At 10m/px, each patch covers $2.24\text{ km} \times 2.24\text{ km}$ ($5.01\text{ km}^2$), providing sufficient context for local land-use patterns.

### 8.2 Local Python-Side Resampling over GEE Export-Side Resampling

* **Elimination of Projection Evaluation Artifacts:** Server-side GEE resampling on complex temporal composites can collapse spatial variance when forced across projection boundaries. Native 30m export followed by Python `rasterio` cubic resampling guarantees deterministic pixel-grid alignment with zero spatial variance degradation ($\text{Std} > 0$).

### 8.3 HDF5 Container Format with LZF Compression

* **I/O Throughput:** LZF compression provides high-speed decompression with minimal CPU overhead compared to GZIP, ensuring that disk reads do not bottleneck GPU utilization during PyTorch training loops.