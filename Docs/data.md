# TECHNICAL SPECIFICATION REPORT: DATASET CONSTRUCTION AND PREPROCESSING PROTOCOL FOR DARES

**Project Title:** Domain Adaptation via $\alpha$-Rényi Entropy for Semantic Segmentation (DARES)

**Target Application:** Multi-Level Radiometric and Cross-Region Deforestation Mapping (Brazil Amazon $\rightarrow$ Colombian Forest Reserves)

**Document Version:** 4.0 (Controlled Multi-Level Covariate Shift & Standardized S2 MSI Benchmark)

---

## 1. Executive Overview & Experimental Setup

This technical specification documents the complete data engineering, geographic partitioning, and radiometric perturbation protocols developed for the **DARES** benchmark. The pipeline establishes an Unsupervised Domain Adaptation (UDA) testbed for dense semantic segmentation under simultaneous **geographic domain shift** and **controlled non-linear radiometric covariate shift**:

* **Source Domain ($\mathcal{D}_s$):** Sentinel-2 MSI multispectral imagery over the active deforestation frontier of the Brazilian Amazon (São Félix do Xingu, Pará; $\sim 10,000\text{ km}^2$). Fully supervised with ground truth annotations ($Y_s$).
* **Target Domain ($\mathcal{D}_t$):** Sentinel-2 MSI multispectral imagery over the Colombian Amazonian deforestation arc (San José del Guaviare; $\sim 10,000\text{ km}^2$). Unlabeled during adaptation training ($X_t$), with spatial block subsets reserved for validation ($Y_{t,\text{val}}$) and benchmark evaluation ($Y_{t,\text{test}}$).
* **Controlled Stress-Test Regimes ($\mathcal{D}_t^{\text{LIME}}$):** Three discrete degradation tiers (Low, Medium, High) generated via Retinex-based Low-Light Image Enhancement (LIME) attenuation operators to evaluate model resilience against atmospheric scattering, cloud shadowing, and illumination non-homogeneity.

```
+-----------------------------------------------------------------------------------+
|                              DARES UDA BENCHMARK DATASET                          |
+-----------------------------------------+-----------------------------------------+
|              SOURCE DOMAIN              |              TARGET DOMAIN              |
|        São Félix do Xingu, Brazil       |       San José del Guaviare, Colombia   |
|            (Sentinel-2 MSI)             |             (Sentinel-2 MSI)            |
+-----------------------------------------+-----------------------------------------+
| Fully Supervised (Ys)                   | Clean Baseline (Target_Original)        |
| Splits: Train / Val / Test              | LIME Low Perturbation (Target_Low)      |
|                                         | LIME Med Perturbation (Target_Medium)   |
|                                         | LIME High Perturbation (Target_High)    |
+-----------------------------------------+-----------------------------------------+

```

---

## 2. Satellite Data Specifications & Spectral Normalization

Both source and target acquisitions are temporally synchronized to the calendar year **2021** to match the thematic epoch of the ESA WorldCover reference layer and minimize inter-annual phenological discrepancies.

### 2.1 Spectral Channel Configuration

Both domains utilize identical 4-channel multispectral tensors composed of the native 10m Ground Sample Distance (GSD) bands from the Sentinel-2 MultiSpectral Instrument (Harmonized Level-2A, Bottom-of-Atmosphere Surface Reflectance):

* **Channel 0 (`B2`):** Blue ($\lambda_c \approx 490\text{ nm}$, $10\text{m GSD}$)
* **Channel 1 (`B3`):** Green ($\lambda_c \approx 560\text{ nm}$, $10\text{m GSD}$)
* **Channel 2 (`B4`):** Red ($\lambda_c \approx 665\text{ nm}$, $10\text{m GSD}$)
* **Channel 3 (`B8`):** Near-Infrared / NIR ($\lambda_c \approx 842\text{ nm}$, $10\text{m GSD}$)

### 2.2 Temporal Aggregation & Reflectance Scaling

* **Aggregation:** Pixel-wise median composite generated over cloud-filtered scenes ($\text{QA60}$ bitmask verified) acquired across 2021.
* **Physical Surface Reflectance Normalization:** Raw integer Surface Reflectance (SR) values ($\text{DN} \in [0, 10000]$) are scaled to physical floating-point reflectances in the range $[0.0, 1.0]$:

$$X = \text{clip}\left(\frac{\text{DN}}{10000.0}, \ 0.0, \ 1.0\right) \in \mathbb{R}^{4 \times H \times W}$$

Residual NoData values are converted into `np.nan` and synchronized across all spectral bands prior to patch extraction.

---

## 3. Ground Truth Taxonomy, Binarization & Exclusion Protocol

Reference annotations are derived from the **ESA WorldCover 2021 (v200)** land cover dataset at 10m spatial resolution.

### 3.1 Binary Deforestation Taxonomy

The multi-class land cover taxonomy is binarized to isolate the deforestation dynamic:

* **Class 1 (Forest / Target Class):** ESA WorldCover Class `10` (*Tree cover*).
* **Class 0 (Non-Forest / Anthropogenic):** ESA WorldCover Classes `20` (*Shrubland*), `30` (*Grassland*), `40` (*Cropland*), `50` (*Built-up*), and `60` (*Bare / sparse vegetation*).

### 3.2 Hydrological & Wetland Masking Protocol

* **Masked Classes:** ESA WorldCover Classes `80` (*Permanent water bodies*) and `90` (*Herbaceous wetland*).
* **Implementation:** Pixels associated with classes `80` and `90` are converted into `NaN` in ground truth rasters and ignored (`ignore_index = 255`) during loss computation.
* **Theoretical Justification:** Water bodies exhibit near-zero NIR reflectance, which creates an artificial bimodal distribution within Class 0 when merged with bare soil or pasture. Masking these classes ensures that the Gram matrices ($K_S, K_T$) evaluated in the Reproducing Kernel Hilbert Space (RKHS) capture terrestrial land-use variation without hydrological bias.

---

## 4. Multi-Level LIME Perturbation Protocol (Controlled Covariate Shift)

To evaluate model stability under non-uniform illumination and atmospheric attenuation, a controlled Retinex-based degradation pipeline is applied offline to the Target domain.

### 4.1 Mathematical Formulation

1. **Initial Illumination Map Estimation ($T_0$):**

$$T_0(x) = \max_{c \in \{1, 2, 3, 4\}} X_c(x)$$

2. **Edge-Preserving Guided Filter Refinement:**
Using the spatial mean of visible channels as guidance $I_{\text{guide}} = \frac{1}{3}\sum_{c=1}^3 X_c$, the illumination map is refined via linear coefficients $(a_k, b_k)$ over local windows $\omega_k$ of radius $r=15$ and regularization $\epsilon=10^{-3}$:

$$T(x) = \bar{a}(x) I_{\text{guide}}(x) + \bar{b}(x), \quad \text{with } T(x) \leftarrow \text{clip}(T(x), 0.01, 1.0)$$

3. **Non-Linear Exponentiation and Spectral Modulation:**
Given a severity parameter $s \in [0.0, 1.0]$, the attenuation exponent $\gamma$ and perturbed tensor $X_{\text{perturbed}}$ are given by:

$$\gamma = 1.0 + 3.5 \cdot s$$

$$X_{\text{perturbed}}(x) = \text{clip}\left(X(x) \odot \frac{T(x)^\gamma}{T(x) + 10^{-6}}, \ 0.0, \ 1.0\right)$$

```
Input Patch X (4, 224, 224)
        │
        ├──> T_0(x) = max_c(X_c) ────────────────┐
        │                                        ▼
        └──> Guide = mean(B2, B3, B4) ──> [Guided Filter 2D] ──> T(x) refined
                                          (r=15, eps=1e-3)         │
                                                                   ▼
                                                            gamma = 1 + 3.5 * s
                                                                   │
                                                                   ▼
X_perturbed = clip( X * (T(x)^gamma / T(x)), 0.0, 1.0 ) <──────────┘

```

### 4.2 Severity Sampling Matrix

Severities are sampled uniformly at random per patch from discrete candidate pools using a fixed pseudo-random seed (`seed = 42`):

| Perturbation Level | Severity Pool ($\mathcal{S}$) | Simulated Physical Condition |
| --- | --- | --- |
| **Clean / Original** | $s = 0.0$ | Clear-sky surface reflectance |
| **Target_Low** | $s \in \{0.1, 0.2\}$ | Mild haze and slight sun-angle variation |
| **Target_Medium** | $s \in \{0.3, 0.4, 0.5\}$ | Moderate haze and cloud margin shadowing |
| **Target_High** | $s \in \{0.6, 0.7\}$ | Severe atmospheric attenuation and deep topographic cast shadows |

---

## 5. Geographic Regions of Interest (ROIs) & Spatial Splitting

Domain partitioning is executed along the north-to-south spatial axis (Y-axis rows) to eliminate spatial autocorrelation and prevent spatial data leakage between splits.

### 5.1 Geographic Bounding Boxes

* **Source ROI (São Félix do Xingu, Brazil):**
`[[-52.30°, -7.00°], [-52.30°, -6.10°], [-51.40°, -6.10°], [-51.40°, -7.00°]]`
*Characteristics:* Industrial pasture conversion, large clearings, fishbone colonization.
* **Target ROI (San José del Guaviare, Colombia):**
`[[-72.90°, 2.10°], [-72.90°, 3.00°], [-72.00°, 3.00°], [-72.00°, 2.10°]]`
*Characteristics:* Active agricultural frontier, fragmented deforestation, Andean-Amazonian transition.

### 5.2 Y-Axis Spatial Block Partitioning

The master raster grids are sliced into contiguous spatial blocks along the Y-axis:

* **Train Split:** Top $70\%$ of rows ($y \in [0, 0.70 \cdot H_{\text{total}}]$).
* **Validation Split:** Intermediate $20\%$ of rows ($y \in [0.70 \cdot H_{\text{total}}, 0.90 \cdot H_{\text{total}}]$).
* **Test Split:** Bottom $10\%$ of rows ($y \in [0.90 \cdot H_{\text{total}}, H_{\text{total}}]$).

---

## 6. Patch Extraction, Filtering & HDF5 Storage Architecture

Sub-images are extracted using a sliding-window protocol ($224 \times 224\text{ px}$, $50\%$ stride / $112\text{ px}$) and passed through quality filters:

1. **NoData / Hydrological Rejection:** Discards patches where missing/water pixels exceed $10\%$ ($\text{ratio}_{\text{NoData}} > 0.10$).
2. **Entropy / Class Balance Constraint:** Retains patches only if the valid forest ratio satisfies:

$$0.15 \le \frac{\sum \mathbb{I}(Y_{\text{valid}} = 1)}{N_{\text{valid}}} \le 0.85$$

### 6.1 Final Consolidated HDF5 Structure

All data subsets are stored in compressed HDF5 containers using LZF compression (`compression="lzf"`) chunked per patch:

```text
/content/drive/MyDrive/GEE_DARES_Dataset/hdf5_processed/
├── Source/
│   ├── source_train.h5         (Images: [N_str, 4, 224, 224] float32 | Masks: [N_str, 224, 224] uint8)
│   ├── source_val.h5           (Images: [N_sva, 4, 224, 224] float32 | Masks: [N_sva, 224, 224] uint8)
│   └── source_test.h5          (Images: [N_ste, 4, 224, 224] float32 | Masks: [N_ste, 224, 224] uint8)
├── Target_Original/
│   ├── target_train.h5         (Images: [N_ttr, 4, 224, 224] float32 | Masks: [N_ttr, 224, 224] uint8)
│   ├── target_val.h5           (Images: [N_tva, 4, 224, 224] float32 | Masks: [N_tva, 224, 224] uint8)
│   └── target_test.h5          (Images: [N_tte, 4, 224, 224] float32 | Masks: [N_tte, 224, 224] uint8)
├── Target_Low/
│   ├── target_train_lime_low.h5
│   ├── target_val_lime_low.h5
│   └── target_test_lime_low.h5
├── Target_Medium/
│   ├── target_train_lime_med.h5
│   ├── target_val_lime_med.h5
│   └── target_test_lime_med.h5
└── Target_High/
    ├── target_train_lime_high.h5
    ├── target_val_lime_high.h5
    └── target_test_lime_high.h5

```

*Final Distribution Container:* `hdf5_processed.zip` (compressed via `ZIP_STORED` / zero overhead to preserve internal LZF block layout).

---

## 7. Experimental Evaluation Matrix

The final evaluation protocol benchmarks model resilience across all target variations:

| Experiment ID | Model / Adaptation Strategy | Target Training Subset | Test Evaluation Set | Target Metric |
| --- | --- | --- | --- | --- |
| **EXP-01** | Source Only (Supervised Baseline) | None (Trained on Source only) | `target_test.h5` | mIoU / F1-Score |
| **EXP-02** | ADVENT (Entropy Minimization) | `target_train.h5` | `target_test.h5` | mIoU / F1-Score |
| **EXP-03** | DACS (Cross-domain Mixed Sampling) | `target_train.h5` | `target_test.h5` | mIoU / F1-Score |
| **EXP-04** | FDA (Fourier Domain Adaptation) | `target_train.h5` | `target_test.h5` | mIoU / F1-Score |
| **EXP-05** | **DARES** (Rényi Alignment $\tilde{I}_2$) | `target_train.h5` | `target_test.h5` | mIoU / F1-Score |
| **EXP-06** | **DARES** vs. Baselines (Stress Low) | `target_train_lime_low.h5` | `target_test_lime_low.h5` | mIoU / F1-Score |
| **EXP-07** | **DARES** vs. Baselines (Stress Med) | `target_train_lime_med.h5` | `target_test_lime_med.h5` | mIoU / F1-Score |
| **EXP-08** | **DARES** vs. Baselines (Stress High) | `target_train_lime_high.h5` | `target_test_lime_high.h5` | mIoU / F1-Score |