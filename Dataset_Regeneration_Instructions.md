# Instrucciones de regeneración del dataset DARES (agua/humedal → 255)

**Objetivo:** corregir el pipeline de preprocesamiento para que el agua/humedal (ESA WorldCover clases 80 y 90) quede etiquetada como **255 (ignorar)**, en vez de plegarse a **0 (no-bosque)**. Después, regenerar los `.h5` y re-subir el dataset a Kaggle.

---

## 1. Contexto del problema

**Qué se quería:** tres valores de etiqueta → `1` (bosque), `0` (no-bosque), `255` (agua/humedal, ignorado).

**Qué pasó:** el script de GEE **enmascaraba** el agua/humedal (NoData) en vez de etiquetarla con 255. Al exportar el GeoTIFF, el NoData se escribió como **0**, y el pipeline Python lo confirmó con:

```python
mask[mask == gt_nodata] = np.nan
...
mask_patch_clean = np.nan_to_num(mask_patch, nan=0.0)   # NoData -> 0
```

Resultado: las máscaras `.h5` solo contenían `{0, 1}` y el agua/humedal se contaba como no-bosque.

**Ya corregido en GEE:** el GT nuevo (`source_brasil_gt_2021_10m.tif`, `target_colombia_gt_2021_10m.tif`) ahora contiene `{0, 1, 255}`:

| Archivo | 0 (no-bosque) | 1 (bosque) | 255 (agua/humedal) |
|---------|---------------|------------|--------------------|
| source_brasil_gt_2021_10m.tif | 52.69% | 45.67% | **1.64%** |
| target_colombia_gt_2021_10m.tif | 39.44% | 59.34% | **1.22%** |

**Falta:** adaptar el cuaderno de preprocesamiento Python para **preservar el 255** y **excluirlo del filtro y de las métricas**.

---

## 2. Mapa de clases (referencia)

| ESA WorldCover | Valor en la máscara | Significado |
|----------------|---------------------|-------------|
| 10 tree cover | **1** | Bosque |
| 20 shrubland | **0** | No-bosque |
| 30 grassland | **0** | No-bosque |
| 40 cropland | **0** | No-bosque |
| 50 built-up | **0** | No-bosque |
| 60 bare/sparse | **0** | No-bosque |
| 70 snow/ice | **255** | Ignorar |
| 80 permanent water | **255** | Ignorar |
| 90 herbaceous wetland | **255** | Ignorar |
| 95 mangroves | **255** | Ignorar |
| 100 lichen/moss | **255** | Ignorar |
| nodata | **255** | Ignorar |

---

## 3. Cambios en el preprocesamiento

### 3.1 `load_raster_pair` — preservar el 255 (no convertirlo a 0)

```python
def load_raster_pair(image_path: Path, gt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(image_path) as src_img:
        image = src_img.read().astype(np.float32)
        img_nodata = src_img.nodata

    with rasterio.open(gt_path) as src_gt:
        mask = src_gt.read(1).astype(np.int16)   # 0, 1 y 255
        gt_nodata = src_gt.nodata

    # Imagen: nodata -> NaN (igual que antes)
    if img_nodata is not None:
        image[:, image[0] == img_nodata] = np.nan

    # GT: cualquier nodata real -> 255 (ignorar), NO -> 0
    if gt_nodata is not None:
        mask[mask == gt_nodata] = 255

    # Donde la imagen no tenga dato válido, también ignoramos la etiqueta
    img_invalid = np.isnan(image).any(axis=0)
    mask[img_invalid] = 255

    # Normalización de reflectancia (DN crudo -> [0,1])
    valid_pixels = image[:, ~img_invalid]
    if valid_pixels.size > 0 and np.nanmax(valid_pixels) > 10.0:
        image = image / 10000.0
    image = np.clip(image, 0.0, 1.0)
    image = np.nan_to_num(image, nan=0.0)   # NaN de imagen -> 0 (la etiqueta ya es 255)

    return image, mask
```

**Diferencias con la versión actual:**
- ❌ Eliminar `mask[mask == gt_nodata] = np.nan`
- ❌ Eliminar `mask[invalid_mask] = np.nan`
- ✅ Añadir `mask[mask == gt_nodata] = 255`
- ✅ Añadir `mask[img_invalid] = 255`
- ✅ Leer la máscara como `np.int16` (para poder marcar 255 sin ambigüedad)

### 3.2 `extract_filtered_patches` — el 255 es "inválido", no "no-bosque"

```python
def extract_filtered_patches(image_block, mask_block, config):
    _, height, width = image_block.shape
    p_size, stride = config.patch_size, config.stride
    extracted_images, extracted_masks = [], []
    total_pixels = p_size * p_size

    row_coords = list(range(0, height - p_size + 1, stride))
    col_coords = list(range(0, width - p_size + 1, stride))

    for r in row_coords:
        for c in col_coords:
            img_patch  = image_block[:, r:r+p_size, c:c+p_size]
            mask_patch = mask_block[r:r+p_size, c:c+p_size]

            # 1) Filtro: ratio de INVÁLIDO (255 = agua/humedal) < 10%
            invalid_count = (mask_patch == 255).sum()
            if (invalid_count / total_pixels) > config.max_nodata_ratio:
                continue

            valid = (mask_patch != 255)
            valid_count = valid.sum()
            if valid_count == 0:
                continue

            # 2) Filtro: ratio forestal en [0.15, 0.85] sobre píxeles VÁLIDOS
            forest_count = (mask_patch[valid] == 1).sum()
            forest_ratio = forest_count / valid_count
            if not (config.min_forest_ratio <= forest_ratio <= config.max_forest_ratio):
                continue

            extracted_images.append(img_patch.astype(np.float32))
            extracted_masks.append(mask_patch.astype(np.uint8))   # preserva 255

    if not extracted_images:
        return (np.empty((0, 4, p_size, p_size), dtype=np.float32),
                np.empty((0, p_size, p_size), dtype=np.uint8))

    return np.stack(extracted_images, axis=0), np.stack(extracted_masks, axis=0)
```

**Diferencias con la versión actual:**
- `nan_count = np.isnan(mask_patch).sum()` → **`(mask_patch == 255).sum()`**
- `valid_mask = ~np.isnan(mask_patch)` → **`valid = (mask_patch != 255)`**
- ❌ Eliminar `np.nan_to_num(mask_patch, nan=0.0)`
- ✅ Usar `mask_patch.astype(np.uint8)` (conserva el 255)

> **Nota:** `config.max_nodata_ratio = 0.10` ahora se interpreta como "máximo ratio de agua/humedal/inválido por parche". Es consistente con la redacción del manuscrito ("rejecting patches with more than 10% missing/water pixels").

### 3.3 `save_to_hdf5` — sin cambios
Guarda `masks` como `uint8`; el 255 se preserva automáticamente.

---

## 4. Cambios en la EVALUACIÓN (excluir el 255)

El cálculo de métricas (mIoU/DICE) debe **ignorar** los píxeles con etiqueta 255:

```python
valid = (target != 255)
pred   = pred[valid]
target = target[valid]
# ... calcular la matriz de confusión y mIoU/DICE sobre los píxeles válidos
```

Si no se excluye, el agua/humedal entraría como "no-bosque" en las métricas.

**Recomendado (para R2-3 / R3-6):** aprovechar y añadir la **evaluación per-patch + prueba pareada (Wilcoxon o t-test)** en el mismo `evaluate`.

---

## 5. Verificación tras regenerar los `.h5`

Ejecutar sobre cada `.h5` generado:

```python
import h5py, numpy as np, os

base = "/content/drive/MyDrive/GEE_DARES_Dataset/hdf5_processed"
files = [
    ("Source", "source_train.h5"), ("Source", "source_val.h5"), ("Source", "source_test.h5"),
    ("Target_Original", "target_train.h5"), ("Target_Original", "target_val.h5"),
    ("Target_Original", "target_test.h5"),
]

for subdir, name in files:
    path = os.path.join(base, subdir, name)
    if not os.path.exists(path):
        print(f"[FALTA] {subdir}/{name}")
        continue
    with h5py.File(path, "r") as h:
        masks = h["masks"]
        n = masks.shape[0]
        hist = {}
        total = 0
        for i in range(n):
            m = masks[i]
            vals, counts = np.unique(m, return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                hist[v] = hist.get(v, 0) + c
            total += m.size
    print(f"\n=== {subdir}/{name} | parches: {n} | píxeles: {total:,} ===")
    for v in sorted(hist):
        print(f"   valor {v:>3}: {hist[v]:>12,} ({100*hist[v]/total:6.2f}%)")
```

**Criterio de éxito:** debe aparecer el **valor 255** en cada split (≈1–2%). Si no aparece → el cambio no se aplicó.

---

## 6. Re-subida a Kaggle

1. **Regenerar** los `.h5` (Source/, Target_Original/, Target_Low/, Target_Medium/, Target_High/).
2. **Actualizar la descripción** del dataset en Kaggle para reflejar la taxonomía de tres valores (`0`, `1`, `255`):
   - Donde hoy dice "Masked / NoData (ignore_index=255): ESA Classes 80 and 90", aclarar que ahora **sí se exportan como 255** en las máscaras.
3. **Subir una nueva versión** del dataset (Dataset → New Version).
4. **Actualizar el enlace/DOI** en el manuscrito si cambia (`v2/R1_template.tex`, sección Data Availability).
5. Verificar que la versión subida contiene el 255.

---

## 7. Impacto esperado

- Los **números del manuscrito cambiarán** (el agua/humedal ya no cuenta como no-bosque). Como el agua es **~1.2–1.6%**, el cambio será **modesto** pero real.
- Habrá que **re-correr todos los experimentos**: main, baselines, ablaciones, arquitecturas, LIME (low/med/high) y `s=0`.
- Habrá que **reescribir la sección de Resultados** (tablas, figuras, trayectoria de estrés).
- Esto **resuelve de forma consistente** la afirmación de §3.1 sobre `ignore_index=255`.

---

## 8. Checklist final

- [ ] `load_raster_pair` modificado (nodata → 255, no → 0)
- [ ] `extract_filtered_patches` modificado (255 = inválido)
- [ ] `save_to_hdf5` guarda 255 (uint8)
- [ ] `evaluate` excluye 255 de las métricas
- [ ] (Recomendado) `evaluate` añade per-patch + t-test
- [ ] `.h5` regenerados y verificados (aparece 255)
- [ ] Nueva versión subida a Kaggle
- [ ] Re-correr experimentos
- [ ] Reescribir Resultados del manuscrito
