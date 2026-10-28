# Data contract

The dataset is not included in this repository.

## Processed directory layout

```text
<processed-data-dir>/
  train/
    sar/*.tif
    dem/*.tif
    mask/*.tif
  val/
    sar/*.tif
    dem/*.tif
    mask/*.tif
  test/
    sar/*.tif
    dem/*.tif
    mask/*.tif
```

For a `VV + VH` model, each SAR tile must contain the VV and VH bands in that order. DEM tiles may remain present on disk, but they are not loaded unless `dem` is listed in `--input-modalities`.

Each SAR, DEM, and mask triplet must share the same filename stem, spatial dimensions, CRS, and affine transform. Masks use flood value `1`, background values `0` or `2`, and ignore value `255` under the reference preprocessing profile.

Raw rasters that are not on the SAR grid are rejected unless preprocessing runs with `--align-to-reference-grid`, which reprojects DEM and mask onto the SAR grid (uncovered mask areas become `255`, uncovered DEM becomes NaN). `floodmap audit-raw-alignment` reports mismatches before preprocessing.

## Units and no-data

| Layer | Stored as | No-data |
|---|---|---|
| SAR (VV, VH) | float32, dB (`sar_transform: db10`) | NaN — from source nodata, non-finite or non-positive linear values (Sentinel-1 GRD uses 0 outside the swath) |
| DEM | float32 metres, clipped to [-100, 6000] | NaN — source nodata, non-finite, or outside [-1000, 9000] m (e.g. -32768 voids) |
| Mask | uint8: 0 background, 1 flood | 255 — source ignore values and every pixel without SAR data |

`preprocessing_manifest.json` records the SAR units and these conventions. During training, a pixel whose label is valid but where any *active* input channel is NaN (for example a DEM void in a VV+VH+DEM model) is also set to 255, so it never contributes to the loss or metrics; the normalisation transform fills such inputs with 0 (the training mean) after scaling.

Morphological clean-up (closing with `morph_kernel`) is applied to **training** masks only (`morphology_splits`). Validation and test masks keep the published labels so reported metrics are measured against the official ground truth.

## Normalisation modes

Fit statistics on the training split with `floodmap fit-normalization`, then train with `--normalization-mode robust_minmax`: each channel is clipped to its 1st–99th training percentile, scaled to [0, 1], and standardised with the mean and standard deviation of that scaled training distribution.

`robust_percentile` standardises the [0, 1]-scaled values with the mean and standard deviation of the *unscaled* channel, which leaves channels on very different scales (a DEM in metres ends up with almost no variation). It is supported for loading checkpoints trained with it and is not recommended for new models.

## Split integrity

- Split membership must be controlled by immutable metadata.
- Events used in event-level cross-validation come only from the training split.
- Validation and test events must not enter fold training or normalisation fitting.
- Normalisation statistics must be fitted on training data only.
- Published results must include split tile counts and an immutable manifest or checksum.

## Reference counts

The reference preprocessing (`configs/reproduction/preprocess_512.yaml`) produces:

```text
train: 3,980 tiles
val:     324 tiles
test:    386 scenes
```

Count differences do not automatically imply an error, but they mean the run is not an exact reproduction of the reference dataset state.
