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

## Split integrity

- Split membership must be controlled by immutable metadata.
- Events used in event-level cross-validation come only from the training split.
- Validation and test events must not enter fold training or normalisation fitting.
- Normalisation statistics must be fitted on training data only.
- Published results must include split tile counts and an immutable manifest or checksum.

## Reference counts

The locked reference preprocessing produced:

```text
train: 3,981 tiles
val:     324 tiles
test:    386 tiles
```

Count differences do not automatically imply an error, but they mean the run is not an exact reproduction of the reference dataset state.
