# Derived flood-context channels

The derived-feature workflow provides a controlled six-channel path for the retained VV/VH/DEM model:

1. `vv`
2. `vh`
3. `dem`
4. `vv_vh_log_ratio`
5. `dem_slope`
6. `dem_tpi`

The derived GeoTIFF stores the final three channels in that order. The formulas are:

- `vv_vh_log_ratio = 10 log10(max(VV, eps)) - 10 log10(max(VH, eps))`
- `dem_slope = degrees(arctan(sqrt((dDEM/dx)^2 + (dDEM/dy)^2)))`
- `dem_tpi = DEM - local mean DEM`

The default topographic-position radius is 15 pixels, giving a 31 by 31 pixel window.

## 1. Build derived tiles

```bash
floodmap derive-features \
  --processed-data-dir "$PROCESSED_DIR" \
  --splits train val test \
  --log-ratio-eps 1e-6 \
  --tpi-radius-pixels 15 \
  --plain-progress
```

Existing outputs are skipped. Add `--overwrite` only when intentionally rebuilding them.

The command writes:

```text
<processed>/<split>/derived/<tile>.tif
<processed>/derived_features_manifest.json
```

## 2. Audit separability before training

```bash
floodmap audit-feature-separability \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-dir "$ARTIFACTS_DIR/derived_feature_audit_v088" \
  --fit-split train \
  --eval-split val \
  --base-modalities vv vh dem \
  --extended-modalities vv vh dem vv_vh_log_ratio dem_slope dem_tpi \
  --max-pixels-per-class-per-tile 128 \
  --max-total-pixels-per-split 250000 \
  --seed 42 \
  --plain-progress
```

Outputs include global and event-level logistic comparisons, univariate statistics, and `feature_audit_summary.json`. This is a sampled pixel-level diagnostic, not a segmentation score.

## 3. Fit six-channel normalization while preserving the baseline

Use the retained baseline normalization JSON as the preservation source. VV, VH, and DEM statistics are copied exactly; only the three added channels are newly fitted.

```bash
floodmap fit-normalization \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-file "$ARTIFACTS_DIR/normalization_six_channel_v088.json" \
  --split train \
  --input-modalities vv vh dem vv_vh_log_ratio dem_slope dem_tpi \
  --preserve-channel-stats-from "$BASELINE_STATS" \
  --q-min 1 \
  --q-max 99 \
  --max-pixels-per-file 4096 \
  --seed 1337 \
  --plain-progress
```

## 4. Prepare the controlled warm-start configuration

```bash
floodmap prepare-derived-experiment \
  --base-config "$BASELINE_CONFIG" \
  --baseline-checkpoint "$BASELINE_CHECKPOINT" \
  --normalization-stats-path "$ARTIFACTS_DIR/normalization_six_channel_v088.json" \
  --output-config "$ARTIFACTS_DIR/configs/flood_resnet50_six_channel_v088.yaml" \
  --run-id flood_resnet50_six_channel_v088 \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --batch-size 4 \
  --epochs 20 \
  --patience 6 \
  --encoder-lr 1e-5 \
  --decoder-lr 1e-5 \
  --max-skipped-batch-fraction 0.02 \
  --plain-progress
```

The command clears specialist samplers/crops, uses geometric augmentation, disables AMP, and sets `init_channel_adaptation: zero_extra`. The original three input weights are copied exactly and the three new input-channel weights are initialised to zero.

## 5. Train

```bash
floodmap train \
  --config "$ARTIFACTS_DIR/configs/flood_resnet50_six_channel_v088.yaml" \
  --processed-data-dir "$PROCESSED_DIR" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --run-id flood_resnet50_six_channel_v088 \
  --no-resume \
  --no-pretrained \
  --no-amp \
  --gpu \
  --progress \
  --plain-progress \
  --num-workers 2
```

The run should log that checkpoint input weights were adapted with zero-initialised added channels. Training aborts when skipped non-finite batches exceed the configured epoch budget rather than silently completing an invalid run.

## Terrain-derivative units

DEM elevations are in metres. The implementation therefore converts horizontal raster spacing to metres before calculating slope. Geographic EPSG:4326 tiles use a tile-centre ellipsoidal metres-per-degree approximation; projected rasters use their declared linear-unit conversion. Existing schema-v1 derived rasters are automatically regenerated.
