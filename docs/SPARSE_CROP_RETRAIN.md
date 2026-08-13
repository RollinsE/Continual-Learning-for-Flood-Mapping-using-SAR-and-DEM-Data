# Sparse-flood crop retraining

This experiment changes only the spatial supervision seen by the training dataset. The model architecture, loss, optimiser, scheduler, normalization, validation set, threshold sweep, and tempered event sampler can remain identical to the selected baseline run.

## What the crop mixture does

For training tiles that contain flood pixels, the default mixture is:

- 50% full 512×512 tile
- 25% flood-centred crop
- 25% hard-background crop from the same flood-containing tile

The crop size is sampled uniformly from 256, 320, 384, and 448 pixels, then resized back to the configured 512×512 model input. Empty tiles remain full-size samples. A hard-background crop is accepted only when its flood ratio is at or below the configured tolerance.

## Controlled retraining command

Use the saved configuration from the baseline/tempered run so every setting other than crop supervision is preserved:

```bash
cd $PROJECT_ROOT

floodmap train \
  --config $ARTIFACTS_DIR/<BASELINE_RUN>/config.yaml \
  --processed-data-dir $PROCESSED_DIR \
  --artifacts-dir $ARTIFACTS_DIR \
  --run-id flood_512_r50_sparse_crop_v082 \
  --image-size 512 \
  --num-workers 2 \
  --event-balanced-sampling \
  --event-balance-power 0.5 \
  --event-tile-weight-cap 5.0 \
  --sparse-crop-supervision \
  --sparse-crop-normal-fraction 0.50 \
  --sparse-crop-flood-fraction 0.25 \
  --sparse-crop-hard-background-fraction 0.25 \
  --sparse-crop-sizes 256 320 384 448 \
  --sparse-crop-attempts 24 \
  --sparse-crop-hard-background-max-fg-ratio 0.001 \
  --sparse-crop-min-valid-ratio 0.50 \
  --threshold-sweep \
  --threshold-metric f1 \
  --monitor-threshold-sweep \
  --gpu \
  --progress \
  --plain-progress \
  --no-resume
```

Do not add architecture, loss, learning-rate, or augmentation overrides unless they are intentionally part of a separate experiment. The saved baseline configuration is the source of truth for those settings.

## Expected logs

At dataset setup, the run reports the configured mixture and eligible crop sizes. After each training epoch it reports:

```text
Training crop supervision requested: normal=... flood-centred=... hard-background=...
Training crop supervision applied: normal=... flood-centred=... hard-background=... | fallbacks to full tile: ...
Training crop sizes before resize: 256=... 320=... 384=... 448=...
```

A fallback means a requested crop could not satisfy the flood/valid-pixel rules and the full tile was used instead.

## Authoritative validation metric

When threshold-sweep monitoring is enabled, the epoch summary now reports all validation metrics at the selected threshold. The checkpoint metric and the displayed validation F1 therefore refer to the same operating point.
