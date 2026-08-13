# Controlled tempered-sampler retraining

This run changes only the training sampler. It keeps the saved U-Net training
configuration, model, loss, normalization and augmentation settings unchanged.

```bash
floodmap train \
  --config $ARTIFACTS_DIR/flood_baseline_512_20260713_194723/config.yaml \
  --processed-data-dir $PROCESSED_DIR \
  --artifacts-dir $ARTIFACTS_DIR \
  --run-id flood_tempered_sampler_control \
  --event-balanced-sampling \
  --event-balance-power 0.5 \
  --event-tile-weight-cap 5.0 \
  --weighted-samples-multiplier 1.0 \
  --max-epochs 30 \
  --patience 10 \
  --threshold-sweep \
  --gpu \
  --plain-progress
```

Expected sampler log:

```text
Event-balanced foreground-ratio sampling: ... | power=0.50 | cap=5.00x median
```

The purpose of this run is to isolate whether correcting the severe exposure
concentration improves validation F1. Crop-aware sampling and boundary losses
should not be introduced until this controlled comparison has completed.
