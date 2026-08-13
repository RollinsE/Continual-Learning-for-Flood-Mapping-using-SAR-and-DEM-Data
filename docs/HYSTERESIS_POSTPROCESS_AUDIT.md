# Seeded hysteresis post-processing audit

`audit-hysteresis-postprocess` tests whether a low probability threshold can recover flood extent without accepting every low-confidence region.

For each tile, the audit:

1. identifies low-threshold connected components;
2. keeps a component only when it contains a minimum number of high-threshold seed pixels;
3. applies the configured minimum connected-component area;
4. compares the result with fixed probability thresholds from the same checkpoint.

The decision is conservative. Hysteresis must outperform both:

- the best guard-eligible fixed threshold; and
- the better fixed threshold at its selected low/high endpoints.

The default decision guard allows no more than the configured recall loss and no increase in the empty-tile false-positive rate relative to the reference operating point.

Example:

```bash
floodmap audit-hysteresis-postprocess \
  --config $ARTIFACTS_DIR/baseline/config.yaml \
  --checkpoint $ARTIFACTS_DIR/baseline/models/model_best.pth \
  --processed-data-dir $PROCESSED_DIR \
  --output-dir $ARTIFACTS_DIR/hysteresis_audit \
  --split val \
  --include-events EMSR342 \
  --fixed-thresholds 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --low-thresholds 0.15 0.20 0.25 0.30 0.35 0.40 \
  --high-thresholds 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --min-seed-pixels 1 16 64 \
  --min-component-areas 96 \
  --reference-threshold 0.50 \
  --reference-min-component-area 96 \
  --max-recall-drop 0.02 \
  --max-empty-fp-rate-increase 0.0 \
  --no-pretrained \
  --no-amp \
  --gpu \
  --plain-progress
```

Outputs:

- `summary.json`
- `hysteresis_sweep.csv`
- `tile_setting_metrics.csv`
- `event_setting_metrics.csv`
- `event_comparison.csv`
