# Permanent-water prior audit

`floodmap audit-water-prior` tests whether long-term surface-water occurrence can
reduce false flood detections without retraining the segmentation model.

The command uses the European Commission Joint Research Centre Global Surface
Water occurrence layer hosted as Cloud Optimized GeoTIFFs by Microsoft
Planetary Computer. The hosted collection summarises water occurrence from
1984 to 2020. Occurrence values are percentages from 0 to 100; 255 is no data.

The command:

1. aligns occurrence to every selected processed MMFlood tile and caches it;
2. runs the retained checkpoint once;
3. evaluates hard exclusion and soft probability penalties;
4. compares every setting with an explicit unmodified reference operating point;
5. reports an unconstrained best setting and a recall-guarded best setting.

The audit does not change model weights. For events before 2020, the 1984-2020
occurrence layer can include observations after the event, so results are a
diagnostic rather than a leakage-free benchmark.

Example targeted audit:

```bash
floodmap audit-water-prior \
  --config $ARTIFACTS_DIR/flood_baseline_512_20260713_194723/config.yaml \
  --checkpoint $ARTIFACTS_DIR/flood_baseline_512_20260713_194723/models/model-020_best_f1-0.4941.pth \
  --processed-data-dir $PROCESSED_DIR \
  --output-dir $ARTIFACTS_DIR/water_prior_audit_EMSR342_v090 \
  --prior-cache-dir $ARTIFACTS_DIR/jrc_water_prior_cache_v090 \
  --split val \
  --include-events EMSR342 \
  --model-thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --occurrence-thresholds 75 90 95 99 \
  --penalty-strengths 0.25 0.50 0.75 1.00 \
  --min-component-areas 96 \
  --reference-threshold 0.50 \
  --reference-min-component-area 96 \
  --max-recall-drop 0.02 \
  --no-pretrained \
  --no-amp \
  --gpu \
  --batch-size 4 \
  --num-workers 2 \
  --plain-progress
```

Important outputs:

- `summary.json`
- `water_prior_sweep.csv`
- `prior_label_overlap.csv`
- `event_comparison.csv`
- `tile_setting_metrics.csv`
- `aligned_water_prior_index.csv`

## Decision interpretation

The audit separates threshold tuning from prior effects. The audit reports:

- the fixed reference operating point;
- the best no-prior threshold/component-area setting;
- the best prior setting;
- the prior gain over the best no-prior setting; and
- the incremental prior gain against the no-prior result at the same model threshold.

A lower model threshold may improve F1, but that improvement is not evidence that the water prior helped. The recommendation only proceeds when the prior itself adds measurable value beyond the no-prior comparator.
