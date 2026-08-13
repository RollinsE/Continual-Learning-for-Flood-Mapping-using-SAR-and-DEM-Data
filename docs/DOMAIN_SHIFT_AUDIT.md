# Target-event domain-shift audit

`floodmap audit-domain-shift` compares one or more difficult target EMSR events
with the processed training split. It is a diagnostic command, not a model
benchmark.

The audit writes:

- `tile_features.csv`: per-tile VV/VH/DEM summaries, normalization clipping
  rates, flood ratio, empty-tile status, connected-component size and boundary
  complexity.
- `pixel_distribution_shift.csv`: all-pixel and class-conditional KS,
  Wasserstein, standardised mean difference and PSI comparisons.
- `tile_feature_shift.csv`: train-versus-target shifts in tile-level input and
  label-geometry features.
- `domain_classifier_metrics.csv`: cross-validated target-event separability
  using sensor/terrain features, label geometry, and both together.
- `domain_classifier_coefficients.csv`: the strongest standardised features
  distinguishing the target event.
- `training_event_similarity.csv`: training events ranked by standardised
  distance to the target event.
- `target_feature_zscores.csv`: the target event's most unusual features
  relative to training-event medians.
- `summary.json`: compact diagnosis and links to all outputs.

Recommended retained-baseline command:

```bash
floodmap audit-domain-shift \
  --config /path/to/retained-run/config.yaml \
  --processed-data-dir /path/to/processed \
  --output-dir /path/to/audits/EMSR342_domain_shift \
  --reference-split train \
  --target-split val \
  --target-events EMSR342 \
  --input-modalities vv vh dem \
  --max-reference-tiles 0 \
  --max-target-tiles 0 \
  --max-pixels-per-tile 256 \
  --max-pixels-per-class-per-tile 128 \
  --max-total-pixels-per-domain 250000 \
  --domain-classifier-reference-ratio 4 \
  --seed 42 \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-level INFO
```

When `--config` supplies `data.normalization_stats_path`, the audit also reports
how frequently the target event lies below or above the train-fitted clipping
limits. Explicit CLI values override config values.

Interpret the classifier AUCs together:

- High `sensor_terrain` AUC indicates the target's input distributions are
  distinguishable from training events.
- High `label_geometry` AUC indicates unusual flood extent, component structure,
  or empty-tile composition in the labelled target event.
- A much higher `combined` AUC than `sensor_terrain` suggests label geometry adds
  material separation beyond input shift.

The classifier uses tile-level cross-validation and can reflect spatial
correlation within an EMSR event. Treat it as evidence for selecting the next
analysis, not as a causal proof of model failure.
