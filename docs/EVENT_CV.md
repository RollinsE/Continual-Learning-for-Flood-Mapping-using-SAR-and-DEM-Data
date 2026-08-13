# Architecture-independent event-level cross-validation

`floodmap event-cv` is the model-selection stage. It uses the same registered model specification as `floodmap train`, so ImageNet, TerraMind, CROMA, SSL4EO, FG-MAE SAR and random initialisation are evaluated under one event-separated protocol.

## Invariants across architectures

For every candidate, the pipeline:

- builds folds from the processed `train` split only;
- holds out complete EMSR events rather than random tiles;
- rejects training/validation event overlap;
- uses the same deterministic fold plan for every candidate;
- selects checkpoints by held-out event-macro F1 and records worst-event F1;
- writes fold checkpoints and per-event metrics;
- creates out-of-fold predictions for threshold calibration;
- leaves the external validation and test splits untouched during model selection.

For MMFlood-derived normalisation, statistics are fitted separately on each fold's training events. Registered provider statistics such as TerraMind or SSL4EO are fixed independently of MMFlood and therefore do not use held-out events.

## Candidate syntax

```text
SOURCE[:DECODER[:ENCODER[:LABEL]]]
```

Examples:

```text
terramind_v1_tiny
croma_sar_base
fgmae_sar_vit_small
ssl4eo_s1_moco
imagenet:unet:resnet34:reference_unet
```

## Outputs

```text
<output-dir>/
├── cv_manifest.json
├── folds.json
├── fold_assignments.csv
├── normalization/<modalities>/fold_XX.json
├── runs/eventcv_<candidate>_<modalities>_foldXX/
│   ├── config.yaml
│   ├── output.log
│   ├── event_validation_history.csv
│   ├── event_validation_metrics.csv
│   ├── cv_result.json
│   └── models/
├── cv_results.csv
├── cv_event_results.csv
├── cv_summary.csv
└── output.log
```

`cv_summary.csv` contains mean event-macro F1, fold variability, worst-event performance, median best epoch and mean fold-selected threshold. The operating threshold is not taken from that mean; it is selected by `calibrate-event-cv` from pooled out-of-fold predictions.

## Path setup

The examples below use shell variables so the same commands work in a local virtual environment, a server, a container, or a notebook runtime:

```bash
export PROCESSED_DIR="/path/to/mmflood_processed"
export RUNS_DIR="/path/to/mmflood_runs"
```

## Plan only

```bash
floodmap event-cv \
  --config configs/training_defaults.yaml \
  --processed-data-dir $PROCESSED_DIR \
  --output-dir $RUNS_DIR/event_cv_terramind_tiny_dem \
  --candidates terramind_v1_tiny \
  --modality-sets vv+vh+dem \
  --folds 5 \
  --seed 42 \
  --plan-only \
  --plain-progress \
  --log-file $RUNS_DIR/event_cv_terramind_tiny_dem/output.log
```

## One-fold smoke test

Use a separate output directory so a one-epoch smoke checkpoint can never be mistaken for a completed full-CV fold.

```bash
floodmap event-cv \
  --config configs/training_defaults.yaml \
  --processed-data-dir $PROCESSED_DIR \
  --output-dir $RUNS_DIR/event_cv_terramind_tiny_dem_smoke \
  --candidates terramind_v1_tiny \
  --modality-sets vv+vh+dem \
  --folds 5 \
  --fold-indices 0 \
  --epochs 1 \
  --patience 1 \
  --batch-size 1 \
  --num-workers 2 \
  --encoder-lr 0.00001 \
  --decoder-lr 0.0001 \
  --weight-decay 0.0001 \
  --optimizer adamw \
  --scheduler poly \
  --loss bce_tversky \
  --loss-alpha 0.3 \
  --loss-beta 0.7 \
  --bce-weight 0.5 \
  --tversky-weight 0.5 \
  --no-amp \
  --gpu \
  --thresholds 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-file $RUNS_DIR/event_cv_terramind_tiny_dem_smoke/output.log
```

## Full five-fold run

```bash
floodmap event-cv \
  --config configs/training_defaults.yaml \
  --processed-data-dir $PROCESSED_DIR \
  --output-dir $RUNS_DIR/event_cv_terramind_tiny_dem \
  --candidates terramind_v1_tiny \
  --modality-sets vv+vh+dem \
  --folds 5 \
  --epochs 12 \
  --patience 3 \
  --batch-size 1 \
  --num-workers 2 \
  --encoder-lr 0.00001 \
  --decoder-lr 0.0001 \
  --weight-decay 0.0001 \
  --optimizer adamw \
  --scheduler poly \
  --loss bce_tversky \
  --loss-alpha 0.3 \
  --loss-beta 0.7 \
  --bce-weight 0.5 \
  --tversky-weight 0.5 \
  --no-amp \
  --gpu \
  --thresholds 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 \
  --skip-completed \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-file $RUNS_DIR/event_cv_terramind_tiny_dem/output.log
```

Re-running the same command skips only folds with a validated successful `cv_result.json` completion marker and resumes an interrupted fold from its own `models/last.ckpt`. An interruption stops the matrix immediately and does not create a completion marker. Compatibility handling also detects and repairs stale interrupted markers written by older releases. The output directory is protected against changes to the fold or normalisation plan.

The learning-rate, optimiser, scheduler and loss overrides are applied identically to every selected fold. Provider-specific input normalisation and provider-safe augmentation remain registry-controlled.

## Out-of-fold threshold calibration

```bash
floodmap calibrate-event-cv \
  --cv-dir $RUNS_DIR/event_cv_terramind_tiny_dem \
  --processed-data-dir $PROCESSED_DIR \
  --output-dir $RUNS_DIR/calibration_terramind_tiny_dem \
  --candidate terramind_v1_tiny \
  --modalities vv+vh+dem \
  --thresholds 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 \
  --batch-size 1 \
  --num-workers 2 \
  --no-amp \
  --gpu \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-file $RUNS_DIR/calibration_terramind_tiny_dem/output.log
```

Only after this comparison should the winning candidate be trained on the full training split and evaluated once on the test split.
