# Reproducing the locked reference workflow

This document separates the reproducible reference protocol from exploratory and diagnostic commands. Do not alter the test split, model, threshold, or post-processing after viewing the test result.

## 1. Define paths

```bash
export PROJECT_ROOT="$(pwd)"
export RAW_DIR="<raw-mmflood-root>"
export PROCESSED_DIR="<processed-tiles-dir>"
export SUMMARY_FILE="<split-metadata-json>"
export ARTIFACTS_DIR="<runs-output-dir>"
```

The expected processed layout is documented in [`DATA.md`](DATA.md).

## 2. Install and record the environment

```bash
python -m pip install -r requirements-colab.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m compileall -q floods scripts tests
python -m pytest -q
python scripts/capture_environment.py --output "$ARTIFACTS_DIR/environment.json"
```

Use `requirements.txt` instead of `requirements-colab.txt` outside Colab.

## 3. Preprocess at 512 pixels

```bash
floodmap preprocess \
  --config "$PROJECT_ROOT/configs/reproduction/preprocess_512.yaml" \
  --raw-data-dir "$RAW_DIR" \
  --processed-data-dir "$PROCESSED_DIR" \
  --summary-file "$SUMMARY_FILE" \
  --subset train val test \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-file "$PROCESSED_DIR/preprocess.log"
```

Reference split counts after preprocessing were 3,981 training tiles, 324 validation tiles, and 386 test tiles. A different count means the data release, split metadata, preprocessing inputs, or filtering differ from the reference run.

## 4. Audit the processed dataset

```bash
floodmap audit-dataset \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-dir "$ARTIFACTS_DIR/processed_dataset_audit" \
  --splits train val test \
  --plain-progress
```

Resolve missing pairs, shape mismatches, invalid masks, or unexpected event assignments before training.

## 5. Event-level cross-validation

The model-development split is the processed `train` split only. Entire events are held out in each fold, and normalisation is refitted on the fold's training events.

```bash
floodmap event-cv \
  --config "$PROJECT_ROOT/configs/reproduction/final_unet_resnet34_vv_vh.yaml" \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-dir "$ARTIFACTS_DIR/event_cv" \
  --candidates imagenet:unet:resnet34:reference_unet \
  --modality-sets vv+vh \
  --folds 5 \
  --epochs 12 \
  --patience 3 \
  --batch-size 8 \
  --num-workers 2 \
  --pretrained \
  --no-amp \
  --gpu \
  --thresholds 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
  --skip-completed \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-file "$ARTIFACTS_DIR/event_cv/output.log"
```

The architecture and modality decision must be based on mean event-macro F1, fold variability, and worst-event F1 rather than one favourable fold.

## 6. Calibrate the operating threshold

```bash
floodmap calibrate-event-cv \
  --cv-dir "$ARTIFACTS_DIR/event_cv" \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-dir "$ARTIFACTS_DIR/event_cv_calibration" \
  --candidate reference_unet \
  --modalities vv+vh \
  --thresholds 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
  --no-amp \
  --gpu \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-file "$ARTIFACTS_DIR/event_cv_calibration/output.log"
```

The locked reference threshold is `0.80`. Its out-of-fold event-macro F1 was approximately `0.4200` across 54 training events.

## 7. Train the final model

```bash
floodmap train \
  --config "$PROJECT_ROOT/configs/reproduction/final_unet_resnet34_vv_vh.yaml" \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-folder "$ARTIFACTS_DIR" \
  --run-id final_unet_resnet34_vv_vh \
  --encoder resnet34 \
  --decoder unet \
  --input-modalities vv vh \
  --epochs 6 \
  --patience 99 \
  --batch-size 8 \
  --num-workers 2 \
  --encoder-lr 0.0001 \
  --decoder-lr 0.0001 \
  --seed 42 \
  --pretrained \
  --no-amp \
  --gpu \
  --train-mask-body-ratio 0.0 \
  --val-mask-body-ratio 0.0 \
  --no-weighted-sampling \
  --no-foreground-balanced-sampling \
  --no-stratified-sampling \
  --no-event-balanced-sampling \
  --no-group-dro \
  --no-hard-example-sampling \
  --no-hard-positive-region-sampling \
  --no-hard-negative-region-sampling \
  --no-sparse-crop-supervision \
  --no-modality-dropout \
  --augmentation-profile composite \
  --threshold-sweep \
  --thresholds 0.80 \
  --threshold-metric f1 \
  --monitor-threshold-sweep \
  --save-last \
  --no-save-epoch-checkpoints \
  --no-visualize \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-file "$ARTIFACTS_DIR/final_unet_resnet34_vv_vh/output.log"
```

The reference run selected the epoch-4 checkpoint with validation F1 `0.4596` at threshold `0.80`. A new run may select another epoch; use its highest locked-threshold validation checkpoint rather than copying the historical filename.

For a genuine interruption, rerun the same command with `--resume` only after the run directory contains `config.yaml` and `models/last.ckpt`.

## 8. Evaluate the test split once

Use batch size `1` to avoid GPU-memory fragmentation on a 15 GB T4. Batch size does not change predictions in evaluation mode.

```bash
floodmap evaluate \
  --config "$ARTIFACTS_DIR/final_unet_resnet34_vv_vh/config.yaml" \
  --checkpoint "<best-validation-checkpoint>" \
  --processed-data-dir "$PROCESSED_DIR" \
  --split test \
  --batch-size 1 \
  --num-workers 2 \
  --input-modalities vv vh \
  --no-pretrained \
  --no-amp \
  --gpu \
  --thresholds 0.80 \
  --threshold-metric f1 \
  --metric-mode global \
  --plain-progress \
  --heartbeat-seconds 30 \
  --log-file "$ARTIFACTS_DIR/final_unet_resnet34_vv_vh/test_evaluation_t080.log"
```

Reference test metrics:

| Metric | Value |
|---|---:|
| F1 | 0.5471 |
| IoU | 0.3765 |
| Precision | 0.4528 |
| Recall | 0.6909 |
| MCC | 0.5358 |
| Empty-tile false-positive rate | 0.0000 |
| Non-empty tile recall | 0.9870 |

## 9. What must be reported

Report the data release and split manifest, tile counts, code version, resolved configuration, environment record, random seed, selected checkpoint, threshold, metric mode, and all test metrics. Do not report the test result as an event-macro estimate unless event-level test aggregation was explicitly run.
