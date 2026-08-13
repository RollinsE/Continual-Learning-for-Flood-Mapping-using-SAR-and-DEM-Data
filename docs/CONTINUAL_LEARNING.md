# Continual-learning workflow

The `continual-train` command runs chronological rehearsal experiments on the processed MMFlood dataset. It keeps the same model, preprocessing, loss, normalization, ignore-mask handling, and global threshold-sweep metrics used by the main `train` and `evaluate` commands.

## Objective

The command supports adaptive flood-mapping experiments where EMSR events are introduced as chronological tasks. It compares replay strategies against a non-CL baseline by training sequentially over year ranges and evaluating both mixed validation performance and task-by-task retention.

Implemented replay strategies:

- `random`: samples uniformly from the replay buffer.
- `least_confidence`: scores replay candidates by low binary confidence.
- `margin`: scores candidates near the binary decision boundary.
- `entropy`: scores candidates by prediction entropy.

## Outputs

Each strategy writes its own run folder. The strategy name is appended to the base run id. Each run contains:

- `config.yaml`: the exact resolved configuration.
- `output.log`: training and evaluation log.
- `models/<strategy>_best.pth`: stable alias for the best model state.
- `models/last.ckpt`: final resumable checkpoint payload for inspection.
- `cl_results.json`: validation history, task definitions, and task-by-task evaluation matrix.

## Example command

```bash
floodmap continual-train \
  --config configs/train_segmentation_vv_vh_dem.yaml \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-folder "$ARTIFACTS_DIR" \
  --run-id flood_512_r50_cl \
  --activations-json-path "$SUMMARY_FILE" \
  --strategies random least_confidence margin entropy \
  --task-year-ranges 2014-2017 2018-2019 2020-2021 \
  --epochs-per-task 5 \
  --replay-buffer-size 100 \
  --replay-batch-size 16 \
  --uncertainty-subset-fraction 1.0 \
  --architecture unet \
  --backbone resnet50 \
  --pretrained \
  --image-size 512 \
  --batch-size 8 \
  --num-workers 0 \
  --input-modalities vv vh dem \
  --normalization-mode robust_percentile \
  --normalization-stats-path "$STATS_FILE" \
  --loss bce_tversky \
  --loss-alpha 0.30 \
  --loss-beta 0.70 \
  --bce-weight 0.5 \
  --tversky-weight 0.5 \
  --pos-weight-from-train \
  --pos-weight-max 20 \
  --augmentation-profile composite \
  --threshold-sweep \
  --threshold-metric f1 \
  --metric-mode global \
  --no-amp \
  --gpu \
  --progress \
  --plain-progress
```

For full-raster task evaluation on the test split, add:

```bash
  --cl-eval-split test \
  --cl-eval-inference-mode sliding_window \
  --window-size 512 \
  --window-overlap 128 \
  --window-batch-size 1
```

Validation is the default task-evaluation split because it is faster and uses the same 512-sized tiles as training. Test split evaluation is supported but can be slower when sliding-window inference is required.

## v061 additions

`continual-train` supports two model modes:

- `--cl-model-mode single`: trains one model per replay strategy.
- `--cl-model-mode ensemble`: trains one model per listed ensemble member and then evaluates the member checkpoints as an ensemble on the validation split.

Example ensemble members use `decoder:encoder[:label]`:

```bash
--cl-model-mode ensemble \
--ensemble-members unet:resnet50 deeplabv3p:resnet50 \
--ensemble-method mean_logit
```

The CL metrics are global pixel-level threshold-sweep metrics. For every mixed-validation evaluation and task-specific evaluation, the command records F1, IoU, precision, recall, MCC, empty-tile false-positive rate and non-empty tile recall. It also writes a `cl_summary` with mean final task score, mean plasticity and mean forgetting derived from the task evaluation matrix.

## Model comparison

Use `compare-models` to compare any number of single models and ensembles. It writes:

- `comparison_summary.csv`
- `comparison_results.json`
- `comparison_best_f1.png`
- `comparison_best_iou.png`
- `comparison_best_mcc.png`
- `threshold_sweep_f1.png`

Example:

```bash
floodmap compare-models \
  --output-dir $ARTIFACTS_DIR/model_comparison \
  --model unet512 /path/unet/config.yaml /path/unet/model.pth \
  --ensemble unet_deeplab mean_logit /path/unet/config.yaml:/path/unet/model.pth /path/deeplab/config.yaml:/path/deeplab/model.pth \
  --processed-data-dir "$PROCESSED_DIR" \
  --split val \
  --input-modalities vv vh dem \
  --normalization-mode robust_percentile \
  --normalization-stats-path "$STATS_FILE" \
  --metric-mode global \
  --threshold-metric f1 \
  --no-pretrained --no-amp --gpu \
  --plain-progress
```

## Deployment manifest

Use `export-deployment` to freeze the selected operating point into a portable deployment bundle. The command copies the configs, checkpoints and normalization statistics beside a relative-path manifest and records the threshold, component filter, modalities and inference-window settings.
