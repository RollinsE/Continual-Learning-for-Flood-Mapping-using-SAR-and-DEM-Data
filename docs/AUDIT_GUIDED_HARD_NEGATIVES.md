# Audit-guided hard-negative region fine-tuning

The pipeline includes region-level hard-negative mining. The miner runs the current
best model on the labelled **training split**, locates high-confidence pixels
that the model wrongly labels as flood, and writes crop coordinates to
`hard_negative_regions.csv`. Training then crops directly around those actual
errors instead of searching for random background windows.

Do not mine from validation data for model development. The validation split is
reserved for checkpoint selection and comparison.

## 1. Mine baseline false-positive regions from the training split

```bash
floodmap mine-hard-negatives \
  --config $ARTIFACTS_DIR/flood_baseline_512_20260713_194723/config.yaml \
  --checkpoint $ARTIFACTS_DIR/flood_baseline_512_20260713_194723/models/model-020_best_f1-0.4941.pth \
  --processed-data-dir $PROCESSED_DIR \
  --output-dir $ARTIFACTS_DIR/baseline_train_hard_negative_regions_v084 \
  --split train \
  --image-size 512 \
  --batch-size 8 \
  --num-workers 2 \
  --input-modalities vv vh dem \
  --threshold 0.60 \
  --crop-sizes 256 320 384 \
  --min-component-area 64 \
  --min-fp-pixels 128 \
  --max-label-fg-ratio 0.001 \
  --min-valid-ratio 0.50 \
  --max-regions-per-tile 3 \
  --nms-iou 0.30 \
  --no-pretrained \
  --gpu \
  --plain-progress
```

The command writes:

- `hard_negative_regions.csv`: exact tile and crop coordinates;
- `summary.json`: mining settings and region counts.

## 2. Fine-tune from the baseline checkpoint

This is a fresh fine-tune. `--init-checkpoint` loads model weights only and
resets the optimiser, scheduler, epoch counter, and early-stopping state.

```bash
floodmap train \
  --config $ARTIFACTS_DIR/flood_baseline_512_20260713_194723/config.yaml \
  --processed-data-dir $PROCESSED_DIR \
  --artifacts-dir $ARTIFACTS_DIR \
  --run-id flood_baseline_hard_negative_regions_v084 \
  --init-checkpoint $ARTIFACTS_DIR/flood_baseline_512_20260713_194723/models/model-020_best_f1-0.4941.pth \
  --no-resume \
  --no-pretrained \
  --hard-negative-region-sampling \
  --hard-negative-manifest $ARTIFACTS_DIR/baseline_train_hard_negative_regions_v084/hard_negative_regions.csv \
  --hard-negative-region-weight 1.25 \
  --hard-negative-region-max-fraction 0.42 \
  --hard-negative-crop-probability 0.50 \
  --weighted-samples-multiplier 1.0 \
  --no-sparse-crop-supervision \
  --epochs 8 \
  --patience 4 \
  --lr 0.00001 \
  --encoder-lr 0.000002 \
  --decoder-lr 0.00001 \
  --threshold-sweep \
  --threshold-metric f1 \
  --monitor-threshold-sweep \
  --save-last \
  --save-epoch-checkpoints \
  --gpu \
  --progress \
  --plain-progress \
  --num-workers 2
```

The controlled objective is to reduce false-positive pixels and poor-overlap
errors without losing the baseline's flood recall. The benchmark remains
validation F1 `0.4941`; component filtering at threshold `0.50` and minimum area
`64` is evaluated after training, not baked into the loss.
