# Continual-learning workflow

`continual-train` runs chronological rehearsal experiments on the processed MMFlood dataset. EMSR events are grouped into tasks by the year of their start date (default) or by geographic region (see [Task splits](#task-splits)), and the model is trained on the tasks in order. It uses the same model, preprocessing, loss, normalisation, ignore-label handling and global threshold-sweep metrics as `train` and `evaluate`.

## Strategies

| Strategy | Replay | Purpose |
|---|---|---|
| `finetune` | none | Sequential training without memory. Lower bound; shows how much is forgotten. |
| `random` | uniform from a reservoir buffer | Standard rehearsal baseline. |
| `least_confidence` | sampled ∝ mean pixel `1 − max(p, 1 − p)` | Uncertainty-guided rehearsal. |
| `entropy` | sampled ∝ mean pixel binary entropy | Uncertainty-guided rehearsal. |
| `hybrid` | half ∝ entropy, half uniform | Uncertainty with diversity. |
| `joint` | none; trains on all tasks seen so far | Upper bound (cumulative training). |

`margin` is also accepted, but for a single sigmoid output it equals 2 × `least_confidence` and selects exactly the same samples, so it should not be reported as a separate strategy.

The replay buffer is a reservoir over all training tiles of completed tasks (`--replay-buffer-size`). Each current-task batch is paired with `--replay-batch-size` replay tiles; with `--replay-mode separate` (default) the two losses are averaged with sample-count weights.

Uncertainty scores are computed **once per epoch**, at the start of the epoch, on non-augmented copies of the buffer tiles, averaging over valid (non-ignore) pixels only. Replay tiles are then drawn **in proportion to their score** for every batch, so uncertain tiles are favoured without replaying the same few tiles all epoch. `--uncertainty-sampling topk` restores deterministic top-k selection; `--uncertainty-subset-fraction` scores a random fraction of the buffer to save time on large buffers.

## Evaluation protocol

After each task *t* the model is evaluated on every task seen so far (1…*t*). This produces the task matrix in `cl_results.json`.

- **Threshold:** one threshold per evaluation row. It is chosen on the **validation** tiles of tasks 1…*t* using `--threshold-metric`, then applied unchanged to every cell of that row, including test cells. `--cl-threshold 0.5` fixes it instead (useful for a locked operating point). The per-cell best threshold is stored as `oracle_best` for diagnostics only; never report it as a result.
- **No checkpoint selection across tasks:** the model at the end of each task is saved (`models/after_task_<t>.pth`) and evaluated. Per-epoch validation is logged on seen-task validation tiles only; it never looks at future tasks.
- **Summary (`cl_summary`)**, all at the selected threshold:
  - `mean_final_task_score` — mean score over all tasks after the last task (average accuracy);
  - `mean_plasticity` — mean score on each task right after learning it;
  - `mean_forgetting` / `max_forgetting` — best-seen minus final score, over all but the last task;
  - `backward_transfer` — final minus just-learned score, over all but the last task (negative = forgetting).
- When more than one strategy is run, `<run-id>_cl_comparison.csv` lists these values side by side.

Every strategy starts from the same seed. The positive-class weight (`--pos-weight-from-train`) is computed once from the full training split and shared by all strategies.

## Outputs

Each strategy writes `<output-folder>/<run-id>_<strategy>/`:

- `config.yaml` — resolved configuration;
- `output.log` — training and evaluation log;
- `models/after_task_<t>.pth` — model after each task;
- `models/<strategy>_final.pth` — model after the last task (`<strategy>_best.pth` is the same file, kept for older commands);
- `models/last.ckpt` — resumable state (model, optimiser, scheduler, AMP scaler, replay buffer, RNG state, progress);
- `cl_results.json` — protocol, task definitions, threshold selection, validation history, task matrix and summary.

## Example command

```bash
floodmap continual-train \
  --config configs/train_segmentation_vv_vh_dem.yaml \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-folder "$ARTIFACTS_DIR" \
  --run-id flood_512_cl \
  --activations-json-path "$SUMMARY_FILE" \
  --strategies finetune random least_confidence entropy hybrid joint \
  --task-year-ranges 2014-2017 2018-2019 2020-2021 \
  --epochs-per-task 5 \
  --replay-buffer-size 100 \
  --replay-batch-size 16 \
  --architecture unet \
  --backbone resnet34 \
  --pretrained \
  --image-size 512 \
  --batch-size 8 \
  --num-workers 2 \
  --input-modalities vv vh dem \
  --normalization-mode robust_minmax \
  --normalization-stats-path "$STATS_FILE" \
  --loss bce_tversky \
  --pos-weight-from-train \
  --augmentation-profile composite \
  --threshold-sweep \
  --threshold-metric f1 \
  --cl-eval-split test \
  --cl-eval-inference-mode sliding_window \
  --window-size 512 \
  --window-overlap 128 \
  --gpu --amp \
  --plain-progress
```

`--cl-eval-split val` (default) is faster and is the right choice while developing; use `test` with `sliding_window` once for the final comparison. `joint` trains on all seen data, so its later tasks take longer than the other strategies.

## Task splits

- **By year** (default): `--task-split year --task-year-ranges 2014-2017 2018-2019 2020-2021`, from the event start
  date in `activations.json`.
- **By region**: `--task-split region --task-regions europe asia+americas+africa+oceania`, from the event country
  (or its lat/lon when the country is missing). Regions are `europe`, `asia`, `africa`, `americas`, `oceania`; join
  them with `+` to merge small regions into one task.

Add `--cl-dry-run` to print each task's tiles, events and countries without training. On MMFlood, Asia alone has
54 training tiles and one test event, which is too small to measure forgetting, so it is merged into the second task.

## Results on MMFlood

U-Net ResNet34, VV+VH, 5 epochs per task, replay batch 8, fixed threshold 0.80, test split. "Final F1" is the mean
over tasks after the last task; "forgetting" is the drop on earlier tasks.

**Region split** (Europe → rest of world), buffer 100, seeds 42, 1 and 2 (mean ± std):

| strategy | final F1 | plasticity | forgetting |
|---|---|---|---|
| finetune | 0.491 ± 0.009 | 0.505 ± 0.003 | 0.028 ± 0.015 |
| hybrid | 0.504 ± 0.019 | 0.507 ± 0.018 | 0.007 ± 0.009 |
| joint | 0.536 ± 0.005 | 0.533 ± 0.002 | 0.003 ± 0.005 |

Hybrid rehearsal had lower forgetting and a higher final F1 than fine-tuning in each of the three seeds; with three
seeds the forgetting difference is suggestive rather than statistically significant. Joint training was best in every
seed, mainly through better adaptation to the second task. Random and entropy replay (seed 42 only) reached 0.506 and
0.505 final F1 with forgetting 0.025 and 0.013.

**Buffer size** (hybrid, region split, seed 42): 50 tiles → 0.533 final F1, forgetting 0.003; 100 → 0.523, 0.000;
150 → 0.535, 0.000. Buffer size had no consistent effect in this range; the differences are smaller than the
variation between seeds.

**Year split** (2014–17, 2018–19, 2020–21), buffer 100, seed 42:

| strategy | final F1 | forgetting | backward transfer |
|---|---|---|---|
| finetune | 0.460 | 0.009 | 0.057 |
| random | 0.441 | 0.015 | 0.045 |
| entropy | 0.451 | 0.003 | 0.056 |
| hybrid | 0.446 | 0.003 | 0.059 |
| joint | 0.449 | 0.000 | 0.051 |

Forgetting across years is negligible for every strategy, and later tasks improve earlier ones (positive backward
transfer). Changes of about ±0.01 occur even for joint training, so differences of that size are noise.

`scripts/aggregate_cl_runs.py` combines repeated runs into mean ± std tables.

`scripts/analyse_cl_runs.py` analyses finished runs without retraining. `metrics` lists precision, recall and F1 per
strategy and task (right after training on the task and after the last task, so forgetting can be split into lost recall
and lost precision) and the training time of each strategy. `replay` rebuilds each replay buffer exactly, scores the
buffered tiles with the models saved at the start and end of each task, and reports how much flood each strategy
replays compared with uniform sampling.

## Resuming

`--resume` continues every strategy (and ensemble member) from its own `models/last.ckpt`, including a task whose training finished but whose evaluation did not. A finished strategy is skipped, and a strategy with no checkpoint yet starts from the beginning, so it is safe to always pass `--resume`. `--resume-from PATH` loads one specific checkpoint and is only allowed with a single strategy in single-model mode; a bare `.pth` also needs `--resume-start-task`.

## Ensembles

`--cl-model-mode ensemble --ensemble-members unet:resnet34 deeplabv3p:resnet34` trains one model per member for each strategy, then evaluates the members **together** after each task (`--ensemble-method mean_logit|mean_prob`) with the same threshold protocol. Results are in `<run-id>_<strategy>_ensemble/cl_ensemble_results.json`.

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
  --normalization-mode robust_minmax \
  --normalization-stats-path "$STATS_FILE" \
  --metric-mode global \
  --threshold-metric f1 \
  --no-pretrained --no-amp --gpu \
  --plain-progress
```

## Deployment manifest

Use `export-deployment` to freeze the selected operating point into a portable deployment bundle. The command copies the configs, checkpoints and normalization statistics beside a relative-path manifest and records the threshold, component filter, modalities and inference-window settings.
