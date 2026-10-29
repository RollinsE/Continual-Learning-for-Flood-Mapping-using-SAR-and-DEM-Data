# Results

Summary tables from the experiments described in the main README and in `docs/CONTINUAL_LEARNING.md`. Model checkpoints,
predictions and per-tile outputs are not included; they are too large for the repository.

All runs used the U-Net ResNet34 model with VV+VH input, the processed MMFlood data (3,980 training tiles, 324 validation
tiles, 386 test scenes) and a fixed decision threshold of 0.80. Continual-learning scores are F1 on the test scenes of
each task. "Final F1" is the mean over tasks after the last task, "plasticity" is the mean F1 on each task right after
training on it, "forgetting" is how much F1 on earlier tasks dropped by the end, and "BWT" (backward transfer) is the
final minus the original score on earlier tasks.

## Files

**`cl_region_split_3_seeds.csv`**: Europe first, then the rest of the world, with a 100-tile replay buffer. Fine-tuning,
hybrid replay and joint training were run with seeds 42, 1 and 2; random and entropy replay with seed 42 only (`seeds`
column). Values are mean and standard deviation over the seeds. Made with `scripts/aggregate_cl_runs.py`.

**`cl_year_split_seed42.csv`**: tasks by event year (2014–17, 2018–19, 2020–21), seed 42, 100-tile buffer. This is the
comparison CSV written by `floodmap continual-train`.

**`cl_buffer_size_hybrid_seed42.csv`**: hybrid replay on the region split with buffers of 50, 100 and 150 tiles,
seed 42. Worked out from `cl_per_task_metrics.csv`; the values match the training logs.

**`cl_per_task_metrics.csv`**: precision, recall and F1 for every run, strategy and task, both right after training on
the task (`_own`) and after the last task (`_final`). The `recall_change` and `precision_change` columns show whether
forgetting came from missed flood or from false flood. Made with `scripts/analyse_cl_runs.py metrics`.

**`cl_replay_composition.csv`**: what each replay strategy actually replayed. The replay buffer was rebuilt exactly as
in training and scored with the model saved at the start (`start`) and end (`end`) of the task in which it was replayed.
`flood_oversampling` is the flood-pixel content of a replay draw relative to uniform sampling; `effective_tiles` is how
many of the buffered tiles the sampler effectively uses. The recomputed uncertainty matched the value logged during
training in 11 of the 12 checks (`mean_uncertainty` against `logged_epoch1_mean`). Made with
`scripts/analyse_cl_runs.py replay`.

**`cl_training_time.csv`**: training time for runs that finished in one Colab session on the same GPU. Times from
different sessions are not comparable, so the year-split runs, which were resumed several times, are left out.

**`event_cv_model_selection.csv`**: event-macro F1 on held-out training events (event-level cross-validation, folds 0
and 1) for the model and loss options that were compared. A change was only adopted if it beat the first row on both
folds by about 0.02; none did. Empty cells are folds that were not run.

**`water_mask_validation.csv`**: validation-split scores with and without the permanent-water mask at three model
thresholds, from `floodmap audit-water-prior`. The 50% setting at threshold 0.80 was then applied once to the test
split, where F1 went from 0.520 to 0.549.

## Main findings

- Across years there was almost no forgetting, with or without replay.
- Across regions, fine-tuning without replay forgot 0.028 ± 0.015 F1 on Europe. Hybrid replay cut this to
  0.007 ± 0.009 and did better than fine-tuning in all three seeds, but with three seeds the difference is not
  statistically significant. Training on all data at once was the most accurate (0.536 ± 0.005).
- Uncertainty-based replay drew 1.1–1.6 times as many flood pixels as random replay, but this did not consistently
  improve final accuracy over random replay.
- Buffer size between 50 and 150 tiles made no consistent difference.
- None of the model or loss changes beat the reference model in cross-validation. Masking permanent water was the only
  change that clearly helped.
