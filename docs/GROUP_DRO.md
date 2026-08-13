# Event-level GroupDRO

`floodmap train --group-dro` applies distributionally robust optimisation across
training events identified from `EMSR###` in each processed mask filename.

For each minibatch, the configured segmentation loss is calculated per tile,
then averaged within every event represented in that minibatch. Event weights
are updated with exponentiated gradient ascent so events with persistently high
loss receive more influence on subsequent model updates.

The implementation:

- leaves the validation split and validation metrics unchanged;
- is compatible with existing training samplers, although a standard shuffled
  loader is recommended for the first controlled comparison;
- supports ordinary and multibranch models;
- stores event weights in `models/last.ckpt` for strict resume;
- writes `group_dro_event_weights.csv` in the run directory after each epoch.

Important options:

```text
--group-dro
--group-dro-eta 0.01
--group-dro-min-weight 0.001
--group-dro-warmup-epochs 1
```

The minimum weight is a probability floor, so it must be smaller than
`1 / number_of_training_events`.


## AMP recovery and logging

GroupDRO training remains as a direct CLI process and retries an
AMP-overflow batch once in full precision before skipping it. The option is
enabled by default and can be stated explicitly for controlled runs:

```bash
floodmap train \
  --config "<baseline-run>/config.yaml" \
  --processed-data-dir "<processed-dir>" \
  --artifacts-dir "<artifacts-dir>" \
  --run-id groupdro_amp_recovery_v094 \
  --init-checkpoint "<baseline-checkpoint.pth>" \
  --group-dro \
  --group-dro-eta 0.01 \
  --group-dro-min-weight 0.001 \
  --group-dro-warmup-epochs 1 \
  --amp \
  --amp-full-precision-retry \
  --plain-progress \
  --save-last \
  --no-resume
```

The run log records GroupDRO event weights, AMP overflows, successful and failed
float32 recoveries, genuinely skipped batches, and gradient-norm statistics.
