# Validation checklist

Use this checklist before long training runs.

1. `python -m compileall -q floods scripts tests`
2. `python -m pytest -q`
3. `floodmap audit-code --project-root "$PROJECT_ROOT" --output-dir "$ARTIFACTS_DIR/audits/code_quality"`
4. `floodmap audit-raw-alignment --raw-data-dir "$RAW_DIR" --output-dir "$ARTIFACTS_DIR/audits/raw_alignment"`
5. `floodmap audit-dataset --processed-data-dir "$PROCESSED_DIR" --output-dir "$ARTIFACTS_DIR/audits/processed_dataset" --splits train val test`
6. `floodmap fit-normalization --processed-data-dir "$PROCESSED_DIR" --output-file "$STATS_FILE" --split train --input-modalities vv vh dem`
7. Confirm preprocessing tile size, training image size, and evaluation window size are intentionally matched.
8. Run a short training smoke test before any full training run.

The package does not ship MMFlood data, split metadata, trained checkpoints, or normalization JSON files. Those are experiment artifacts and should live outside the repository.

## Training exposure audit

Use the saved training configuration to simulate the exact configured sampler without running model optimisation:

```bash
floodmap audit-training-exposure \
  --config /path/to/run/config.yaml \
  --processed-data-dir $PROCESSED_DIR \
  --output-dir $ARTIFACTS_DIR/training_exposure_audit \
  --epochs 20 \
  --negative-max-ratio 0.001 \
  --negative-clusters 8 \
  --plain-progress
```

The audit writes per-epoch, per-batch, per-tile, negative-descriptor and negative-cluster CSV files plus a compact JSON summary. It recreates the configured sampler, including weighted, stratified, event-balanced, foreground-balanced and hard-example modes.
