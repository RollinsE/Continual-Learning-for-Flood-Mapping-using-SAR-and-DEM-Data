# Colab usage

This document uses environment variables rather than embedded user-specific paths. Set them once at the top of the notebook.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
export PROJECT_ROOT="<repo-root>"
export RAW_DIR="<raw-mmflood-root>"
export PROCESSED_DIR="<processed-tiles-dir>"
export SUMMARY_FILE="<activations-or-split-metadata-json>"
export ARTIFACTS_DIR="<runs-and-audits-output-dir>"
export STATS_FILE="<normalization-stats-json>"
```

## Install without replacing Colab PyTorch

Always move to `/content` before deleting or replacing the repository. This prevents the notebook from being left inside a directory that no longer exists. Run the installer before importing NumPy or SciPy in notebook cells.

```bash
cd /content
rm -rf "$PROJECT_ROOT"
unzip -q /content/flood_extent_mapping_v<version>.zip -d /content
cd "$PROJECT_ROOT"
bash scripts/install_colab.sh
```

The installer preserves Colab's CUDA-enabled PyTorch package and reinstalls the tested NumPy/SciPy/Rasterio stack together, preventing stale compiled extensions from being paired with a newly upgraded NumPy. It writes `install.log` and `environment_report.json`, runs the test suite, and invokes:

```bash
floodmap doctor --strict
```

If NumPy or SciPy was imported earlier in the notebook, use **Runtime → Restart session** after installation. A notebook kernel cannot unload already-imported binary extension modules safely.

If the runtime reports `getcwd: cannot access parent directories`, execute `%cd /content` before starting another shell command. Do not reinstall PyTorch over Colab's preinstalled CUDA build.


Run the code audit:

```bash
floodmap audit-code \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$ARTIFACTS_DIR/audits/code_quality" \
  --plain-progress
cat "$ARTIFACTS_DIR/audits/code_quality/summary.json"
```


Preprocess with a runtime tile size:

```bash
floodmap preprocess \
  --config "$PROJECT_ROOT/configs/preprocess_mmflood.yaml" \
  --raw-data-dir "$RAW_DIR" \
  --processed-data-dir "$PROCESSED_DIR" \
  --summary-file "$SUMMARY_FILE" \
  --subset train val test \
  --scale 1 \
  --tile-size 256 \
  --tile-max-overlap 128 \
  --sar-transform log1p \
  --no-decibel \
  --clip-dem \
  --morphology \
  --morph-kernel 5 \
  --mask-flood-values 1 \
  --mask-background-values 0 2 \
  --mask-ignore-values 255 \
  --preserve-mask-ignore \
  --tiling \
  --no-make-context \
  --plain-progress
```

Use `--tile-size 128`, `--tile-size 256`, or `--tile-size 512`; set training `--image-size` to the same value.

Run dataset audit after preprocessing:

```bash
floodmap audit-dataset \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-dir "$ARTIFACTS_DIR/audits/processed_dataset" \
  --splits train val test \
  --plain-progress
```

Fit normalization:

```bash
floodmap fit-normalization \
  --processed-data-dir "$PROCESSED_DIR" \
  --output-file "$STATS_FILE" \
  --split train \
  --input-modalities vv vh dem \
  --q-min 1 \
  --q-max 99 \
  --max-pixels-per-file 4096 \
  --seed 42 \
  --plain-progress
```

## Live CLI output

The CLI configures stdout and stderr for immediate writes, uses the
same timestamped logger for every subcommand, and emits a command-running heartbeat
during otherwise quiet work. Run the package directly through the CLI; no Python
`subprocess.Popen`, `python -u`, or `tee` wrapper is required:

```bash
floodmap train --config /content/path/to/config.yaml --plain-progress
```

Use `--plain-progress` for Colab newline progress records. The CLI also writes
automatic file logs for runs and output directories. See `docs/LOGGING.md`.


## Direct CLI training example

The model process itself must remain a normal CLI command:

```bash
floodmap train \
  --config "$ARTIFACTS_DIR/flood_baseline_512_20260713_194723/config.yaml" \
  --processed-data-dir "$PROCESSED_DIR" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --run-id flood_groupdro_resnet50_amp_recovery_v094 \
  --init-checkpoint "$ARTIFACTS_DIR/flood_baseline_512_20260713_194723/models/model-020_best_f1-0.4941.pth" \
  --epochs 25 \
  --patience 8 \
  --batch-size 8 \
  --num-workers 2 \
  --encoder-lr 1e-5 \
  --decoder-lr 5e-5 \
  --group-dro \
  --group-dro-eta 0.01 \
  --group-dro-min-weight 0.001 \
  --group-dro-warmup-epochs 1 \
  --threshold-sweep \
  --monitor-threshold-sweep \
  --amp \
  --amp-full-precision-retry \
  --plain-progress \
  --heartbeat-seconds 30 \
  --save-last \
  --no-resume
```

The live console and `<artifacts-dir>/<run-id>/output.log` contain the same
timestamped records.
