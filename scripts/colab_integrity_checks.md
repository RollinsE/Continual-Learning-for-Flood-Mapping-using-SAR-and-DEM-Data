# Colab integrity checks

Set these paths in your notebook/session before running the checks:

```bash
export PROJECT_ROOT="<repo-root>"
export PROCESSED_DIR="<processed-tiles-dir>"
export ARTIFACTS_DIR="<runs-and-audits-output-dir>"
```

```bash
cd "$PROJECT_ROOT"
python -m compileall -q floods scripts tests
python -m pytest -q
floodmap audit-code --project-root "$PROJECT_ROOT" --output-dir "$ARTIFACTS_DIR/audits/code_quality"
cat "$ARTIFACTS_DIR/audits/code_quality/summary.json"
floodmap audit-dataset --processed-data-dir "$PROCESSED_DIR" --output-dir "$ARTIFACTS_DIR/audits/processed_dataset" --splits train val test
```
