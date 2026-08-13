#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${FLOODMAP_REPO_DIR:-$(pwd)}"
INSTALL_LOG="${FLOODMAP_INSTALL_LOG:-${REPO_DIR}/install.log}"

cd "$REPO_DIR"

{
  echo "Flood Extent Mapping Colab installation"
  echo "Repository: $REPO_DIR"
  echo "Installing the tested NumPy/SciPy/Rasterio binary stack first..."
} | tee "$INSTALL_LOG"

# Reinstall the binary stack together so a partially upgraded NumPy cannot be
# paired with stale SciPy/Rasterio extension modules.
python -m pip install \
  --no-cache-dir \
  --force-reinstall \
  -c constraints-colab.txt \
  numpy scipy rasterio 2>&1 | tee -a "$INSTALL_LOG"

python -m pip install \
  --no-cache-dir \
  -r requirements-colab.txt 2>&1 | tee -a "$INSTALL_LOG"

python -m pip install --no-deps --no-build-isolation -e . 2>&1 | tee -a "$INSTALL_LOG"
python -m compileall -q floods scripts tests
python -m floods.cli doctor --strict --json-output environment_report.json 2>&1 | tee -a "$INSTALL_LOG"
python -m pytest -q 2>&1 | tee -a "$INSTALL_LOG"

python - <<'PY'
import torch
import floods
print("floodmap", floods.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

echo "Install log: $INSTALL_LOG"
echo "Environment report: ${REPO_DIR}/environment_report.json"
echo "If NumPy/SciPy were imported in the current notebook before this script ran, restart the Colab session now."
