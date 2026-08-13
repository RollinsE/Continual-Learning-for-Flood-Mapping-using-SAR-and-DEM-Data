# Flood Extent Mapping

**Release 0.15.20**

Flood Extent Mapping is a modular Python pipeline for training, evaluating and deploying flood-segmentation models using Sentinel-1 SAR and DEM data. It supports standard training and continual-learning experiments, event-level cross-validation, several model and pretraining options, detailed error analysis, portable deployment bundles and a Streamlit app for running predictions on new scenes.

The reference deployment uses Sentinel-1 VV and VH. DEM and derived terrain features are supported where the selected model requires them, so the package is not tied to a single input combination.

The project runs through the `floodmap` command-line interface and is designed for normal Python environments as well as notebooks and servers. Google Colab is supported, but the package does not depend on Colab-specific paths or notebook code.

## Dataset and attribution

The reference experiments use the **MMFlood dataset**. MMFlood is the dataset name, not the name of this application. Dataset-specific configuration and mask conventions therefore still use `mmflood` where that is technically accurate, for example `configs/preprocess_mmflood.yaml`.

This repository includes and extends code from the original MMFlood project by Fabio Montello, Edoardo Arnaudo and Claudio Rossi (LINKS Foundation), released under the MIT licence. The dataset paper is *MMFlood: A Multimodal Dataset for Flood Delineation From Satellite Imagery*, IEEE Access 10 (2022), 96774–96787, DOI `10.1109/ACCESS.2022.3205419`.

## Reference model

The current reference deployment is a U-Net with an ImageNet-pretrained ResNet34 encoder and Sentinel-1 `VV + VH` inputs.

| Item | Reference value |
|---|---:|
| Training tiles | 3,981 |
| Validation tiles | 324 |
| Test tiles | 386 |
| Selected epoch | 4 |
| Operating threshold | 0.80 |
| Test F1 | **0.5471** |
| Test IoU | 0.3765 |
| Test precision | 0.4528 |
| Test recall | 0.6909 |
| Test MCC | 0.5358 |

These numbers describe the retained test split for this model. They are not a general estimate of performance on every flood event or location. See [`MODEL_CARD.md`](MODEL_CARD.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the evaluation setup and limitations.

## What the package supports

- Sentinel-1 and DEM preprocessing with train-fitted normalisation.
- U-Net, U-Net++, DeepLabV3+, SegFormer and registered foundation/pretrained backbones.
- ImageNet, SSL4EO/DeCUR, FG-MAE SAR, CROMA and TerraMind weights where supported.
- Five-fold event-level cross-validation and out-of-fold threshold calibration.
- Standard ERM, continual learning, GroupDRO, hard-example and region-guided training.
- F1, IoU, precision, recall and MCC evaluation, plus tile- and event-level error audits.
- Portable deployment bundles containing the model config, checkpoint and normalisation assets needed for inference.
- Single-tile and multi-tile prediction through the CLI or Streamlit app.
- Automatic SAR band inspection, including separate VV/VH files when they can be identified safely.
- DEM and optional ground-truth-mask matching by tile identifier rather than upload order.
- Post-inference whole-area mosaics on a common equal-area grid, without resampling the native SAR before inference.
- Area-based multi-tile summaries and labelled evaluation on the same output grid when masks are supplied.

## Repository layout

```text
configs/                 Preprocessing and training configurations
floods/                  Python package and CLI implementation
docs/                    Data, deployment, validation and reproducibility guides
reproducibility/         Machine-readable reference metrics
scripts/                 Installation, smoke-test and environment helpers
streamlit_app/           Streamlit prediction interface
tests/                   Automated regression tests
data/                    Local data placeholder; contents ignored by git
outputs/                 Local output placeholder; contents ignored by git
```

Raw datasets, processed tiles, checkpoints and local experiment outputs are excluded from normal Git tracking. Public Streamlit deployment checkpoints should be stored with Git LFS.

## Installation

### Standard Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .

floodmap doctor --strict
python -m pytest -q
```

### Google Colab

From the repository root in a fresh runtime:

```bash
bash scripts/install_colab.sh
```

The installer keeps Colab's CUDA-compatible PyTorch build where possible, installs the tested dependency stack, runs the environment doctor and executes the package checks. See [`docs/COLAB.md`](docs/COLAB.md).

## CLI

The main commands are:

```bash
floodmap --help
floodmap preprocess --help
floodmap train --help
floodmap model-catalog
floodmap event-cv --help
floodmap calibrate-event-cv --help
floodmap evaluate --help
floodmap export-deployment --help
floodmap predict-scene --help
```

Long-running commands can write a persistent log while showing progress in the terminal:

```text
--log-file PATH
--log-level INFO
--plain-progress
--heartbeat-seconds 30
```

## Model selection workflow

The shared training defaults live in `configs/training_defaults.yaml`. Architectures, pretrained weights and input modalities are selected through the CLI, so users do not need to edit the package to switch between supported model families.

A typical model-selection workflow is:

1. Preprocess and audit the data.
2. Run event-level cross-validation on the training events.
3. Calibrate the decision threshold from held-out fold predictions.
4. Choose the architecture, inputs, training rule and threshold.
5. Train the selected final model.
6. Evaluate the held-out test split once.
7. Export the selected checkpoint as a portable deployment bundle.

See [`docs/EVENT_CV.md`](docs/EVENT_CV.md), [`docs/PRETRAINED_MODELS.md`](docs/PRETRAINED_MODELS.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Portable deployment

A trained model can be exported into a self-contained deployment directory:

```bash
floodmap export-deployment \
  --output-file /path/to/deployment/deployment_manifest.yaml \
  --model-name unet_resnet34_vv_vh \
  --config /path/to/run/config.yaml \
  --checkpoint /path/to/run/models/best.pth \
  --threshold 0.80 \
  --input-modalities vv vh
```

The manifest uses paths relative to the deployment bundle. Moving the complete bundle to another machine therefore does not require rewriting local training paths.

Run inference with:

```bash
floodmap predict-scene \
  --manifest /path/to/deployment/deployment_manifest.yaml \
  --sar-path /path/to/sar.tif \
  --output-dir /path/to/prediction \
  --write-probability \
  --write-html-report
```

## Streamlit app

The app in `streamlit_app/` uses the same `floods.deployment.predict_scene()` function as the CLI. It adds file upload, automatic tile matching and result display; it does not maintain a separate inference implementation.

It can process one SAR tile or a collection of tiles, recognise supported SAR layouts, match DEM and optional masks by tile ID, show per-tile outputs and build a whole-area mosaic after inference.

```bash
python -m pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

See [`streamlit_app/README.md`](streamlit_app/README.md) for model-bundle staging and Streamlit Community Cloud deployment.

## Reproducibility records

For a published experiment, keep at least:

- the resolved `config.yaml`;
- the exact CLI command;
- the persistent `output.log`;
- the selected checkpoint and `last.ckpt` when enabled;
- train-fitted normalisation statistics;
- event-CV plan and fold manifests;
- threshold-calibration outputs;
- environment capture from `scripts/capture_environment.py`;
- fixed dataset and split metadata.

## Checks

```bash
python -m compileall -q floods streamlit_app scripts tests
python -m pytest -q
python scripts/runtime_smoke.py
floodmap audit-code --project-root . --output-dir outputs/code_audit --plain-progress
```

GitHub Actions runs compile and regression checks on pushes and pull requests to `main`.

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md) — reference model, performance and limitations.
- [`docs/DATA.md`](docs/DATA.md) — dataset structure and preprocessing assumptions.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — CLI deployment and portable bundles.
- [`docs/EVENT_CV.md`](docs/EVENT_CV.md) — event-level validation.
- [`docs/PRETRAINED_MODELS.md`](docs/PRETRAINED_MODELS.md) — registered weight sources.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — reference experiment and environment records.
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`NOTICE.md`](NOTICE.md).

## Licence

MIT. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Dataset and pretrained-model licences remain with their respective providers.
