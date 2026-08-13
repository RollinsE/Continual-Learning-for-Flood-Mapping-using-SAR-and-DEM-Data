# Flood Extent Mapping — Streamlit interface

This directory contains the Streamlit interface for deployed flood-mapping models. The app calls `floods.deployment.predict_scene()` directly, so CLI and Streamlit predictions use the same inference code.

## Model catalogue

Stage one or more complete portable bundles created by `floodmap export-deployment` under:

```text
streamlit_app/deployments/
├── unet_resnet34_vv_vh/
│   ├── deployment_manifest.yaml
│   └── assets/...
└── another_model/
    ├── deployment_manifest.yaml
    └── assets/...
```

If more than one valid manifest is present, the app displays a model selector. The selected manifest controls required modalities, threshold, window size and model assets. A legacy single bundle under `streamlit_app/deployment/` is still discovered.

For a custom location, set:

```bash
export FLOODMAP_DEPLOYMENT_MANIFEST=/absolute/path/to/deployment_manifest.yaml
```

## Input behaviour

Users can upload the SAR files they have without first checking whether each file is single-band or combined.

- Two-band/multiband SAR files are recognised automatically.
- Separate single-band files are paired as VV/VH when filename or GeoTIFF metadata identifies the polarization safely.
- Multiple spatial SAR tiles are processed independently by the deployment model.
- DEM files, when required by the selected model, are matched to SAR by canonical tile ID.
- Ground-truth masks are optional and matched the same way. Missing masks never prevent prediction; they only reduce the number of tiles included in evaluation.
- DEM/VH grids that differ from the SAR reference grid are geospatially reprojected onto the SAR grid rather than resized by array shape alone.

For multi-tile runs, the native SAR grid is left unchanged during inference. Prediction outputs are reprojected afterwards to an equal-area collection grid so the app can report a whole-area flood mask, probability raster, mapped area and flood coverage even when source tiles have different resolutions. If masks are supplied, multi-tile evaluation is calculated on the same grid so a finer-resolution source tile does not receive more weight simply because it contains more pixels.

## Git LFS

Deployment checkpoints can exceed GitHub's ordinary file-size limit. Install Git LFS before committing public model bundles:

```bash
git lfs install
git add .gitattributes
git add streamlit_app/deployments

git lfs ls-files
```

Confirm the intended checkpoint appears in `git lfs ls-files` before pushing. Do not commit raw training data, processed datasets, experiment directories, or private credentials.

## Local run

From the repository root:

```bash
python -m pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## Streamlit Community Cloud

Use:

```text
Entry point: streamlit_app/app.py
Python:      3.12
```

Community Cloud installs `streamlit_app/requirements.txt`. Inference runs on CPU unless the hosting environment provides and is configured for another device. Generated prediction files are temporary for each app session, so users should download outputs they need to retain.
