# Deployment inputs and scene discovery

Before the first deployment in a newly installed runtime, verify the binary stack:

```bash
floodmap doctor --strict
```

VV/VH configuration loading and prediction no longer import SciPy merely to validate modality names. SciPy is loaded only when an operation actually calculates the `dem_tpi` derived feature.

The deployment commands expect analysis-ready SAR/DEM rasters. They do not run low-level Sentinel-1 SAFE preprocessing such as orbit correction, radiometric calibration or terrain correction.

## Export a portable deployment bundle

`export-deployment` creates a self-contained directory by default. The manifest, model checkpoint, resolved training configuration and normalization statistics are kept together, and the manifest uses paths relative to itself rather than machine-specific locations.

```bash
floodmap export-deployment \
  --output-file deployment/unet_resnet34_vv_vh/deployment_manifest.yaml \
  --model-name unet_resnet34_vv_vh \
  --config runs/final_model/config.yaml \
  --checkpoint runs/final_model/models/best.pth \
  --threshold 0.80 \
  --input-modalities vv vh \
  --inference-mode sliding_window \
  --window-size 512 \
  --window-overlap 128
```

The resulting directory is portable as one unit:

```text
deployment/unet_resnet34_vv_vh/
├── deployment_manifest.yaml
├── deployment_bundle.json
├── DEPLOYMENT_README.md
└── assets/
    ├── configs/member_01_config.yaml
    ├── checkpoints/member_01_checkpoint.pth
    └── normalization/normalization_stats.json
```

Copy or archive the entire directory. `predict-scene` resolves every relative path from the manifest location, so the bundle may be moved between Linux, Windows, macOS, servers, containers, mounted drives and Colab without editing the manifest. The Flood Extent Mapping Python package still needs to be installed in the destination environment.

Use `--reference-only` only when external absolute asset paths are intentional. Such a manifest is tied to that environment and is not portable.

## Input modes

### Direct raster input

Use this when the exact SAR and DEM rasters are already known.

```bash
floodmap predict-scene \
  --manifest deployment_manifest.yaml \
  --sar-path scene_vv_vh.tif \
  --dem-path scene_dem.tif \
  --output-dir predictions/scene_001 \
  --plain-progress
```

The SAR raster should contain VV in band 1 and VH in band 2.

Generate the self-contained HTML report through the same CLI command:

```bash
floodmap predict-scene \
  --manifest deployment_manifest.yaml \
  --sar-path scene_vv_vh.tif \
  --mask-path scene_mask.tif \
  --evaluate \
  --output-dir predictions/scene_001 \
  --write-probability \
  --write-html-report \
  --plain-progress
```

Inside Colab, run the module with `%run -m floods.cli ... --display-inline` when the report should also be rendered directly below the cell. The HTML file is still written to the output directory. Inline report styles are scoped to the Flood Extent Mapping report container, so rendering a report cannot restyle or obscure preceding notebook logs.

### Folder input

Use this when a scene folder contains one or more SAR rasters.

```bash
floodmap discover-scene \
  --scene-dir EMSR001/sar \
  --output-file EMSR001_inventory.csv \
  --plain-progress
```

The inventory includes candidate names, SAR paths, date when available, CRS, shape, bounds and pixel size.

### Explicit CSV input

Use this when automatic discovery is not enough or when exact pairing is important.

```csv
candidate_id,sar_path,vv_path,vh_path,dem_path,mask_path,date,mosaic_group
tile_a,/data/sar_a.tif,,,/data/dem.tif,/data/mask_a.tif,2020-01-01,acq_20200101
tile_b,/data/sar_b.tif,,,/data/dem.tif,/data/mask_b.tif,2020-01-01,acq_20200101
```

```bash
floodmap predict-scene \
  --manifest deployment_manifest.yaml \
  --input-csv scenes.csv \
  --output-dir predictions/from_csv \
  --plain-progress
```

## Candidate names

If no acquisition date is found, candidate names use the clean file stem. For example, `EMSR107-7-2.tif` becomes `EMSR107-7-2`, not `undated_EMSR107-7-2`.

You can customise names with:

```bash
--scene-id EMSR001
--candidate-prefix review
--candidate-name-template "{scene_id}_{date}_{stem}"
```

Available template fields are `{scene_id}`, `{date}`, `{stem}`, `{index}` and `{kind}`.

## Safe mosaicking

Compatible multiband VV/VH tiles can be mosaicked before prediction:

```bash
floodmap predict-scene \
  --manifest deployment_manifest.yaml \
  --scene-dir EMSR001/sar \
  --dem-dir EMSR001/dem \
  --mosaic-compatible-sar-tiles \
  --output-dir predictions/mosaicked \
  --plain-progress
```

Mosaicking is conservative. Rasters must have matching CRS, pixel size and band count, and they must overlap or be spatially adjacent. Undated rasters are not mosaicked unless explicitly requested with `--mosaic-undated` or grouped through the CSV `mosaic_group` column.

### Mosaic planning without inspecting CSVs

`predict-scene` prints a mosaic decision whenever more than one SAR candidate is selected. A normal user should not need to open the discovery CSV before deployment.

The default is:

```bash
floodmap predict-scene ... --mosaic-mode smart
```

`smart` makes the safest useful decision in one run:

- for prediction-only deployment, compatible SAR tiles are mosaicked automatically;
- for labelled evaluation, compatible SAR tiles are mosaicked only when the matching masks can also be mosaicked safely onto the same grid;
- if mask pairing cannot be resolved safely, candidates are kept separate and the reason is printed.

For inspection without mosaicking:

```bash
floodmap predict-scene ... --mosaic-mode plan
```

`plan` prints the grouping decision and keeps candidates separate.

For conservative automatic mosaicking without smart labelled-mask handling:

```bash
floodmap predict-scene ... --mosaic-mode auto
```

For manual override:

```bash
floodmap predict-scene ... --mosaic-mode force
```

`force` still requires CRS/resolution/band/bounds compatibility, but it allows name-based undated groups. Use it only when the files are known to be tiles from the same acquisition.

## Forensic audit of labelled deployment errors

After running one or more labelled deployments, rank and inspect the worst false-positive and false-negative cases without rerunning inference:

```bash
floodmap audit-deployment-errors \
  --deployment-dir $ARTIFACTS_DIR/deployment_outputs/EMSR107_1_ensemble_eval \
  --deployment-dir $ARTIFACTS_DIR/deployment_outputs/EMSR107_5_ensemble_eval \
  --deployment-dir $ARTIFACTS_DIR/deployment_outputs/EMSR107_6_ensemble_eval \
  --deployment-dir $ARTIFACTS_DIR/deployment_outputs/EMSR107_7_ensemble_eval \
  --output-dir $ARTIFACTS_DIR/deployment_error_forensics \
  --max-montages 30 \
  --plain-progress
```

The command writes a worst-first ranking CSV, detailed JSON, an HTML report and montages containing VV, VH, DEM, flood probability, ground truth, prediction and a TP/FP/FN error map. Its diagnoses are deliberately limited to behaviour visible in the available rasters; it does not invent land-cover labels.
