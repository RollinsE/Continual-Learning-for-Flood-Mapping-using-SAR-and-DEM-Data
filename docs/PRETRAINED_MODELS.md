# Registered pretrained models

Flood Extent Mapping uses one central model registry for ordinary training, event-level cross-validation, out-of-fold threshold calibration, checkpoint evaluation, model comparison, and deployment reconstruction. Users select a source through the CLI; no package YAML or Python file needs to be edited.

List the installed registry:

```bash
floodmap model-catalog
```

## Available sources

| `--weights-source` | Default encoder | Default decoder | Supported inputs | Normalisation |
|---|---|---|---|---|
| `random` | chosen with `--encoder` | chosen with `--decoder` | configurable | base config |
| `imagenet` | chosen with `--encoder` | chosen with `--decoder` | configurable | train-fitted MMFlood statistics |
| `ssl4eo_s1_moco` | ResNet50 | U-Net | VV, VH | SSL4EO Sentinel-1 statistics |
| `ssl4eo_s1_decur` | ResNet50 | U-Net | VV, VH | SSL4EO Sentinel-1 statistics |
| `fgmae_sar_vit_small` | ViT-S/16 | SegFormer | VV, VH | SSL4EO Sentinel-1 statistics |
| `fgmae_sar_vit_base` | ViT-B/16 | SegFormer | VV, VH | SSL4EO Sentinel-1 statistics |
| `croma_sar_base` | CROMA Base | SegFormer | VV, VH | fold-specific train-fitted statistics |
| `terramind_v1_tiny` | TerraMind Tiny | SegFormer | VV, VH; optional DEM | TerraMind provider statistics |
| `terramind_v1_small` | TerraMind Small | SegFormer | VV, VH; optional DEM | TerraMind provider statistics |
| `terramind_v1_base` | TerraMind Base | SegFormer | VV, VH; optional DEM | TerraMind provider statistics |

The registry validates incompatible combinations before training. For example, CROMA and FG-MAE SAR reject DEM, while TerraMind accepts either `vv vh` or `vv vh dem`.

## Direct training syntax

Registered Earth-observation models normally need only the source and modalities:

```bash
floodmap train \
  --config configs/training_defaults.yaml \
  --weights-source terramind_v1_tiny \
  --input-modalities vv vh dem \
  ...
```

ImageNet remains architecture-configurable:

```bash
floodmap train \
  --config configs/training_defaults.yaml \
  --weights-source imagenet \
  --encoder resnet34 \
  --decoder unet \
  --input-modalities vv vh \
  ...
```

Every run stores the fully resolved provider, encoder, decoder, modalities and normalisation policy in its own `config.yaml`.

## Event-CV candidate syntax

The same registry is used by `floodmap event-cv`:

```text
SOURCE[:DECODER[:ENCODER[:LABEL]]]
```

Examples:

```text
terramind_v1_tiny
croma_sar_base
fgmae_sar_vit_small
ssl4eo_s1_moco
imagenet:unet:resnet34:reference_unet
```

Provider candidates normally require only `SOURCE`. ImageNet and random candidates require an encoder and decoder because those sources are architecture-agnostic.
