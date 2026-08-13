# Model card: U-Net ResNet34 VV+VH reference model

## Model description

- Task: binary flood segmentation.
- Architecture: U-Net decoder with ResNet34 encoder.
- Inputs: Sentinel-1 VV and VH channels.
- Initialisation: ImageNet-pretrained encoder adapted from three to two input channels.
- Output: per-pixel flood probability.
- Operating threshold: 0.80.

## Selection protocol

Architecture and modality selection used event-level cross-validation on training events. The threshold was calibrated from out-of-fold predictions. The final model was trained on the designated training split, selected on the designated validation split, and evaluated once on the test split.

## Reference performance

On 386 test tiles:

| Metric | Value |
|---|---:|
| F1 | 0.5471 |
| IoU | 0.3765 |
| Precision | 0.4528 |
| Recall | 0.6909 |
| MCC | 0.5358 |

## Intended use

Research, benchmarking, and decision-support workflows where model outputs are reviewed alongside source imagery and contextual information.

## Limitations

- Performance varies substantially between flood events and geographic domains.
- Raw DEM did not provide a sufficiently consistent cross-event benefit to become part of the default profile; this is not evidence that terrain is irrelevant.
- The model may miss small, low-contrast, or domain-shifted floods and may create false positives in difficult non-flood surfaces.
- Test metrics describe the designated dataset split and should not be treated as universal operational performance.
- The model should not be the sole basis for emergency, evacuation, insurance, or life-safety decisions.

## Reproducibility

See `docs/REPRODUCIBILITY.md`, `configs/reproduction/`, and `reproducibility/reference_metrics.json`.
