# Model card: U-Net ResNet34 VV+VH flood segmentation

## Model description

- Task: binary flood segmentation.
- Architecture: U-Net decoder with ResNet34 encoder.
- Inputs: Sentinel-1 VV and VH backscatter in dB.
- Initialisation: ImageNet-pretrained encoder adapted from three to two input channels.
- Output: per-pixel flood probability.
- Operating threshold: 0.80.

## Training data

MMFlood training split (3,980 tiles of 512 × 512 pixels), with the published validation (324 tiles) and test (386 scenes) splits held out. SAR is stored in dB, no-data pixels are ignored in the loss and metrics, and morphological mask clean-up is applied to training masks only. Normalisation statistics are fitted on the training split.

## Selection protocol

Modality selection used event-level cross-validation on training events (VV+VH: event-macro F1 0.478 on fold 0; VV+VH+DEM: 0.460). The operating threshold was calibrated on out-of-fold predictions (event-macro F1 0.424 at 0.80). The final model was trained on the training split for six epochs, the checkpoint with the best validation F1 (epoch 6) was kept, and the model was evaluated once on the test split.

## Performance

On the 386 test scenes with the published labels, sliding-window inference (512 px windows, 128 px overlap), threshold 0.80:

| Metric | Value |
|---|---:|
| F1 | 0.520 |
| IoU | 0.351 |
| Precision | 0.439 |
| Recall | 0.637 |
| MCC | 0.508 |

Metrics are pooled over all test pixels. They vary considerably between events and between areas of the same event.

## Intended use

Research, benchmarking, and decision-support workflows where model outputs are reviewed alongside source imagery and contextual information.

## Limitations

- Performance varies substantially between flood events and geographic domains.
- The model detects open water. Permanent water bodies are often predicted as flood, while flooded vegetation, urban flooding and shallow water with no clear SAR signature are often missed.
- Raw DEM did not improve cross-event performance enough to be included; this is not evidence that terrain is irrelevant.
- Test metrics describe the designated dataset split and should not be treated as universal operational performance.
- The model should not be the sole basis for emergency, evacuation, insurance, or life-safety decisions.

## Reproducibility

See `docs/REPRODUCIBILITY.md`, `configs/reproduction/`, `reproducibility/reference_metrics.json` and `notebooks/colab_pipeline.ipynb`.
