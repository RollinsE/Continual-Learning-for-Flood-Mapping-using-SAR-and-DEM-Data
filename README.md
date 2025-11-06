This script provides a robust continual learning (CL) and preprocessing pipeline for flood extent mapping using satellite image triplets (SAR, DEM, and masks). 
It covers tiling, data preprocessing, quality filtering (e.g., NaN masking), experiment orchestration, model training, continual/few-shot learning strategies, and detailed 
evaluation metric tracking.

Features
Geospatial Preprocessing: Tiles geo-referenced images, checks window validity, reads and saves GeoTIFF tiles.

Data Filtering: Excludes tiles with excessive NaN/invalid values.

File Matching: Matches SAR, DEM, and mask files by EMSR code and subset membership.

Pipeline Execution: Loads activation/metadata files, orchestrates data splits for train/validation/test.

Continual Learning: Supports multi-task CL strategies with buffer/replay, staged evaluation, and metrics/logging for plasticity, forgetting, F1/IoU, and more.

Evaluation and Cleanup: Calculates and logs advanced CL metrics, handles memory management, and cleans up large objects between runs.

Dependencies
Python libraries: numpy, pandas, torch (PyTorch), rasterio, glob, tqdm, seaborn, json, and standard scientific stack.
