"""Fast filesystem validation before expensive training setup begins."""

from __future__ import annotations

from pathlib import Path


def validate_training_data_path(data_config) -> None:
    """Fail before creating a run folder when the processed dataset is absent."""
    root = Path(data_config.path).expanduser()
    required = [root / "train" / "sar", root / "train" / "mask", root / "val" / "sar", root / "val" / "mask"]
    if bool(data_config.include_dem):
        required.extend([root / "train" / "dem", root / "val" / "dem"])
    problems = []
    for folder in required:
        if not folder.is_dir():
            problems.append(f"missing directory: {folder}")
            continue
        has_raster = any(folder.glob("*.tif")) or any(folder.glob("*.tiff"))
        if not has_raster:
            problems.append(f"no GeoTIFF files found: {folder}")
    if problems:
        details = "\n  - ".join(problems)
        raise FileNotFoundError(
            f"Processed training dataset is not ready at {root}.\n  - {details}\n"
            "Pass --processed-data-dir pointing to the directory that contains train/sar, train/mask, val/sar, and val/mask."
        )
