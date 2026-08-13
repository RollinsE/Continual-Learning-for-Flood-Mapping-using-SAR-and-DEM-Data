from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from floods.derived_features import build_derived_features
from floods.feature_audit import audit_feature_separability


def _write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs="EPSG:32631",
        transform=from_origin(0.0, 320.0, 10.0, 10.0),
    ) as dst:
        dst.write(data)


def _make_split(root: Path, split: str, event: str, offset: float) -> None:
    height = width = 32
    yy, xx = np.mgrid[:height, :width]
    flood = (xx + yy) > 34
    vv = np.where(flood, 0.12 + offset, 0.04 + offset).astype(np.float32)
    vh = np.where(flood, 0.018, 0.011).astype(np.float32)
    dem = (200.0 - yy * 3.0 + xx * 0.2).astype(np.float32)
    stem = f"{event}-1-1_0_0"
    _write(root / split / "sar" / f"{stem}.tif", np.stack([vv, vh]))
    _write(root / split / "dem" / f"{stem}.tif", dem)
    _write(root / split / "mask" / f"{stem}.tif", flood.astype(np.uint8))


def test_feature_audit_writes_global_and_event_comparisons(tmp_path):
    _make_split(tmp_path, "train", "EMSR001", 0.0)
    _make_split(tmp_path, "val", "EMSR342", 0.002)
    build_derived_features(tmp_path, splits=("train", "val"))

    output = tmp_path / "audit"
    summary = audit_feature_separability(
        tmp_path,
        output,
        max_pixels_per_class_per_tile=64,
        max_total_pixels_per_split=500,
    )

    comparison = pd.read_csv(output / "logistic_comparison.csv")
    assert set(comparison["model"]) == {
        "base",
        "base_plus_ratio",
        "base_plus_terrain",
        "extended",
    }
    event = pd.read_csv(output / "event_logistic_comparison.csv")
    assert set(event["event_id"]) == {"EMSR342"}
    assert summary["recommendation"] in {
        "proceed_to_controlled_training",
        "weak_gain_run_only_one_controlled_training_experiment",
        "do_not_train_no_clear_added_separability",
    }
