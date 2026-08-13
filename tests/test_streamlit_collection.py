from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from streamlit_app.collection import build_collection_mosaic, build_equal_area_evaluation, pooled_evaluation


def _write_raster(path: Path, values: np.ndarray, transform, *, dtype: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(values.astype(dtype), 1)
    return path


def test_collection_mosaic_handles_different_source_pixel_resolutions(tmp_path: Path):
    mask1 = np.zeros((10, 10), dtype=np.uint8)
    mask1[2:6, 2:6] = 1
    prob1 = mask1.astype(np.float32) * 0.9
    mask2 = np.zeros((8, 8), dtype=np.uint8)
    mask2[1:5, 1:5] = 1
    prob2 = mask2.astype(np.float32) * 0.8

    p1 = _write_raster(tmp_path / "a_mask.tif", mask1, from_origin(-1.0, 51.0, 0.001, 0.001), dtype="uint8")
    q1 = _write_raster(tmp_path / "a_prob.tif", prob1, from_origin(-1.0, 51.0, 0.001, 0.001), dtype="float32")
    p2 = _write_raster(tmp_path / "b_mask.tif", mask2, from_origin(-0.99, 51.0, 0.002, 0.002), dtype="uint8")
    q2 = _write_raster(tmp_path / "b_prob.tif", prob2, from_origin(-0.99, 51.0, 0.002, 0.002), dtype="float32")

    result = build_collection_mosaic(
        [
            {"candidate_id": "a", "output_mask": str(p1), "output_probability": str(q1)},
            {"candidate_id": "b", "output_mask": str(p2), "output_probability": str(q2)},
        ],
        tmp_path / "collection",
    )
    assert result["created"] is True
    assert result["crs"] == "EPSG:6933"
    assert Path(result["output_mask"]).is_file()
    assert Path(result["output_probability"]).is_file()
    assert result["mapped_area_km2"] > 0
    assert result["flood_area_km2"] > 0
    assert 0 < result["flood_fraction"] < 1


def test_pooled_evaluation_sums_confusion_counts():
    result = pooled_evaluation(
        [
            {"evaluation_metrics": {"tp": 8, "fp": 2, "fn": 2, "tn": 20}},
            {"evaluation_metrics": {"tp": 4, "fp": 1, "fn": 1, "tn": 10}},
            {},
        ]
    )
    assert result is not None
    assert result["tiles"] == 2
    assert result["tp"] == 12
    assert result["fp"] == 3
    assert result["fn"] == 3
    assert result["precision"] == result["recall"] == 0.8
    assert abs(result["f1"] - 0.8) < 1e-12


def test_equal_area_evaluation_uses_collection_grid_and_partial_masks(tmp_path: Path):
    mask1 = np.zeros((10, 10), dtype=np.uint8)
    mask1[2:6, 2:6] = 1
    prob1 = mask1.astype(np.float32) * 0.9
    pred1 = mask1.copy()
    truth1 = mask1.copy()

    mask2 = np.zeros((8, 8), dtype=np.uint8)
    mask2[1:5, 1:5] = 1
    prob2 = mask2.astype(np.float32) * 0.8

    p1 = _write_raster(tmp_path / "a_mask.tif", pred1, from_origin(-1.0, 51.0, 0.001, 0.001), dtype="uint8")
    q1 = _write_raster(tmp_path / "a_prob.tif", prob1, from_origin(-1.0, 51.0, 0.001, 0.001), dtype="float32")
    gt1 = _write_raster(tmp_path / "a_truth.tif", truth1, from_origin(-1.0, 51.0, 0.001, 0.001), dtype="uint8")
    p2 = _write_raster(tmp_path / "b_mask.tif", mask2, from_origin(-0.99, 51.0, 0.002, 0.002), dtype="uint8")
    q2 = _write_raster(tmp_path / "b_prob.tif", prob2, from_origin(-0.99, 51.0, 0.002, 0.002), dtype="float32")

    predictions = [
        {"candidate_id": "a", "output_mask": str(p1), "output_probability": str(q1), "mask_path": str(gt1)},
        {"candidate_id": "b", "output_mask": str(p2), "output_probability": str(q2), "mask_path": None},
    ]
    mosaic = build_collection_mosaic(predictions, tmp_path / "collection")
    result = build_equal_area_evaluation(predictions, mosaic, tmp_path / "collection")
    assert result is not None
    assert result["mode"] == "equal_area"
    assert result["tiles"] == 1
    assert result["labelled_area_km2"] > 0
    assert result["f1"] > 0.95
    assert Path(result["ground_truth_mosaic"]).is_file()
    assert Path(result["evaluation_overlay"]).is_file()


def test_collection_preview_marks_nodata_neutral_grey(tmp_path: Path):
    from matplotlib import image as mpimg
    from streamlit_app.collection import _save_preview

    values = np.array([[255, 0], [1, 0]], dtype=np.uint8)
    preview = _save_preview(values, tmp_path / "preview.png", discrete=True)
    rgba = mpimg.imread(preview)
    # The top-left source quadrant is no-data and should render as a neutral grey,
    # rather than the white page background or valid non-flood purple.
    sample = rgba[rgba.shape[0] // 4, rgba.shape[1] // 4, :3]
    assert np.max(sample) - np.min(sample) < 0.03
    assert 0.65 < float(sample.mean()) < 0.85
