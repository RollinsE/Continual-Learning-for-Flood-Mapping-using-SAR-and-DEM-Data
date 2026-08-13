import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from floods.datasets.flood import FloodDataset
from floods.derived_features import (
    DERIVED_BAND_ORDER,
    DERIVED_SCHEMA_VERSION,
    build_derived_features,
    derive_feature_stack,
    pixel_spacing_meters,
)
from floods.normalization import fit_normalization_stats


def _write_tif(path: Path, data: np.ndarray) -> None:
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
        transform=from_origin(100.0, 200.0, 10.0, 10.0),
    ) as dst:
        dst.write(data)


def _make_processed_dataset(root: Path, splits=("train", "val", "test")) -> None:
    height = width = 32
    yy, xx = np.mgrid[:height, :width]
    vv = (0.05 + xx * 0.001).astype(np.float32)
    vh = (0.01 + yy * 0.0005).astype(np.float32)
    dem = (100.0 + xx * 2.0 + yy).astype(np.float32)
    mask = ((xx > 15) & (yy > 10)).astype(np.uint8)
    for split in splits:
        stem = "EMSR342-1-1_0_0"
        _write_tif(root / split / "sar" / f"{stem}.tif", np.stack([vv, vh]))
        _write_tif(root / split / "dem" / f"{stem}.tif", dem)
        _write_tif(root / split / "mask" / f"{stem}.tif", mask)


def test_derive_feature_stack_has_expected_formula_and_flat_terrain():
    height = width = 32
    vv = np.full((height, width), 0.1, dtype=np.float32)
    vh = np.full((height, width), 0.01, dtype=np.float32)
    dem = np.full((height, width), 50.0, dtype=np.float32)

    result = derive_feature_stack(
        np.stack([vv, vh]),
        dem,
        pixel_size_x=10.0,
        pixel_size_y=-10.0,
    )

    assert result.shape == (3, height, width)
    np.testing.assert_allclose(result[0], 10.0, atol=1e-5)
    np.testing.assert_allclose(result[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(result[2], 0.0, atol=1e-6)


def test_build_derived_features_and_six_channel_dataset(tmp_path):
    _make_processed_dataset(tmp_path)
    manifest = build_derived_features(tmp_path, splits=("train", "val", "test"))

    assert manifest["band_order"] == list(DERIVED_BAND_ORDER)
    derived_path = tmp_path / "train" / "derived" / "EMSR342-1-1_0_0.tif"
    with rasterio.open(derived_path) as src:
        assert src.count == 3
        assert tuple(src.descriptions) == DERIVED_BAND_ORDER

    dataset = FloodDataset(
        tmp_path,
        subset="train",
        input_modalities=[
            "vv",
            "vh",
            "dem",
            "vv_vh_log_ratio",
            "dem_slope",
            "dem_tpi",
        ],
    )
    image, label = dataset[0]
    assert image.shape == (32, 32, 6)
    assert label.shape == (32, 32)
    np.testing.assert_allclose(image[..., 3], 10.0 * np.log10(image[..., 0]) - 10.0 * np.log10(image[..., 1]), atol=1e-4)


def test_fit_normalization_can_preserve_baseline_channels(tmp_path):
    _make_processed_dataset(tmp_path, splits=("train",))
    build_derived_features(tmp_path, splits=("train",))

    preserved = {
        "schema_version": 2,
        "channels": [
            {
                "channel": "vv",
                "count": 123,
                "q_min": 1.0,
                "q_max": 99.0,
                "clip_min": 0.001,
                "clip_max": 0.2,
                "mean": 0.055,
                "std": 0.044,
                "raw_mean": 0.056,
                "raw_std": 0.045,
                "robust_mean": 0.25,
                "robust_std": 0.2,
                "raw_min": 0.0,
                "raw_max": 0.3,
            }
        ],
    }
    preserve_path = tmp_path / "baseline_stats.json"
    preserve_path.write_text(json.dumps(preserved), encoding="utf-8")
    output_path = tmp_path / "six_channel_stats.json"

    payload = fit_normalization_stats(
        tmp_path,
        output_path,
        input_modalities=[
            "vv",
            "vh",
            "dem",
            "vv_vh_log_ratio",
            "dem_slope",
            "dem_tpi",
        ],
        max_pixels_per_file=256,
        preserve_channel_stats_from=preserve_path,
    )

    channels = {item["channel"]: item for item in payload["channels"]}
    assert payload["preserved_channels"] == ["vv"]
    assert channels["vv"]["mean"] == 0.055
    assert channels["vv_vh_log_ratio"]["count"] > 0


def test_geographic_pixel_spacing_produces_realistic_slope():
    height = width = 32
    transform = from_origin(-1.0, 51.0, 0.0001426, 0.0000899)
    crs = rasterio.crs.CRS.from_epsg(4326)
    x_m, y_m = pixel_spacing_meters(transform, crs, width, height)

    assert 9.0 < x_m < 11.5
    assert 9.0 < y_m < 11.5

    # A DEM rising by one horizontal pixel spacing per pixel has a 45-degree slope.
    xx = np.arange(width, dtype=np.float32)[None, :]
    dem = np.repeat(xx * x_m, height, axis=0).astype(np.float32)
    vv = np.full((height, width), 0.1, dtype=np.float32)
    vh = np.full((height, width), 0.01, dtype=np.float32)
    result = derive_feature_stack(
        np.stack([vv, vh]),
        dem,
        pixel_size_x=x_m,
        pixel_size_y=y_m,
    )

    np.testing.assert_allclose(result[1], 45.0, atol=0.25)


def test_projected_pixel_spacing_respects_linear_units():
    transform = from_origin(100.0, 200.0, 10.0, 20.0)
    crs = rasterio.crs.CRS.from_epsg(32631)
    x_m, y_m = pixel_spacing_meters(transform, crs, 32, 32)
    assert x_m == 10.0
    assert y_m == 20.0


def test_stale_derived_schema_is_regenerated(tmp_path):
    _make_processed_dataset(tmp_path, splits=("train",))
    first = build_derived_features(tmp_path, splits=("train",))
    assert first["schema_version"] == DERIVED_SCHEMA_VERSION

    derived_path = tmp_path / "train" / "derived" / "EMSR342-1-1_0_0.tif"
    with rasterio.open(derived_path, "r+") as dst:
        dst.update_tags(derived_schema_version="1")

    second = build_derived_features(tmp_path, splits=("train",))
    assert second["splits"]["train"]["stale_rewritten"] == 1
    assert second["splits"]["train"]["skipped_existing"] == 0

    with rasterio.open(derived_path) as src:
        assert src.tags()["derived_schema_version"] == str(DERIVED_SCHEMA_VERSION)
        assert src.tags()["slope_horizontal_units"] == "metres"
