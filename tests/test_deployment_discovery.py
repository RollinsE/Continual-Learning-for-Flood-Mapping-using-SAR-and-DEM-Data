from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from floods.deployment import discover_scene


def _write_tif(path: Path, count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((count, 8, 8), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=count,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 8, 1, 1),
    ) as dst:
        dst.write(data)


def test_discover_scene_finds_multiband_and_vv_vh_pair(tmp_path: Path):
    _write_tif(tmp_path / "EMSR001_20200101_sar_stack.tif", count=2)
    _write_tif(tmp_path / "EMSR001_20200202_tile01_VV.tif", count=1)
    _write_tif(tmp_path / "EMSR001_20200202_tile01_VH.tif", count=1)
    _write_tif(tmp_path / "EMSR001_dem.tif", count=1)
    rows = discover_scene(tmp_path, output_file=tmp_path / "inventory.csv")
    ready = [row for row in rows if row["status"] == "ready"]
    assert len(ready) == 2
    assert any(row["kind"] == "multiband_vv_vh" for row in ready)
    assert any(row["kind"] == "separate_vv_vh" for row in ready)
    assert (tmp_path / "inventory.csv").exists()


def test_deployment_raster_profile_removes_incompatible_block_metadata():
    from floods.deployment import _deployment_raster_profile

    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 2,
        "dtype": "float32",
        "blockxsize": 256,
        "blockysize": 256,
        "tiled": False,
        "bounds": (0.0, 0.0, 8.0, 8.0),
        "res": (1.0, 1.0),
    }
    out = _deployment_raster_profile(profile, dtype="uint8", nodata=0)
    assert out["driver"] == "GTiff"
    assert out["count"] == 1
    assert out["dtype"] == "uint8"
    assert "blockxsize" not in out
    assert "blockysize" not in out
    assert "tiled" not in out
    assert "bounds" not in out
    assert "res" not in out


def test_visual_report_embeds_prediction_images_and_summary(tmp_path: Path):
    from floods.deployment import _save_png, _write_visual_report

    img = _save_png(tmp_path / "preview.png", np.ones((4, 4), dtype=np.float32))
    metadata = {
        "threshold": 0.45,
        "flood_pixels": 12,
        "output_mask": str(tmp_path / "mask.tif"),
        "output_probability": str(tmp_path / "probability.tif"),
    }
    report = _write_visual_report(tmp_path / "report.html", "Deployment report", metadata, [("Final flood prediction", img)])
    text = report.read_text(encoding="utf-8")
    assert "Final flood prediction" in text
    assert "Predicted flood pixels" in text
    assert "data:image/png;base64" in text
    assert "overflow-wrap: anywhere" in text
    assert "Prediction metadata" in text
    assert "<details" in text


def test_deployment_binary_metrics_ignore_255():
    from floods.deployment import _compute_binary_metrics

    pred = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    target = np.array([[1, 0, 0], [1, 255, 0]], dtype=np.uint8)
    metrics = _compute_binary_metrics(pred, target)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["ignored_pixels"] == 1



def test_deployment_result_table_uses_report_filename(caplog):
    from floods.deployment import _print_deployment_result_table, _set_deploy_output_mode

    _set_deploy_output_mode("concise")
    summary = {
        "output_dir": "/tmp/out",
        "predictions": [
            {
                "candidate_id": "candidate_a",
                "flood_pixels": 42,
                "flood_fraction": 0.125,
                "flood_area_km2": 0.004,
                "visual_report": "/tmp/out/candidate_a/candidate_a_report.html",
            }
        ],
    }
    with caplog.at_level("INFO"):
        _print_deployment_result_table(summary)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "candidate_a_report.html" in text
    assert " yes" not in text


def test_mask_alignment_info_reports_overlap_and_resampling(tmp_path: Path):
    from floods.deployment import _mask_alignment_info

    sar_path = tmp_path / "sar.tif"
    mask_path = tmp_path / "mask_ground_truth.tif"
    _write_tif(sar_path, count=2)
    _write_tif(mask_path, count=1)
    with rasterio.open(sar_path) as src:
        profile = src.profile.copy()
        profile["bounds"] = tuple(float(v) for v in src.bounds)
        shape = (src.height, src.width)

    info = _mask_alignment_info(mask_path, profile, shape)
    assert info["crs_match"] is True
    assert info["bounds_overlap_ratio"] == 1.0
    assert info["resampled_to_sar_grid"] is False




def test_mask_reader_refuses_shape_only_alignment_without_georeferencing(tmp_path: Path):
    from floods.deployment import _read_mask_for_evaluation

    mask_path = tmp_path / "mask_no_crs.tif"
    values = np.zeros((4, 4), dtype=np.uint8)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(values, 1)

    sar_profile = {
        "height": 8,
        "width": 8,
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": from_origin(0, 8, 1, 1),
    }
    try:
        _read_mask_for_evaluation(mask_path, sar_profile, (8, 8))
    except ValueError as exc:
        assert "rather than resizing the mask by array shape alone" in str(exc)
    else:
        raise AssertionError("Expected unsafe shape-only mask alignment to be rejected")


def test_visual_report_includes_mask_alignment_section(tmp_path: Path):
    from floods.deployment import _save_png, _write_visual_report

    img = _save_png(tmp_path / "preview.png", np.ones((4, 4), dtype=np.float32))
    metadata = {
        "threshold": 0.45,
        "flood_pixels": 12,
        "output_mask": str(tmp_path / "mask.tif"),
        "output_probability": str(tmp_path / "probability.tif"),
        "evaluation_metrics": {"f1": 0.5, "iou": 0.33, "precision": 0.4, "recall": 0.8, "mcc": 0.45, "tp": 4, "tn": 8, "fp": 6, "fn": 1},
        "mask_alignment": {"mask_path": str(tmp_path / "gt.tif"), "crs_match": True, "bounds_overlap_ratio": 0.98, "resampled_to_sar_grid": False, "sar_shape": [4, 4], "mask_shape": [4, 4]},
    }
    report = _write_visual_report(tmp_path / "report.html", "Deployment report", metadata, [("SAR + ground-truth mask overlay", img)])
    text = report.read_text(encoding="utf-8")
    assert "Labelled-scene evaluation and mask alignment" in text
    assert "Bounds overlap" in text
    assert "SAR + ground-truth mask overlay" in text


def test_discover_scene_uses_clean_fallback_name_and_bounds(tmp_path: Path):
    _write_tif(tmp_path / "EMSR107-7-2.tif", count=2)
    rows = discover_scene(tmp_path, output_file=tmp_path / "inventory.csv")
    ready = [row for row in rows if row["status"] == "ready"]
    assert len(ready) == 1
    assert ready[0]["candidate_id"] == "EMSR107-7-2"
    assert not ready[0]["candidate_id"].startswith("undated")
    assert ready[0]["left"] != ""
    assert ready[0]["bottom"] != ""
    assert ready[0]["right"] != ""
    assert ready[0]["top"] != ""
    text = (tmp_path / "inventory.csv").read_text(encoding="utf-8")
    assert "left,bottom,right,top,res_x,res_y" in text


def test_discover_scene_supports_candidate_name_template(tmp_path: Path):
    _write_tif(tmp_path / "EMSR001_20200101_stack.tif", count=2)
    rows = discover_scene(
        tmp_path,
        scene_id="event42",
        candidate_name_template="{scene_id}_{date}_{stem}",
    )
    assert rows[0]["candidate_id"] == "event42_2020-01-01_EMSR001_20200101_stack"


def test_input_csv_candidates_are_parsed(tmp_path: Path):
    from floods.deployment import _candidates_from_input_csv

    sar = tmp_path / "sar.tif"
    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    _write_tif(sar, count=2)
    _write_tif(dem, count=1)
    _write_tif(mask, count=1)
    csv_path = tmp_path / "inputs.csv"
    csv_path.write_text(
        "candidate_id,sar_path,dem_path,mask_path,date,mosaic_group\n"
        f"tile_a,{sar},{dem},{mask},2020-01-01,group_a\n",
        encoding="utf-8",
    )
    candidates = _candidates_from_input_csv(csv_path)
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "tile_a"
    assert candidates[0].dem_path == dem
    assert candidates[0].mask_path == mask
    assert candidates[0].mosaic_group == "group_a"


def test_mosaic_multiband_candidates_combines_compatible_tiles(tmp_path: Path):
    from floods.deployment import _mosaic_multiband_candidates, _candidate_from_row

    left = tmp_path / "left.tif"
    right = tmp_path / "right.tif"
    data = np.ones((2, 8, 8), dtype=np.float32)
    for path, transform in [
        (left, from_origin(0, 8, 1, 1)),
        (right, from_origin(8, 8, 1, 1)),
    ]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)
    candidates = [
        _candidate_from_row({"candidate_id": "left", "kind": "multiband_vv_vh", "sar_path": str(left), "date": "2020-01-01", "status": "ready"}),
        _candidate_from_row({"candidate_id": "right", "kind": "multiband_vv_vh", "sar_path": str(right), "date": "2020-01-01", "status": "ready"}),
    ]
    out = _mosaic_multiband_candidates(candidates, tmp_path / "out")
    assert len(out) == 1
    assert out[0].kind == "mosaic_multiband_vv_vh"
    assert out[0].sar_path is not None and out[0].sar_path.exists()
    with rasterio.open(out[0].sar_path) as src:
        assert src.count == 2
        assert src.width == 16
        assert src.height == 8


def test_mosaic_multiband_candidates_can_mosaic_matching_masks(tmp_path: Path):
    from floods.deployment import _mosaic_multiband_candidates, _candidate_from_row

    left = tmp_path / "EMSR001-1.tif"
    right = tmp_path / "EMSR001-2.tif"
    mask_dir = tmp_path / "masks"
    left_mask = mask_dir / "EMSR001-1.tif"
    right_mask = mask_dir / "EMSR001-2.tif"
    data = np.ones((2, 8, 8), dtype=np.float32)
    mask = np.zeros((1, 8, 8), dtype=np.uint8)
    mask[:, 2:6, 2:6] = 1
    for path, mpath, transform in [
        (left, left_mask, from_origin(0, 8, 1, 1)),
        (right, right_mask, from_origin(8, 8, 1, 1)),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            mpath,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
            nodata=255,
        ) as dst:
            dst.write(mask)
    candidates = [
        _candidate_from_row({"candidate_id": "EMSR001-1", "kind": "multiband_vv_vh", "sar_path": str(left), "status": "ready"}),
        _candidate_from_row({"candidate_id": "EMSR001-2", "kind": "multiband_vv_vh", "sar_path": str(right), "status": "ready"}),
    ]
    out = _mosaic_multiband_candidates(
        candidates,
        tmp_path / "out",
        use_name_group=True,
        evaluating=True,
        mask_dir=mask_dir,
        require_mask_mosaic=True,
    )
    assert len(out) == 1
    assert out[0].sar_path is not None and out[0].sar_path.exists()
    assert out[0].mask_path is not None and out[0].mask_path.exists()
    with rasterio.open(out[0].mask_path) as src:
        arr = src.read(1)
        assert src.width == 16
        assert src.height == 8
        assert int((arr == 1).sum()) == 32


def test_reference_grid_reader_reprojects_instead_of_shape_only_resize(tmp_path: Path):
    from floods.deployment import _read_band_to_reference_grid
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    reference_profile = {
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": from_origin(0.0, 1.0, 0.1, 0.1),
        "height": 10,
        "width": 10,
    }
    src_path = tmp_path / "dem.tif"
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        width=5,
        height=5,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 0.2, 0.2),
    ) as dst:
        dst.write(values, 1)

    out = _read_band_to_reference_grid(
        src_path,
        reference_profile,
        10,
        10,
        resampling=Resampling.bilinear,
        label="DEM",
    )
    assert out.shape == (10, 10)
    assert np.isfinite(out).any()
