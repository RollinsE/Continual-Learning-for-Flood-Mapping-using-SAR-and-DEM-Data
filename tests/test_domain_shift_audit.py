from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from floods.cli import build_cli_parser
from floods.domain_shift_audit import audit_domain_shift


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
        transform=from_origin(0.0, 160.0, 10.0, 10.0),
    ) as dst:
        dst.write(data)


def _make_tile(root: Path, split: str, event: str, index: int, shift: float) -> None:
    height = width = 16
    yy, xx = np.mgrid[:height, :width]
    flood = ((xx - (6 + index % 3)) ** 2 + (yy - 8) ** 2) < (10 + index)
    if index % 5 == 0:
        flood[:] = False
    vv = (0.03 + shift + 0.002 * xx + 0.035 * flood).astype(np.float32)
    vh = (0.009 + shift * 0.2 + 0.0005 * yy + 0.008 * flood).astype(np.float32)
    dem = (50.0 + shift * 2000.0 + yy * 2.0 + xx * 0.3).astype(np.float32)
    stem = f"{event}-1-{index + 1}_{index * 16}_0"
    _write(root / split / "sar" / f"{stem}.tif", np.stack([vv, vh]))
    _write(root / split / "dem" / f"{stem}.tif", dem)
    _write(root / split / "mask" / f"{stem}.tif", flood.astype(np.uint8))


def test_domain_shift_audit_writes_diagnostic_outputs(tmp_path):
    for event_index, event in enumerate(("EMSR001", "EMSR002", "EMSR003")):
        for tile_index in range(4):
            _make_tile(tmp_path, "train", event, tile_index, shift=0.002 * event_index)
    for tile_index in range(6):
        _make_tile(tmp_path, "val", "EMSR342", tile_index, shift=0.08)

    output = tmp_path / "domain_audit"
    summary = audit_domain_shift(
        tmp_path,
        output,
        target_events=["EMSR342"],
        input_modalities=["vv", "vh", "dem"],
        max_pixels_per_tile=64,
        max_pixels_per_class_per_tile=32,
        max_total_pixels_per_domain=1000,
        write_plots=False,
    )

    assert summary["reference_tiles"] == 12
    assert summary["target_tiles"] == 6
    assert summary["reference_event_count"] == 3
    assert summary["deployable_sensor_terrain_roc_auc"] >= 0.8
    classifier = pd.read_csv(output / "domain_classifier_metrics.csv")
    assert "deployable_sensor_terrain" in set(classifier["feature_set"])
    assert "label_conditioned_input" in set(classifier["feature_set"])
    assert (output / "summary.json").exists()
    assert (output / "tile_features.csv").exists()
    pixel = pd.read_csv(output / "pixel_distribution_shift.csv")
    assert set(pixel["feature"]) == {"vv", "vh", "dem"}
    assert set(pixel["stratum"]) == {"all", "flood", "background"}
    similarity = pd.read_csv(output / "training_event_similarity.csv")
    assert len(similarity) == 3
    assert "distance_combined" in similarity.columns


def test_domain_shift_cli_parses_target_event_and_sampling_options():
    parser = build_cli_parser()
    args = parser.parse_args(
        [
            "audit-domain-shift",
            "--processed-data-dir",
            "/tmp/processed",
            "--output-dir",
            "/tmp/audit",
            "--target-events",
            "EMSR342",
            "--input-modalities",
            "vv",
            "vh",
            "dem",
            "--max-reference-tiles",
            "1000",
            "--no-write-plots",
        ]
    )
    assert args.command == "audit-domain-shift"
    assert args.target_events == ["EMSR342"]
    assert args.input_modalities == ["vv", "vh", "dem"]
    assert args.max_reference_tiles == 1000
    assert args.write_plots is False
