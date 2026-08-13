from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from streamlit_app.raster_inputs import (
    canonical_tile_id,
    inspect_upload,
    prepare_sar_candidates,
    stage_auxiliary_uploads,
)


class Upload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.size = len(data)

    def getbuffer(self):
        return memoryview(self._data)


def _tif_bytes(count: int = 1, *, description: str | None = None) -> bytes:
    data = np.ones((count, 4, 5), dtype=np.float32)
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=5,
            height=4,
            count=count,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(-1, 51, 0.001, 0.001),
        ) as dst:
            dst.write(data)
            if description:
                dst.set_band_description(1, description)
        return mem.read()


def test_combined_sar_is_detected_without_user_declaring_band_layout(tmp_path: Path):
    upload = Upload("EMSR445-1-0.tif", _tif_bytes(count=2))
    info = inspect_upload(upload)
    assert info.count == 2
    candidates, errors, warnings = prepare_sar_candidates([upload], tmp_path)
    assert errors == []
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "EMSR445-1-0"
    assert candidates[0].kind == "multiband_vv_vh"


def test_single_band_vv_vh_pair_is_detected_from_filename(tmp_path: Path):
    vv = Upload("scene_07_vv.tif", _tif_bytes(count=1))
    vh = Upload("scene_07_vh.tif", _tif_bytes(count=1))
    candidates, errors, _ = prepare_sar_candidates([vh, vv], tmp_path)
    assert errors == []
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "scene_07"
    assert candidates[0].vv_path is not None
    assert candidates[0].vh_path is not None


def test_one_single_band_file_requests_matching_polarization_instead_of_inference(tmp_path: Path):
    vv = Upload("scene_07_vv.tif", _tif_bytes(count=1))
    candidates, errors, _ = prepare_sar_candidates([vv], tmp_path)
    assert candidates == []
    assert any("matching VV/VH polarization" in message for message in errors)


def test_auxiliary_files_match_by_tile_id_not_upload_order(tmp_path: Path):
    a = Upload("EMSR445-1-1_mask.tif", _tif_bytes(count=1))
    b = Upload("EMSR445-1-0_mask.tif", _tif_bytes(count=1))
    mapping, errors = stage_auxiliary_uploads([a, b], tmp_path, "mask")
    assert errors == []
    assert sorted(mapping) == ["EMSR445-1-0", "EMSR445-1-1"]
    assert mapping["EMSR445-1-0"].name == "EMSR445-1-0_mask.tif"


def test_canonical_tile_id_preserves_spatial_tile_number():
    assert canonical_tile_id("EMSR445-1-6.tif") == "EMSR445-1-6"
    assert canonical_tile_id("EMSR445-1-6_dem.tif", auxiliary=True) == "EMSR445-1-6"
    assert canonical_tile_id("EMSR445-1-6_vh.tif") == "EMSR445-1-6"
