from pathlib import Path

import numpy as np
import pandas as pd

from floods.hard_negative_regions import (
    AuditGuidedHardNegativeCropSupervision,
    MODE_AUDIT_HARD_NEGATIVE,
    _candidate_for_component,
    _connected_components,
    manifest_matching_indices,
    prepare_hard_negative_region_sampler,
)


def test_mined_manifest_crop_is_applied(tmp_path):
    manifest = tmp_path / "hard_negative_regions.csv"
    pd.DataFrame([
        {
            "tile_id": "EMSR001-1-1_0_0",
            "file": "EMSR001-1-1_0_0.tif",
            "x0": 4,
            "y0": 8,
            "crop_size": 16,
            "fp_pixels": 100,
            "score": 50.0,
        }
    ]).to_csv(manifest, index=False)

    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[8:24, 4:20, :] = 7.0
    mask = np.zeros((32, 32), dtype=np.uint8)

    cropper = AuditGuidedHardNegativeCropSupervision(manifest, target_size=32, probability=1.0)
    cropped_image, cropped_mask, metadata = cropper(
        image=image,
        mask=mask,
        sample_path=Path("/data/train/mask/EMSR001-1-1_0_0.tif"),
    )

    assert cropped_image.shape == (32, 32, 3)
    assert cropped_mask.shape == (32, 32)
    assert np.allclose(cropped_image, 7.0)
    assert metadata.requested_mode == MODE_AUDIT_HARD_NEGATIVE
    assert metadata.applied_mode == MODE_AUDIT_HARD_NEGATIVE
    assert metadata.crop_size == 16


def test_manifest_matching_indices_uses_tile_stem(tmp_path):
    manifest = tmp_path / "hard_negative_regions.csv"
    pd.DataFrame([
        {"tile_id": "tile_b", "x0": 0, "y0": 0, "crop_size": 16, "score": 1.0}
    ]).to_csv(manifest, index=False)
    labels = ["/data/train/mask/tile_a.tif", "/data/train/mask/tile_b.tif"]
    assert manifest_matching_indices(labels, manifest) == {1}


def test_candidate_rejects_regions_that_include_too_much_true_flood():
    target = np.zeros((64, 64), dtype=np.uint8)
    target[16:48, 16:48] = 1
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[0:20, 0:20] = 0.9
    hard = (prob >= 0.6) & (target == 0)
    component = _connected_components(hard)[0]
    candidate = _candidate_for_component(
        component=component,
        prob=prob,
        target=target,
        threshold=0.6,
        crop_sizes=[64],
        min_fp_pixels=16,
        max_label_fg_ratio=0.001,
        min_valid_ratio=0.5,
    )
    assert candidate is None


def test_sampler_warns_when_cap_is_below_natural_prevalence(tmp_path, caplog):
    manifest = tmp_path / "hard_negative_regions.csv"
    pd.DataFrame([
        {"tile_id": "tile_a", "x0": 0, "y0": 0, "crop_size": 16, "score": 1.0},
        {"tile_id": "tile_b", "x0": 0, "y0": 0, "crop_size": 16, "score": 1.0},
    ]).to_csv(manifest, index=False)

    class Dataset:
        label_files = [
            "/data/train/mask/tile_a.tif",
            "/data/train/mask/tile_b.tif",
            "/data/train/mask/tile_c.tif",
        ]

        def __len__(self):
            return len(self.label_files)

    with caplog.at_level("WARNING"):
        prepare_hard_negative_region_sampler(
            dataset=Dataset(),
            manifest_path=manifest,
            weight=2.0,
            max_fraction=0.50,
            samples_multiplier=1.0,
        )
    assert "at or below the natural region-tile prevalence" in caplog.text
