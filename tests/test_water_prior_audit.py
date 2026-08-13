from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from floods.water_prior_audit import (
    PriorSetting,
    PlanetaryComputerOccurrenceProvider,
    _annotate_sweep,
    _choose_best_setting,
    _choose_best_strategy_setting,
    _water_prior_recommendation,
    apply_water_prior,
    build_prior_settings,
)


def test_apply_water_prior_hard_exclusion_preserves_nodata():
    probability = np.array([[0.8, 0.8, 0.8]], dtype=np.float32)
    occurrence = np.array([[20, 95, 255]], dtype=np.uint8)
    setting = PriorSetting("hard_exclude", occurrence_threshold=90, penalty_strength=1.0)
    adjusted = apply_water_prior(probability, occurrence, setting)
    np.testing.assert_allclose(adjusted, [[0.8, 0.0, 0.8]], rtol=0, atol=1e-7)


def test_apply_water_prior_soft_linear_uses_occurrence_fraction():
    probability = np.array([[0.8, 0.8, 0.8]], dtype=np.float32)
    occurrence = np.array([[20, 95, 255]], dtype=np.uint8)
    setting = PriorSetting("soft_linear", occurrence_threshold=90, penalty_strength=0.5)
    adjusted = apply_water_prior(probability, occurrence, setting)
    expected_middle = 0.8 * (1.0 - 0.5 * 0.95)
    np.testing.assert_allclose(adjusted, [[0.8, expected_middle, 0.8]], rtol=0, atol=1e-7)


def test_build_prior_settings_includes_baseline_hard_and_soft():
    settings = build_prior_settings([90, 95], [0.5, 1.0])
    assert settings[0].strategy == "none"
    assert len(settings) == 1 + 2 + 4
    assert {setting.key for setting in settings} == {
        "none",
        "hard_occ090",
        "hard_occ095",
        "soft_occ090_strength0.50",
        "soft_occ090_strength1.00",
        "soft_occ095_strength0.50",
        "soft_occ095_strength1.00",
    }


def test_choose_best_setting_respects_recall_guard():
    sweep = pd.DataFrame(
        [
            {"setting_key": "none", "f1": 0.49, "iou": 0.32, "mcc": 0.45, "precision": 0.46, "recall": 0.54},
            {"setting_key": "aggressive", "f1": 0.52, "iou": 0.35, "mcc": 0.48, "precision": 0.70, "recall": 0.40},
            {"setting_key": "guarded", "f1": 0.50, "iou": 0.33, "mcc": 0.46, "precision": 0.49, "recall": 0.53},
        ]
    )
    reference = sweep.iloc[0].to_dict()
    unconstrained, guarded = _choose_best_setting(sweep, reference, max_recall_drop=0.02)
    assert unconstrained["setting_key"] == "aggressive"
    assert guarded["setting_key"] == "guarded"


def test_local_occurrence_asset_is_reprojected_to_target_grid(tmp_path):
    source_path = tmp_path / "source.tif"
    source_transform = from_origin(0.0, 2.0, 1.0, 1.0)
    source_values = np.array([[10, 20], [90, 100]], dtype=np.uint8)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=source_transform,
        nodata=255,
    ) as dst:
        dst.write(source_values, 1)

    item = SimpleNamespace(
        id="local-item",
        bbox=[0.0, 0.0, 2.0, 2.0],
        assets={"occurrence": SimpleNamespace(href=str(source_path))},
    )
    target = {
        "width": 2,
        "height": 2,
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": source_transform,
        "bounds_4326": (0.0, 0.0, 2.0, 2.0),
    }
    output_path = tmp_path / "aligned.tif"
    provider = PlanetaryComputerOccurrenceProvider(tmp_path / "cache")
    try:
        result = provider._write_aligned_prior(output_path, target, [item])
    finally:
        provider.close()

    assert result["valid_fraction"] == 1.0
    with rasterio.open(output_path) as src:
        np.testing.assert_array_equal(src.read(1), source_values)
        assert src.tags()["jrc_collection"] == "jrc-gsw"


def test_sweep_annotation_separates_threshold_tuning_from_prior_gain():
    sweep = pd.DataFrame(
        [
            {
                "strategy": "none",
                "setting_key": "none",
                "model_threshold": 0.50,
                "min_component_area": 96,
                "precision": 0.49,
                "recall": 0.386,
                "f1": 0.431,
                "iou": 0.275,
                "mcc": 0.378,
            },
            {
                "strategy": "none",
                "setting_key": "none",
                "model_threshold": 0.30,
                "min_component_area": 96,
                "precision": 0.445,
                "recall": 0.459,
                "f1": 0.452,
                "iou": 0.292,
                "mcc": 0.388,
            },
            {
                "strategy": "soft_linear",
                "setting_key": "soft_occ090_strength0.50",
                "model_threshold": 0.30,
                "min_component_area": 96,
                "precision": 0.446,
                "recall": 0.456,
                "f1": 0.451,
                "iou": 0.291,
                "mcc": 0.387,
            },
        ]
    )
    reference = sweep.iloc[0].to_dict()
    annotated = _annotate_sweep(sweep, reference, max_recall_drop=0.02)

    tuned_none = annotated[
        annotated["strategy"].eq("none")
        & np.isclose(annotated["model_threshold"], 0.30)
    ].iloc[0]
    prior = annotated[annotated["strategy"].ne("none")].iloc[0]

    assert bool(tuned_none["recall_guard_eligible"])
    assert tuned_none["f1_change_vs_reference"] > 0.02
    assert tuned_none["incremental_f1_gain_vs_same_threshold"] == 0.0
    assert prior["f1_change_vs_reference"] > 0.0
    assert prior["incremental_f1_gain_vs_same_threshold"] < 0.0


def test_prior_recommendation_does_not_count_threshold_only_gain():
    sweep = pd.DataFrame(
        [
            {
                "strategy": "none",
                "setting_key": "none",
                "model_threshold": 0.50,
                "min_component_area": 96,
                "precision": 0.49,
                "recall": 0.386,
                "f1": 0.431,
                "iou": 0.275,
                "mcc": 0.378,
            },
            {
                "strategy": "none",
                "setting_key": "none",
                "model_threshold": 0.30,
                "min_component_area": 96,
                "precision": 0.445,
                "recall": 0.459,
                "f1": 0.452,
                "iou": 0.292,
                "mcc": 0.388,
            },
            {
                "strategy": "soft_linear",
                "setting_key": "soft_occ090_strength0.50",
                "model_threshold": 0.30,
                "min_component_area": 96,
                "precision": 0.446,
                "recall": 0.456,
                "f1": 0.451,
                "iou": 0.291,
                "mcc": 0.387,
            },
        ]
    )
    reference = sweep.iloc[0].to_dict()
    annotated = _annotate_sweep(sweep, reference, max_recall_drop=0.02)
    _, best_none = _choose_best_strategy_setting(
        annotated,
        use_prior=False,
        reference=reference,
        max_recall_drop=0.02,
    )
    _, best_prior = _choose_best_strategy_setting(
        annotated,
        use_prior=True,
        reference=reference,
        max_recall_drop=0.02,
    )

    recommendation, gain, incremental = _water_prior_recommendation(
        best_prior,
        best_none,
    )

    assert recommendation == "do_not_use_water_prior_no_incremental_gain"
    assert gain < 0.0
    assert incremental < 0.0
