from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floods.hysteresis_audit import (
    HysteresisSetting,
    annotate_hysteresis_sweep,
    apply_hysteresis_threshold,
    apply_setting,
    build_hysteresis_settings,
    hysteresis_recommendation,
)


def test_hysteresis_grows_only_low_component_with_high_seed() -> None:
    probability = np.zeros((6, 8), dtype=np.float32)
    probability[1:4, 1:4] = 0.35
    probability[2, 2] = 0.80
    probability[1:4, 5:7] = 0.35
    valid = np.ones_like(probability, dtype=bool)

    prediction = apply_hysteresis_threshold(
        probability,
        valid,
        low_threshold=0.30,
        high_threshold=0.70,
        min_seed_pixels=1,
    )

    assert prediction[1:4, 1:4].all()
    assert not prediction[1:4, 5:7].any()
    assert int(prediction.sum()) == 9


def test_hysteresis_respects_minimum_seed_pixels() -> None:
    probability = np.full((4, 4), 0.35, dtype=np.float32)
    probability[0, 0] = 0.90
    probability[0, 1] = 0.85
    valid = np.ones_like(probability, dtype=bool)

    kept = apply_hysteresis_threshold(
        probability,
        valid,
        low_threshold=0.30,
        high_threshold=0.80,
        min_seed_pixels=2,
    )
    dropped = apply_hysteresis_threshold(
        probability,
        valid,
        low_threshold=0.30,
        high_threshold=0.80,
        min_seed_pixels=3,
    )

    assert kept.all()
    assert not dropped.any()


def test_fixed_setting_matches_global_threshold() -> None:
    probability = np.array([[0.2, 0.5], [0.7, np.nan]], dtype=np.float32)
    valid = np.array([[True, True], [False, True]])
    setting = HysteresisSetting("fixed", 0.5, 0.5, 0)
    prediction = apply_setting(probability, valid, setting)
    expected = np.array([[False, True], [False, False]])
    assert np.array_equal(prediction, expected)


def test_build_settings_rejects_invalid_seed_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_hysteresis_settings([0.5], [0.3], [0.6], [0])


def test_annotation_and_recommendation_separate_threshold_gain() -> None:
    rows = [
        {
            "strategy": "fixed",
            "setting_key": "fixed_thr0.300",
            "low_threshold": 0.3,
            "high_threshold": 0.3,
            "min_seed_pixels": 0,
            "min_component_area": 96,
            "precision": 0.44,
            "recall": 0.46,
            "f1": 0.45,
            "iou": 0.29,
            "mcc": 0.38,
            "empty_fp_rate": 0.80,
        },
        {
            "strategy": "fixed",
            "setting_key": "fixed_thr0.500",
            "low_threshold": 0.5,
            "high_threshold": 0.5,
            "min_seed_pixels": 0,
            "min_component_area": 96,
            "precision": 0.49,
            "recall": 0.39,
            "f1": 0.43,
            "iou": 0.27,
            "mcc": 0.37,
            "empty_fp_rate": 0.72,
        },
        {
            "strategy": "hysteresis",
            "setting_key": "hyst_low0.300_high0.500_seed16",
            "low_threshold": 0.3,
            "high_threshold": 0.5,
            "min_seed_pixels": 16,
            "min_component_area": 96,
            "precision": 0.46,
            "recall": 0.45,
            "f1": 0.455,
            "iou": 0.295,
            "mcc": 0.39,
            "empty_fp_rate": 0.72,
        },
    ]
    sweep = pd.DataFrame(rows)
    reference = sweep.iloc[1].to_dict()
    annotated = annotate_hysteresis_sweep(
        sweep,
        reference,
        max_recall_drop=0.02,
        max_empty_fp_rate_increase=0.0,
    )
    hysteresis = annotated[annotated["strategy"].eq("hysteresis")].iloc[0]
    assert hysteresis["incremental_f1_gain_vs_best_endpoint"] == pytest.approx(0.005)
    recommendation, gain, endpoint_gain = hysteresis_recommendation(
        hysteresis.to_dict(),
        annotated[annotated["setting_key"].eq("fixed_thr0.300")].iloc[0].to_dict(),
    )
    assert recommendation == "proceed_to_full_validation_hysteresis_audit"
    assert gain == pytest.approx(0.005)
    assert endpoint_gain == pytest.approx(0.005)
