from pathlib import Path

import pandas as pd
import pytest

from floods.cli import build_cli_parser
from floods.event_cv_calibration import aggregate_calibration_rows, select_cv_runs


def test_calibration_cli_parses_oof_inputs(tmp_path):
    parser = build_cli_parser()
    args = parser.parse_args([
        "calibrate-event-cv",
        "--cv-dir", str(tmp_path / "cv"),
        "--processed-data-dir", str(tmp_path / "processed"),
        "--output-dir", str(tmp_path / "calibration"),
        "--architecture", "unet_resnet34",
        "--modalities", "vv+vh",
        "--thresholds", "0.70", "0.80", "0.90",
        "--no-amp",
        "--gpu",
    ])
    assert args.command == "calibrate-event-cv"
    assert args.thresholds == [0.7, 0.8, 0.9]
    assert args.amp is False
    assert args.cpu is False


def test_select_cv_runs_deduplicates_folds_and_filters_configuration():
    frame = pd.DataFrame([
        {"run_id": "old", "fold": 0, "held_out_events": "EMSR001", "architecture": "unet_resnet34", "modalities": "vv+vh", "checkpoint": "a"},
        {"run_id": "new", "fold": 0, "held_out_events": "EMSR001", "architecture": "unet_resnet34", "modalities": "vv+vh", "checkpoint": "b"},
        {"run_id": "fold1", "fold": 1, "held_out_events": "EMSR002", "architecture": "unet_resnet34", "modalities": "vv+vh", "checkpoint": "c"},
        {"run_id": "dem", "fold": 0, "held_out_events": "EMSR001", "architecture": "unet_resnet34", "modalities": "vv+vh+dem", "checkpoint": "d"},
    ])
    selected = select_cv_runs(frame, architecture="unet_resnet34", modalities="vv+vh")
    assert selected["run_id"].tolist() == ["new", "fold1"]


def test_select_cv_runs_rejects_missing_requested_fold():
    frame = pd.DataFrame([
        {"run_id": "fold0", "fold": 0, "held_out_events": "EMSR001", "architecture": "unet_resnet34", "modalities": "vv+vh", "checkpoint": "a"},
    ])
    with pytest.raises(ValueError, match="Requested folds"):
        select_cv_runs(frame, architecture="unet_resnet34", modalities="vv+vh", fold_indices=[0, 1])


def test_aggregate_calibration_rows_uses_event_macro_and_pooled_counts():
    fold_rows = [
        {"fold": 0, "threshold": 0.8, "tp": 8, "tn": 80, "fp": 2, "fn": 2, "empty_tiles": 5, "empty_tile_fp": 1, "nonempty_tiles": 5, "nonempty_tile_detected": 4},
        {"fold": 1, "threshold": 0.8, "tp": 4, "tn": 40, "fp": 1, "fn": 6, "empty_tiles": 5, "empty_tile_fp": 2, "nonempty_tiles": 5, "nonempty_tile_detected": 3},
        {"fold": 0, "threshold": 0.9, "tp": 6, "tn": 81, "fp": 1, "fn": 4, "empty_tiles": 5, "empty_tile_fp": 1, "nonempty_tiles": 5, "nonempty_tile_detected": 3},
        {"fold": 1, "threshold": 0.9, "tp": 3, "tn": 41, "fp": 0, "fn": 7, "empty_tiles": 5, "empty_tile_fp": 0, "nonempty_tiles": 5, "nonempty_tile_detected": 2},
    ]
    event_rows = [
        {"event_id": "EMSR001", "threshold": 0.8, "f1": 0.8, "iou": 0.7, "precision": 0.8, "recall": 0.8},
        {"event_id": "EMSR002", "threshold": 0.8, "f1": 0.4, "iou": 0.3, "precision": 0.5, "recall": 0.4},
        {"event_id": "EMSR001", "threshold": 0.9, "f1": 0.7, "iou": 0.6, "precision": 0.9, "recall": 0.6},
        {"event_id": "EMSR002", "threshold": 0.9, "f1": 0.3, "iou": 0.2, "precision": 1.0, "recall": 0.2},
    ]
    summary = aggregate_calibration_rows(fold_rows, event_rows)
    first = summary.iloc[0]
    assert first["event_macro_f1"] == pytest.approx(0.6)
    assert first["worst_event_f1"] == pytest.approx(0.4)
    assert first["empty_tile_fp_rate"] == pytest.approx(0.3)
    assert first["nonempty_tile_recall"] == pytest.approx(0.7)


def test_saved_provider_normalization_configs_can_be_reloaded():
    from floods.config import TrainConfig

    terramind = TrainConfig(
        data={
            "input_modalities": ["vv", "vh", "dem"],
            "normalization_mode": "terramind_v1",
            "source_sar_transform": "auto",
        },
        model={
            "weights_source": "terramind_v1_tiny",
            "encoder": "terramind_v1_tiny",
            "decoder": "segformer",
            "pretrained": False,
        },
    )
    ssl4eo = TrainConfig(
        data={
            "input_modalities": ["vv", "vh"],
            "normalization_mode": "ssl4eo_s1",
            "source_sar_transform": "auto",
        },
        model={
            "weights_source": "ssl4eo_s1_moco",
            "encoder": "resnet50",
            "decoder": "unet",
            "pretrained": False,
        },
    )

    assert terramind.data.normalization_mode == "terramind_v1"
    assert ssl4eo.data.normalization_mode == "ssl4eo_s1"
