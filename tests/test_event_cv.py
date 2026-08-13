from pathlib import Path

import pytest
import torch

from floods.cli import build_cli_parser
from floods.config import TrainConfig
from floods.evaluation import EventMacroThresholdSweep
from floods.event_cv import _cv_result_is_complete, build_balanced_folds, run_event_cross_validation
from floods.resume import build_resume_signature


def test_balanced_event_folds_are_disjoint_and_deterministic():
    counts = {f"EMSR{i:03d}": value for i, value in enumerate([100, 80, 60, 40, 30, 20, 10], start=1)}
    folds_a = build_balanced_folds(counts, n_splits=3, seed=42)
    folds_b = build_balanced_folds(counts, n_splits=3, seed=42)
    assert folds_a == folds_b
    flattened = [event for fold in folds_a for event in fold]
    assert sorted(flattened) == sorted(counts)
    assert len(flattened) == len(set(flattened))
    totals = [sum(counts[event] for event in fold) for fold in folds_a]
    assert max(totals) - min(totals) <= max(counts.values())



def test_event_cv_completion_marker_validation_supports_successful_legacy_runs():
    assert _cv_result_is_complete({
        "result": {"stop_reason": "completed"},
    })
    assert _cv_result_is_complete({
        "result": {"stop_reason": "early_stopping"},
    })
    assert not _cv_result_is_complete({
        "result": {"stop_reason": "interrupted"},
    })
    assert not _cv_result_is_complete({
        "status": "completed",
        "result": {"stop_reason": "interrupted"},
    })

def test_event_macro_threshold_sweep_reports_macro_and_worst_event():
    meter = EventMacroThresholdSweep(["EMSR001", "EMSR002"], thresholds=[0.5], device="cpu")
    target = torch.tensor([
        [[1, 0], [0, 0]],
        [[1, 1], [0, 0]],
    ])
    logits = torch.tensor([
        [[10.0, -10.0], [-10.0, -10.0]],
        [[10.0, -10.0], [-10.0, -10.0]],
    ])
    meter.update(target, logits, torch.tensor([0, 1]))
    best = meter.best()
    assert best.threshold == 0.5
    assert abs(best.event_metrics[0]["f1"] - 1.0) < 1e-6
    assert abs(best.event_metrics[1]["f1"] - (2.0 / 3.0)) < 1e-6
    assert abs(best.macro_f1 - (5.0 / 6.0)) < 1e-6
    assert abs(best.worst_f1 - (2.0 / 3.0)) < 1e-6


def test_event_cv_cli_parses_architectures_and_fold_subset(tmp_path):
    parser = build_cli_parser()
    args = parser.parse_args([
        "event-cv",
        "--config", str(tmp_path / "config.yaml"),
        "--processed-data-dir", str(tmp_path / "processed"),
        "--output-dir", str(tmp_path / "out"),
        "--architectures", "unet:resnet34", "segformer:pvt_v2_b0",
        "--modality-sets", "vv+vh", "vv+vh+dem",
        "--folds", "5",
        "--fold-indices", "0", "2",
        "--no-amp",
        "--gpu",
    ])
    assert args.command == "event-cv"
    assert args.fold_indices == [0, 2]
    assert args.amp is False
    assert args.cpu is False
    assert args.architectures[-1] == "segformer:pvt_v2_b0"


def test_event_cv_plan_only_writes_leakage_free_fold_plan(tmp_path):
    mask_dir = tmp_path / "processed" / "train" / "mask"
    mask_dir.mkdir(parents=True)
    for event, count in {"EMSR001": 3, "EMSR002": 2, "EMSR003": 2, "EMSR004": 1}.items():
        for index in range(count):
            (mask_dir / f"{event}_tile_{index:03d}.tif").touch()
    result = run_event_cross_validation(
        TrainConfig(),
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "cv",
        architectures=["unet:resnet34"],
        modality_sets=["vv+vh"],
        folds=2,
        plan_only=True,
    )
    assert result["planned_runs"] == 2
    assert (tmp_path / "cv" / "folds.json").exists()
    assert (tmp_path / "cv" / "fold_assignments.csv").exists()


def test_event_cv_staged_invocations_preserve_completed_fold_rows(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import pandas as pd
    import floods.event_cv as event_cv_module

    mask_dir = tmp_path / "processed" / "train" / "mask"
    mask_dir.mkdir(parents=True)
    for event in ("EMSR001", "EMSR002", "EMSR003", "EMSR004"):
        (mask_dir / f"{event}_tile_000.tif").touch()

    def fake_fit_normalization_stats(*, output_file, **kwargs):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(json.dumps({"modalities": kwargs["input_modalities"]}))

    def fake_train(config):
        run_dir = Path(config.output_folder) / config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        held_out = list(config.data.val_include_events)
        pd.DataFrame([{
            "epoch": 1,
            "threshold": 0.5,
            "macro_f1": 0.4 + 0.01 * len(held_out),
            "macro_iou": 0.3,
            "worst_event_f1": 0.2,
            "mean_precision": 0.5,
            "mean_recall": 0.5,
        }]).to_csv(run_dir / "event_validation_history.csv", index=False)
        pd.DataFrame([{
            "epoch": 1,
            "event_id": event,
            "threshold": 0.5,
            "f1": 0.4,
            "iou": 0.3,
            "precision": 0.5,
            "recall": 0.5,
            "tp": 1,
            "fp": 1,
            "fn": 1,
        } for event in held_out]).to_csv(run_dir / "event_validation_metrics.csv", index=False)
        checkpoint = run_dir / "models" / "model-001_best_event_macro_f1-0.4000.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
        return {
            "best_epoch": 1,
            "best_checkpoint": str(checkpoint),
            "best_score": 0.4,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(event_cv_module, "fit_normalization_stats", fake_fit_normalization_stats)
    fake_training_module = types.ModuleType("floods.training")
    fake_training_module.train = fake_train
    monkeypatch.setitem(sys.modules, "floods.training", fake_training_module)

    common = dict(
        base_config=TrainConfig(),
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "cv",
        architectures=["unet:resnet34"],
        modality_sets=["vv+vh"],
        folds=2,
    )
    run_event_cross_validation(**common, fold_indices=[0])
    run_event_cross_validation(**common, fold_indices=[1])

    results = pd.read_csv(tmp_path / "cv" / "cv_results.csv")
    assert set(results["fold"]) == {0, 1}
    assert len(results) == 2
    summary = pd.read_csv(tmp_path / "cv" / "cv_summary.csv")
    assert int(summary.iloc[0]["folds_completed"]) == 2


def test_event_cv_rejects_reusing_output_dir_with_different_fold_plan(tmp_path):
    mask_dir = tmp_path / "processed" / "train" / "mask"
    mask_dir.mkdir(parents=True)
    for event in ("EMSR001", "EMSR002", "EMSR003", "EMSR004"):
        (mask_dir / f"{event}_tile_000.tif").touch()
    common = dict(
        base_config=TrainConfig(),
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "cv",
        architectures=["unet:resnet34"],
        modality_sets=["vv+vh"],
        folds=2,
        plan_only=True,
    )
    run_event_cross_validation(**common, seed=42)
    with pytest.raises(ValueError, match="different fold or normalization plan"):
        run_event_cross_validation(**common, seed=43)


def test_event_cv_resume_signature_tracks_event_partition_and_macro_selection():
    first = TrainConfig()
    first.data.train_source_split = "train"
    first.data.val_source_split = "train"
    first.data.train_exclude_events = ["EMSR342"]
    first.data.val_include_events = ["EMSR342"]
    first.trainer.event_macro_validation = True
    second = first.copy(deep=True)
    second.data.val_include_events = ["EMSR424"]
    assert build_resume_signature(first) != build_resume_signature(second)
    assert build_resume_signature(first)["trainer"]["event_macro_validation"] is True


def test_segformer_pvt_uses_all_four_feature_levels():
    from floods.models.decoders.segformer import SegFormerDecoder
    assert SegFormerDecoder.required_indices("pvt_v2_b0") == [0, 1, 2, 3]


def test_event_cv_architecture_validation_fails_before_normalization(tmp_path, monkeypatch):
    import floods.event_cv as event_cv_module
    mask_dir = tmp_path / "processed" / "train" / "mask"
    mask_dir.mkdir(parents=True)
    for event in ("EMSR001", "EMSR002"):
        (mask_dir / f"{event}_tile_000.tif").touch()
    called = {"normalization": False}
    def fail_if_called(**kwargs):
        called["normalization"] = True
    monkeypatch.setattr(event_cv_module, "fit_normalization_stats", fail_if_called)
    with pytest.raises(ValueError, match="not available in the installed timm version"):
        run_event_cross_validation(
            TrainConfig(),
            processed_data_dir=tmp_path / "processed",
            output_dir=tmp_path / "cv",
            architectures=["segformer:definitely_not_a_timm_model"],
            modality_sets=["vv+vh"],
            folds=2,
        )
    assert called["normalization"] is False


def test_event_cv_interruption_does_not_create_completion_marker(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import floods.event_cv as event_cv_module

    mask_dir = tmp_path / "processed" / "train" / "mask"
    mask_dir.mkdir(parents=True)
    for event in ("EMSR001", "EMSR002", "EMSR003", "EMSR004"):
        (mask_dir / f"{event}_tile_000.tif").touch()

    def fake_fit_normalization_stats(*, output_file, **kwargs):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(json.dumps({"modalities": kwargs["input_modalities"]}))

    calls = []

    def fake_train(config):
        calls.append(config.name)
        run_dir = Path(config.output_folder) / config.name
        models = run_dir / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "last.ckpt").touch()
        return {
            "best_epoch": 1,
            "best_checkpoint": None,
            "best_score": 0.3,
            "stop_reason": "interrupted",
        }

    monkeypatch.setattr(event_cv_module, "fit_normalization_stats", fake_fit_normalization_stats)
    fake_training_module = types.ModuleType("floods.training")
    fake_training_module.train = fake_train
    monkeypatch.setitem(sys.modules, "floods.training", fake_training_module)

    output_dir = tmp_path / "cv"
    with pytest.raises(KeyboardInterrupt):
        run_event_cross_validation(
            TrainConfig(),
            processed_data_dir=tmp_path / "processed",
            output_dir=output_dir,
            architectures=["unet:resnet34"],
            modality_sets=["vv+vh"],
            folds=2,
        )

    assert len(calls) == 1
    run_dir = output_dir / "runs" / calls[0]
    assert (run_dir / "models" / "last.ckpt").exists()
    assert not (run_dir / "cv_result.json").exists()
    assert not (output_dir / "cv_results.csv").exists()


def test_event_cv_repairs_legacy_interrupted_marker_and_resumes(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import pandas as pd
    import floods.event_cv as event_cv_module

    mask_dir = tmp_path / "processed" / "train" / "mask"
    mask_dir.mkdir(parents=True)
    for event in ("EMSR001", "EMSR002", "EMSR003", "EMSR004"):
        (mask_dir / f"{event}_tile_000.tif").touch()

    def fake_fit_normalization_stats(*, output_file, **kwargs):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(json.dumps({"modalities": kwargs["input_modalities"]}))

    output_dir = tmp_path / "cv"
    run_id = "eventcv_unet_resnet34_vv_vh_fold00"
    run_dir = output_dir / "runs" / run_id
    models = run_dir / "models"
    models.mkdir(parents=True)
    (models / "last.ckpt").touch()
    stale_row = {
        "run_id": run_id,
        "fold": 0,
        "candidate": "unet_resnet34",
        "architecture": "unet_resnet34",
        "weights_source": "imagenet",
        "provider": "builtin",
        "adapter": "timm",
        "decoder": "unet",
        "encoder": "resnet34",
        "modalities": "vv+vh",
        "normalization_mode": "train_fitted",
        "best_epoch": 1,
        "checkpoint": None,
        "event_macro_f1": 0.3,
        "event_macro_iou": 0.2,
        "worst_event_f1": 0.1,
        "threshold": 0.5,
        "stop_reason": "interrupted",
    }
    run_dir.joinpath("cv_result.json").write_text(
        json.dumps({"result": stale_row, "events": []}), encoding="utf-8"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stale_row]).to_csv(output_dir / "cv_results.csv", index=False)

    called = {"resume": False}

    def fake_train(config):
        called["resume"] = bool(config.resume)
        held_out = list(config.data.val_include_events)
        pd.DataFrame([{
            "epoch": 2,
            "threshold": 0.7,
            "macro_f1": 0.45,
            "macro_iou": 0.3,
            "worst_event_f1": 0.2,
            "mean_precision": 0.5,
            "mean_recall": 0.5,
        }]).to_csv(run_dir / "event_validation_history.csv", index=False)
        pd.DataFrame([{
            "epoch": 2,
            "event_id": event,
            "threshold": 0.7,
            "f1": 0.45,
            "iou": 0.3,
            "precision": 0.5,
            "recall": 0.5,
            "tp": 1,
            "fp": 1,
            "fn": 1,
        } for event in held_out]).to_csv(run_dir / "event_validation_metrics.csv", index=False)
        checkpoint = models / "model-002_best_event_macro_f1-0.4500.pth"
        checkpoint.touch()
        return {
            "best_epoch": 2,
            "best_checkpoint": str(checkpoint),
            "best_score": 0.45,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(event_cv_module, "fit_normalization_stats", fake_fit_normalization_stats)
    fake_training_module = types.ModuleType("floods.training")
    fake_training_module.train = fake_train
    monkeypatch.setitem(sys.modules, "floods.training", fake_training_module)

    run_event_cross_validation(
        TrainConfig(),
        processed_data_dir=tmp_path / "processed",
        output_dir=output_dir,
        architectures=["unet:resnet34"],
        modality_sets=["vv+vh"],
        folds=2,
        fold_indices=[0],
    )

    assert called["resume"] is True
    assert (run_dir / "cv_result.interrupted.json").exists()
    repaired = json.loads((run_dir / "cv_result.json").read_text(encoding="utf-8"))
    assert repaired["schema_version"] == 2
    assert repaired["status"] == "completed"
    assert repaired["result"]["stop_reason"] == "completed"
    results = pd.read_csv(output_dir / "cv_results.csv")
    assert len(results) == 1
    assert results.iloc[0]["event_macro_f1"] == pytest.approx(0.45)
