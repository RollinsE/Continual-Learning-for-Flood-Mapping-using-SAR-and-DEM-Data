from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from floods.config import TrainConfig
from floods.eval_collate import pad_segmentation_batch
from floods.evaluation import BinaryThresholdSweep, EventMacroThresholdSweep, load_checkpoint_state
from floods.group_dro import EventIndexedDataset
from floods.utils.common import get_logger
from floods.utils.console import progress_iter

LOG = get_logger(__name__)


def _load_trusted_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = yaml.load(text, Loader=yaml.FullLoader)
    return data or {}


def _normalise_modalities(value: str | Sequence[str]) -> str:
    if isinstance(value, str):
        tokens = [token.strip().lower() for token in value.replace(",", "+").split("+") if token.strip()]
    else:
        tokens = [str(token).strip().lower() for token in value if str(token).strip()]
    return "+".join(tokens)


def _normalise_thresholds(values: Iterable[float]) -> list[float]:
    thresholds = sorted({round(float(value), 6) for value in values})
    if not thresholds:
        raise ValueError("At least one calibration threshold is required")
    invalid = [value for value in thresholds if not 0.0 < value < 1.0]
    if invalid:
        raise ValueError(f"Calibration thresholds must be strictly between 0 and 1: {invalid}")
    return thresholds


def select_cv_runs(
    frame: pd.DataFrame,
    *,
    candidate: str | None = None,
    architecture: str | None = None,
    modalities: str,
    fold_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    required = {"run_id", "fold", "held_out_events", "modalities", "checkpoint"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"cv_results.csv is missing required columns: {missing}")

    modality_key = _normalise_modalities(modalities)
    selected = frame[frame["modalities"].astype(str).map(_normalise_modalities) == modality_key].copy()
    selector = candidate or architecture
    if not selector:
        raise ValueError("Calibration requires --candidate (or legacy --architecture)")
    selector = str(selector)
    if candidate:
        masks = []
        if "candidate" in selected.columns:
            masks.append(selected["candidate"].astype(str) == selector)
        if "weights_source" in selected.columns:
            masks.append(selected["weights_source"].astype(str) == selector)
        if not masks:
            raise ValueError("cv_results.csv has no candidate or weights_source column")
        mask = masks[0]
        for value in masks[1:]:
            mask = mask | value
        selected = selected[mask]
    else:
        if "architecture" not in selected.columns:
            raise ValueError("Legacy --architecture requires an architecture column in cv_results.csv")
        selected = selected[selected["architecture"].astype(str) == selector]

    if fold_indices is not None:
        wanted = {int(value) for value in fold_indices}
        selected = selected[selected["fold"].astype(int).isin(wanted)]
    selected = selected.drop_duplicates(subset=["fold"], keep="last").sort_values("fold")
    if selected.empty:
        columns = [value for value in ("candidate", "weights_source", "architecture", "modalities") if value in frame.columns]
        available = frame[columns].drop_duplicates().to_dict(orient="records")
        raise ValueError(
            f"No completed CV runs match selector={selector!r} modalities={modality_key!r}. "
            f"Available combinations: {available}"
        )
    if fold_indices is not None:
        found = {int(value) for value in selected["fold"].tolist()}
        missing_folds = sorted({int(value) for value in fold_indices} - found)
        if missing_folds:
            raise ValueError(f"Requested folds have no completed matching run: {missing_folds}")
    return selected


def aggregate_calibration_rows(
    fold_rows: Sequence[dict],
    event_rows: Sequence[dict],
) -> pd.DataFrame:
    fold_frame = pd.DataFrame(fold_rows)
    event_frame = pd.DataFrame(event_rows)
    if fold_frame.empty or event_frame.empty:
        raise ValueError("Calibration aggregation requires fold and event rows")

    records: list[dict] = []
    for threshold in sorted(event_frame["threshold"].astype(float).unique()):
        events = event_frame[event_frame["threshold"].astype(float) == float(threshold)]
        folds = fold_frame[fold_frame["threshold"].astype(float) == float(threshold)]
        tp = float(folds["tp"].sum())
        tn = float(folds["tn"].sum())
        fp = float(folds["fp"].sum())
        fn = float(folds["fn"].sum())
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-12)
        iou = tp / max(tp + fp + fn, 1e-12)
        denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
        mcc = ((tp * tn) - (fp * fn)) / denominator
        empty_tiles = float(folds["empty_tiles"].sum())
        empty_tile_fp = float(folds["empty_tile_fp"].sum())
        nonempty_tiles = float(folds["nonempty_tiles"].sum())
        nonempty_detected = float(folds["nonempty_tile_detected"].sum())
        records.append({
            "threshold": float(threshold),
            "events": int(events["event_id"].nunique()),
            "folds": int(folds["fold"].nunique()),
            "event_macro_f1": float(events["f1"].mean()),
            "event_macro_iou": float(events["iou"].mean()),
            "worst_event_f1": float(events["f1"].min()),
            "mean_event_precision": float(events["precision"].mean()),
            "mean_event_recall": float(events["recall"].mean()),
            "global_f1": f1,
            "global_iou": iou,
            "global_precision": precision,
            "global_recall": recall,
            "global_mcc": mcc,
            "empty_tile_fp_rate": empty_tile_fp / max(empty_tiles, 1e-12),
            "nonempty_tile_recall": nonempty_detected / max(nonempty_tiles, 1e-12),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })
    return pd.DataFrame(records).sort_values("threshold").reset_index(drop=True)


def _find_run_config(cv_dir: Path, run_id: str) -> Path:
    path = Path(cv_dir) / "runs" / str(run_id) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Saved CV run configuration not found: {path}")
    return path


def _resolve_checkpoint(cv_dir: Path, row: pd.Series) -> Path:
    direct = Path(str(row["checkpoint"]))
    if direct.exists():
        return direct
    run_dir = Path(cv_dir) / "runs" / str(row["run_id"]) / "models"
    candidates = sorted(run_dir.glob("*best_event_macro_f1*.pth"))
    if not candidates:
        candidates = sorted(run_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"Checkpoint from cv_results.csv does not exist ({direct}) and no replacement was found under {run_dir}"
        )
    LOG.warning("Checkpoint path was stale; using discovered checkpoint: %s", candidates[-1])
    return candidates[-1]


def run_event_cv_calibration(
    *,
    cv_dir: Path,
    processed_data_dir: Path,
    output_dir: Path,
    candidate: str | None = None,
    architecture: str | None = None,
    modalities: str = "vv+vh",
    fold_indices: Sequence[int] | None = None,
    thresholds: Sequence[float],
    batch_size: int | None = None,
    num_workers: int | None = None,
    amp: bool = False,
    cpu: bool = False,
) -> dict:
    from accelerate import Accelerator
    from floods.evaluation import _filter_dataset_by_events
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.utils.ml import seed_everything, seed_worker

    cv_dir = Path(cv_dir)
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_values = _normalise_thresholds(thresholds)

    results_path = cv_dir / "cv_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Event-CV results not found: {results_path}")
    selected = select_cv_runs(
        pd.read_csv(results_path),
        candidate=candidate,
        architecture=architecture,
        modalities=modalities,
        fold_indices=fold_indices,
    )
    LOG.info(
        "OOF threshold calibration plan: candidate=%s | modalities=%s | folds=%s | thresholds=%d",
        candidate or architecture,
        _normalise_modalities(modalities),
        [int(value) for value in selected["fold"].tolist()],
        len(threshold_values),
    )

    try:
        accelerator = Accelerator(mixed_precision="fp16" if amp else "no", cpu=cpu)
    except TypeError:
        accelerator = Accelerator(fp16=amp, cpu=cpu)

    fold_rows: list[dict] = []
    event_rows: list[dict] = []
    for _, row in selected.iterrows():
        fold = int(row["fold"])
        run_id = str(row["run_id"])
        held_out = [value.strip() for value in str(row["held_out_events"]).split(",") if value.strip()]
        config_path = _find_run_config(cv_dir, run_id)
        checkpoint = _resolve_checkpoint(cv_dir, row)
        config = TrainConfig(**_load_trusted_yaml(config_path))
        config.data.path = str(processed_data_dir)
        config.model.pretrained = False
        config.trainer.amp = bool(amp)
        config.trainer.cpu = bool(cpu)
        if batch_size is not None:
            config.trainer.batch_size = int(batch_size)
        if num_workers is not None:
            config.trainer.num_workers = int(num_workers)
        config.visualize = False
        seed_everything(config.seed, deterministic=True)

        dataset, _, _ = prepare_evaluation_dataset(config, split="train")
        _filter_dataset_by_events(dataset, include_events=held_out)
        indexed = EventIndexedDataset(dataset, require_multiple=False)
        observed = set(indexed.event_names)
        if observed != set(held_out):
            raise ValueError(
                f"Held-out event mismatch for fold {fold}: expected={sorted(held_out)} observed={sorted(observed)}"
            )
        loader = DataLoader(
            dataset=indexed,
            batch_size=config.trainer.batch_size,
            shuffle=False,
            num_workers=config.trainer.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=pad_segmentation_batch,
        )
        model = prepare_model(config=config, num_classes=1, stage="eval")
        model.load_state_dict(load_checkpoint_state(checkpoint), strict=not config.model.multibranch)
        model = model.to(accelerator.device)
        model, loader = accelerator.prepare(model, loader)
        global_sweep = BinaryThresholdSweep(thresholds=threshold_values, device=accelerator.device)
        event_sweep = EventMacroThresholdSweep(
            event_names=indexed.event_names,
            thresholds=threshold_values,
            device=accelerator.device,
        )

        LOG.info(
            "Calibrating fold %d | run=%s | checkpoint=%s | events=%d | tiles=%d",
            fold,
            run_id,
            checkpoint,
            len(indexed.event_names),
            len(indexed),
        )
        model.eval()
        with torch.no_grad():
            for batch in progress_iter(
                loader,
                desc=f"OOF calibration fold {fold}",
                unit="batch",
                colour="green",
            ):
                x, y, event_indices = batch
                x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
                with accelerator.autocast():
                    logits = BinaryThresholdSweep._main_prediction(model(x))
                y_true = accelerator.gather(y)
                y_pred = accelerator.gather(logits)
                groups = accelerator.gather(event_indices)
                global_sweep.update(y_true, y_pred)
                event_sweep.update(y_true, y_pred, groups)

        global_results = global_sweep.compute()
        event_results = event_sweep.compute()
        for threshold_idx, (global_result, event_result) in enumerate(zip(global_results, event_results)):
            fold_rows.append({
                "run_id": run_id,
                "fold": fold,
                "checkpoint": str(checkpoint),
                "threshold": global_result.threshold,
                "event_macro_f1": event_result.macro_f1,
                "event_macro_iou": event_result.macro_iou,
                "worst_event_f1": event_result.worst_f1,
                "global_f1": global_result.f1,
                "global_iou": global_result.iou,
                "global_precision": global_result.precision,
                "global_recall": global_result.recall,
                "global_mcc": global_result.mcc,
                "tp": global_result.tp,
                "tn": global_result.tn,
                "fp": global_result.fp,
                "fn": global_result.fn,
                "empty_tiles": float(global_sweep.empty_tiles[threshold_idx].detach().cpu()),
                "empty_tile_fp": float(global_sweep.empty_tile_fp[threshold_idx].detach().cpu()),
                "nonempty_tiles": float(global_sweep.nonempty_tiles[threshold_idx].detach().cpu()),
                "nonempty_tile_detected": float(global_sweep.nonempty_tile_detected[threshold_idx].detach().cpu()),
            })
            for event in event_result.event_metrics:
                event_rows.append({
                    "run_id": run_id,
                    "fold": fold,
                    **event,
                })
        fold_best = max(event_results, key=lambda item: (item.macro_f1, item.worst_f1, -item.threshold))
        LOG.info(
            "Fold %d calibration best | threshold=%.2f | macro_f1=%.4f | worst_event_f1=%.4f",
            fold,
            fold_best.threshold,
            fold_best.macro_f1,
            fold_best.worst_f1,
        )
        del model, loader, global_sweep, event_sweep
        if hasattr(accelerator, "free_memory"):
            accelerator.free_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_frame = pd.DataFrame(fold_rows)
    event_frame = pd.DataFrame(event_rows)
    summary = aggregate_calibration_rows(fold_rows, event_rows)
    best_index = max(
        range(len(summary)),
        key=lambda index: (
            float(summary.iloc[index]["event_macro_f1"]),
            float(summary.iloc[index]["worst_event_f1"]),
            float(summary.iloc[index]["global_f1"]),
            -float(summary.iloc[index]["threshold"]),
        ),
    )
    best = summary.iloc[best_index].to_dict()

    fold_frame.to_csv(output_dir / "calibration_fold_results.csv", index=False)
    event_frame.to_csv(output_dir / "calibration_event_results.csv", index=False)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    payload = {
        "schema_version": 1,
        "cv_dir": str(cv_dir),
        "processed_data_dir": str(processed_data_dir),
        "candidate": candidate or architecture,
        "architecture": architecture,
        "modalities": _normalise_modalities(modalities),
        "folds": [int(value) for value in selected["fold"].tolist()],
        "thresholds": threshold_values,
        "selection_metric": "out_of_fold_event_macro_f1",
        "tie_breakers": ["worst_event_f1", "global_f1", "lower_threshold"],
        "recommended_threshold": float(best["threshold"]),
        "best": best,
    }
    (output_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOG.info(
        "OOF calibration best | threshold=%.2f | event_macro_f1=%.4f | worst_event_f1=%.4f | "
        "global_f1=%.4f | empty_fp=%.4f | nonempty_recall=%.4f",
        float(best["threshold"]),
        float(best["event_macro_f1"]),
        float(best["worst_event_f1"]),
        float(best["global_f1"]),
        float(best["empty_tile_fp_rate"]),
        float(best["nonempty_tile_recall"]),
    )
    LOG.info(
        "OOF calibration full table:\n%s",
        summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        extra={"floodmap_file_only": True},
    )
    if best_index == len(summary) - 1:
        LOG.warning("Recommended threshold is the highest tested value; calibration is boundary-limited.")
    LOG.info("Calibration outputs written to: %s", output_dir)
    return payload
