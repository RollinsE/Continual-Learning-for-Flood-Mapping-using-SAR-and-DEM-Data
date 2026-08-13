from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from floods.evaluation import BinaryThresholdSweep, DEFAULT_THRESHOLDS, load_checkpoint_state
from floods.utils.common import get_logger
from floods.utils.console import progress_iter

LOG = get_logger(__name__)


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, y, index


def _event_id(path: str | Path) -> str:
    match = re.search(r"(EMSR\d+)", Path(path).name, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def _filter_dataset_by_events(dataset: Any,
                              include_events: Optional[Iterable[str]] = None,
                              exclude_events: Optional[Iterable[str]] = None) -> None:
    include = {str(v).upper() for v in (include_events or [])}
    exclude = {str(v).upper() for v in (exclude_events or [])}
    if not include and not exclude:
        return
    keep: List[bool] = []
    for path in dataset.label_files:
        event = _event_id(path)
        selected = not include or event in include
        if event in exclude:
            selected = False
        keep.append(selected)
    before = len(dataset)
    dataset.add_mask(keep)
    LOG.info("Event filter: kept %d/%d samples (include=%s exclude=%s)", len(dataset), before, sorted(include), sorted(exclude))
    if len(dataset) == 0:
        raise ValueError("Event filter removed every sample")


def parse_ablation_spec(spec: str, modalities: Sequence[str]) -> Tuple[str, Tuple[int, ...], Tuple[str, ...]]:
    """Parse an ablation token such as ``none``, ``dem`` or ``vv+vh``.

    The returned label is canonical and channel indices follow the checkpoint's
    configured modality order. Inputs are zeroed *after* normalization, making
    zero the neutral normalized value rather than a raw-data zero.
    """
    ordered = [str(m).lower() for m in modalities]
    raw = str(spec).strip().lower().replace(",", "+").replace("_", "+")
    if raw in {"", "none", "baseline", "full", "all"}:
        return "none", tuple(), tuple()
    names = tuple(part.strip() for part in raw.split("+") if part.strip())
    if not names:
        return "none", tuple(), tuple()
    unknown = [name for name in names if name not in ordered]
    if unknown:
        raise ValueError(f"Unknown ablation modalities {unknown}; available modalities: {ordered}")
    deduped = tuple(dict.fromkeys(names))
    indices = tuple(ordered.index(name) for name in deduped)
    label = "+".join(deduped)
    return label, indices, deduped


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _tile_metrics(target: torch.Tensor, logits: torch.Tensor, threshold: float, ignore_index: int = 255) -> List[Dict[str, float]]:
    target = target.detach().cpu()
    logits = BinaryThresholdSweep._squeeze_logits(logits).detach().float().cpu()
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    valid = target != ignore_index
    truth = (target > 0) & valid
    pred = (torch.sigmoid(logits) >= float(threshold)) & valid
    rows: List[Dict[str, float]] = []
    for index in range(target.shape[0]):
        tp = float(torch.count_nonzero(pred[index] & truth[index]))
        fp = float(torch.count_nonzero(pred[index] & (~truth[index]) & valid[index]))
        tn = float(torch.count_nonzero((~pred[index]) & (~truth[index]) & valid[index]))
        fn = float(torch.count_nonzero((~pred[index]) & truth[index]))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        iou = _safe_div(tp, tp + fp + fn)
        rows.append({
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "true_pixels": tp + fn,
            "pred_pixels": tp + fp,
        })
    return rows


def _result_dict(result: Any) -> Dict[str, float]:
    return {
        "threshold": float(result.threshold),
        "f1": float(result.f1),
        "iou": float(result.iou),
        "precision": float(result.precision),
        "recall": float(result.recall),
        "mcc": float(result.mcc),
        "empty_fp": float(result.empty_tile_fp_rate),
        "nonempty_recall": float(result.nonempty_tile_recall),
        "tp": float(result.tp),
        "tn": float(result.tn),
        "fp": float(result.fp),
        "fn": float(result.fn),
    }


def _threshold_result(sweep: BinaryThresholdSweep, threshold: float) -> Any:
    results = sweep.compute()
    return min(results, key=lambda row: abs(float(row.threshold) - float(threshold)))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def audit_modality_ablation(config: Any,
                            checkpoint_path: Path,
                            output_dir: Path,
                            split: str = "val",
                            target_events: Optional[Iterable[str]] = None,
                            include_events: Optional[Iterable[str]] = None,
                            exclude_events: Optional[Iterable[str]] = None,
                            ablations: Optional[Sequence[str]] = None,
                            thresholds: Optional[Iterable[float]] = None,
                            threshold_metric: str = "f1",
                            operating_threshold: float = 0.50,
                            max_samples: Optional[int] = None) -> Dict[str, Any]:
    """Measure checkpoint dependence on each normalized input modality.

    Each ablation keeps the original three-channel architecture and checkpoint
    intact, then replaces selected normalized input channels with zero at
    inference time. This isolates immediate channel dependence without retraining.
    Results are reported for the complete evaluated split and, when requested,
    the target-event subset using both a shared operating threshold and an
    independently optimized threshold sweep.
    """
    from accelerate import Accelerator
    from floods.eval_collate import pad_segmentation_batch
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.utils.ml import seed_everything, seed_worker

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    seed_everything(config.seed, deterministic=True)

    threshold_values = sorted({float(v) for v in (thresholds or DEFAULT_THRESHOLDS)} | {float(operating_threshold)})
    target_set = {str(v).upper() for v in (target_events or [])}

    try:
        accelerator = Accelerator(mixed_precision="fp16" if config.trainer.amp else "no", cpu=config.trainer.cpu)
    except TypeError:
        accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)

    dataset, modalities, _ = prepare_evaluation_dataset(config, split=split)
    _filter_dataset_by_events(dataset, include_events=include_events, exclude_events=exclude_events)
    if max_samples is not None and int(max_samples) > 0 and len(dataset) > int(max_samples):
        dataset.add_mask([index < int(max_samples) for index in range(len(dataset))])
        LOG.info("Sample cap: kept first %d tiles", len(dataset))

    event_ids = [_event_id(path) for path in dataset.label_files]
    file_names = [Path(path).name for path in dataset.label_files]
    target_count = sum(event in target_set for event in event_ids) if target_set else 0
    if target_set and target_count == 0:
        raise ValueError(f"No target-event tiles found for {sorted(target_set)}")

    parsed: List[Tuple[str, Tuple[int, ...], Tuple[str, ...]]] = []
    seen = set()
    for token in (ablations or ["none", *modalities]):
        item = parse_ablation_spec(token, modalities)
        if item[0] not in seen:
            parsed.append(item)
            seen.add(item[0])
    if "none" not in seen:
        parsed.insert(0, parse_ablation_spec("none", modalities))

    loader = DataLoader(
        _IndexedDataset(dataset),
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        worker_init_fn=seed_worker,
        collate_fn=pad_segmentation_batch,
    )
    model = prepare_model(config=config, num_classes=1, stage="eval")
    state = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state, strict=not config.model.multibranch)
    model = model.to(accelerator.device)
    model, loader = accelerator.prepare(model, loader)
    model.eval()

    sweeps: Dict[Tuple[str, str], BinaryThresholdSweep] = {}
    for label, _, _ in parsed:
        sweeps[("all", label)] = BinaryThresholdSweep(threshold_values, device=accelerator.device)
        if target_set:
            sweeps[("target", label)] = BinaryThresholdSweep(threshold_values, device=accelerator.device)

    tile_rows: List[Dict[str, Any]] = []
    LOG.info("Modality ablation audit | split=%s samples=%d target_events=%s target_tiles=%d", split, len(dataset), sorted(target_set), target_count)
    LOG.info("Checkpoint modalities=%s | ablations=%s | normalized replacement value=0.0", modalities, [item[0] for item in parsed])

    with torch.no_grad():
        for x, y, indices in progress_iter(loader, desc=f"Modality ablation {split}", unit="batch", colour="green"):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            gathered_indices = accelerator.gather(indices).detach().cpu().tolist()
            gathered_y = accelerator.gather(y)
            batch_target_mask_cpu = torch.tensor([event_ids[int(i)] in target_set for i in gathered_indices], dtype=torch.bool)

            for label, channel_indices, dropped_names in parsed:
                ablated = x if not channel_indices else x.clone()
                for channel_index in channel_indices:
                    ablated[:, int(channel_index)] = 0.0
                with accelerator.autocast():
                    output = model(ablated)
                logits = BinaryThresholdSweep._main_prediction(output)
                gathered_logits = accelerator.gather(logits)
                sweeps[("all", label)].update(gathered_y, gathered_logits)

                if target_set and batch_target_mask_cpu.any():
                    mask_device = batch_target_mask_cpu.to(gathered_y.device)
                    sweeps[("target", label)].update(gathered_y[mask_device], gathered_logits[mask_device])

                batch_metrics = _tile_metrics(gathered_y, gathered_logits, operating_threshold)
                for local_index, metrics in enumerate(batch_metrics):
                    dataset_index = int(gathered_indices[local_index])
                    tile_rows.append({
                        "file": file_names[dataset_index],
                        "event": event_ids[dataset_index],
                        "is_target_event": bool(event_ids[dataset_index] in target_set),
                        "ablation": label,
                        "dropped_modalities": "+".join(dropped_names),
                        "operating_threshold": float(operating_threshold),
                        **metrics,
                    })

    summary_rows: List[Dict[str, Any]] = []
    scopes = ["all"] + (["target"] if target_set else [])
    for scope in scopes:
        samples = len(dataset) if scope == "all" else target_count
        for label, _, dropped_names in parsed:
            sweep = sweeps[(scope, label)]
            best = sweep.best(threshold_metric)
            fixed = _threshold_result(sweep, operating_threshold)
            summary_rows.append({
                "scope": scope,
                "events": "ALL" if scope == "all" else "+".join(sorted(target_set)),
                "samples": int(samples),
                "ablation": label,
                "dropped_modalities": "+".join(dropped_names),
                "best_metric": threshold_metric,
                **{f"best_{key}": value for key, value in _result_dict(best).items()},
                **{f"operating_{key}": value for key, value in _result_dict(fixed).items()},
            })

    summary_df = pd.DataFrame(summary_rows)
    baseline_by_scope = {
        scope: summary_df[(summary_df["scope"] == scope) & (summary_df["ablation"] == "none")].iloc[0]
        for scope in scopes
    }
    delta_rows: List[Dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        base = baseline_by_scope[str(row["scope"])]
        delta_rows.append({
            **row.to_dict(),
            "delta_best_f1": float(row["best_f1"] - base["best_f1"]),
            "delta_best_recall": float(row["best_recall"] - base["best_recall"]),
            "delta_best_precision": float(row["best_precision"] - base["best_precision"]),
            "delta_operating_f1": float(row["operating_f1"] - base["operating_f1"]),
            "delta_operating_recall": float(row["operating_recall"] - base["operating_recall"]),
            "delta_operating_precision": float(row["operating_precision"] - base["operating_precision"]),
        })
    delta_df = pd.DataFrame(delta_rows)
    tile_df = pd.DataFrame(tile_rows)

    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    delta_df.to_csv(output_dir / "ablation_deltas.csv", index=False)
    tile_df.to_csv(output_dir / "tile_ablation_metrics.csv", index=False)

    target_diagnosis = None
    if target_set:
        target_nonbaseline = delta_df[(delta_df["scope"] == "target") & (delta_df["ablation"] != "none")]
        if not target_nonbaseline.empty:
            best_row = target_nonbaseline.sort_values(["delta_best_f1", "delta_operating_recall"], ascending=False).iloc[0]
            target_diagnosis = {
                "most_beneficial_ablation": str(best_row["ablation"]),
                "delta_best_f1": float(best_row["delta_best_f1"]),
                "delta_operating_f1": float(best_row["delta_operating_f1"]),
                "delta_operating_recall": float(best_row["delta_operating_recall"]),
            }

    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "split": str(split),
        "samples": int(len(dataset)),
        "target_events": sorted(target_set),
        "target_samples": int(target_count),
        "modalities": list(modalities),
        "ablation_semantics": "selected normalized channels replaced with 0.0; model weights and architecture unchanged",
        "operating_threshold": float(operating_threshold),
        "threshold_metric": str(threshold_metric),
        "thresholds": threshold_values,
        "results": delta_df.to_dict(orient="records"),
        "target_diagnosis": target_diagnosis,
    }
    _write_json(output_dir / "summary.json", payload)

    for scope in scopes:
        ranked = delta_df[delta_df["scope"] == scope].sort_values("best_f1", ascending=False)
        best_row = ranked.iloc[0]
        LOG.info(
            "Ablation result | scope=%s | best=%s | f1=%.4f threshold=%.2f | precision=%.4f recall=%.4f | delta_vs_full=%+.4f",
            scope,
            best_row["ablation"],
            best_row["best_f1"],
            best_row["best_threshold"],
            best_row["best_precision"],
            best_row["best_recall"],
            best_row["delta_best_f1"],
        )
    LOG.info("Modality ablation outputs written to: %s", output_dir)
    return payload
