from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from floods.utils.console import progress_iter

from floods.error_audit import (IndexedDataset, _filter_dataset_by_events, _event_id_from_name, _tile_offsets_from_name,
    _foreground_bin, _safe_div, _metrics_from_counts, _remove_small_components, _select_overlay_rows,
    _aggregate_metrics, _sweep_rows_for_sample, _write_overlay, _json_default, _normalization_mode)
from floods.evaluation import BinaryThresholdSweep, DEFAULT_THRESHOLDS, load_checkpoint_state
from floods.utils.common import get_logger
from floods.sliding_window import ensemble_sliding_window_logits
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import numpy as np
import pandas as pd

LOG = get_logger(__name__)


def _main_logits(output: Any) -> torch.Tensor:
    logits = BinaryThresholdSweep._main_prediction(output)
    if logits.ndim == 4 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 4 and logits.shape[1] > 1:
        return logits[:, 1]
    return logits


def _prob_to_logits(prob: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(prob.dtype).eps
    prob = prob.clamp(min=eps, max=1.0 - eps)
    return torch.log(prob / (1.0 - prob))


def ensemble_error_audit_checkpoints(configs: Sequence[Any],
                                     checkpoint_paths: Sequence[Path],
                                     output_dir: Path,
                                     split: str = "val",
                                     thresholds: Optional[Iterable[float]] = None,
                                     threshold: float = 0.5,
                                     threshold_metric: str = "f1",
                                     min_component_area: int = 0,
                                     sweep_component_areas: Optional[Iterable[int]] = None,
                                     max_overlays_per_category: int = 12,
                                     max_samples: Optional[int] = None,
                                     include_events: Optional[Iterable[str]] = None,
                                     exclude_events: Optional[Iterable[str]] = None,
                                     ensemble_method: str = "mean_logit",
                                     inference_mode: str = "direct",
                                     window_size: int = 256,
                                     window_overlap: int = 64,
                                     window_batch_size: int = 4) -> Dict[str, Any]:
    """Audit an ensemble at a fixed operating threshold and write CSV/PNG artifacts."""
    from floods.datasets.flood import FloodDataset, RGBFloodDataset
    from floods.normalization import describe_stats, load_normalization_stats
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.utils.ml import seed_everything, seed_worker
    from floods.eval_collate import pad_segmentation_batch

    if len(configs) != len(checkpoint_paths):
        raise ValueError("configs and checkpoint_paths must have the same length")
    if not configs:
        raise ValueError("At least one ensemble member is required")
    method = str(ensemble_method or "mean_logit").lower().replace("-", "_")
    if method not in {"mean_prob", "mean_logit"}:
        raise ValueError("ensemble_method must be mean_prob or mean_logit")
    inference_mode = str(inference_mode or "direct").lower().replace("-", "_")
    if inference_mode not in {"direct", "sliding_window"}:
        raise ValueError("inference_mode must be direct or sliding_window")

    config = configs[0]
    seed_everything(config.seed, deterministic=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not bool(config.trainer.cpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = bool(config.trainer.amp and use_cuda)

    norm_mode = _normalization_mode(config)
    dataset, modalities, use_rgb = prepare_evaluation_dataset(config, split=split)
    _filter_dataset_by_events(dataset, include_events=include_events, exclude_events=exclude_events)
    if max_samples is not None and int(max_samples) > 0:
        indices = list(range(min(int(max_samples), len(dataset))))
        indexed_dataset: Dataset = torch.utils.data.Subset(IndexedDataset(dataset), indices)
    else:
        indexed_dataset = IndexedDataset(dataset)
    loader_batch_size = 1 if inference_mode == "sliding_window" else config.trainer.batch_size
    loader = DataLoader(dataset=indexed_dataset, batch_size=loader_batch_size, shuffle=False,
                        num_workers=config.trainer.num_workers, worker_init_fn=seed_worker,
                        collate_fn=pad_segmentation_batch)

    models = []
    for idx, (cfg, checkpoint_path) in enumerate(zip(configs, checkpoint_paths), start=1):
        model = prepare_model(config=cfg, num_classes=1, stage="eval")
        state = load_checkpoint_state(Path(checkpoint_path))
        model.load_state_dict(state, strict=not cfg.model.multibranch)
        model = model.to(device)
        model.eval()
        models.append(model)
        LOG.info("Loaded ensemble member %d/%d: decoder=%s encoder=%s checkpoint=%s", idx, len(checkpoint_paths), cfg.model.decoder, cfg.model.encoder, checkpoint_path)

    thresholds = list(thresholds or DEFAULT_THRESHOLDS)
    threshold = float(threshold)
    min_component_area = int(min_component_area or 0)
    sweep_component_areas = [int(v) for v in (sweep_component_areas or [0, 8, 16, 32, 64])]
    if min_component_area not in sweep_component_areas:
        sweep_component_areas = sorted(set(sweep_component_areas + [min_component_area]))

    sweep_accumulator = BinaryThresholdSweep(thresholds=thresholds, device=device)
    rows = []
    overlay_payloads = {}
    sweep_parts = defaultdict(lambda: {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0, "empty": 0.0, "empty_fp": 0.0, "nonempty": 0.0, "nonempty_detected": 0.0})

    LOG.info("Auditing %d-model ensemble using %s", len(models), method)
    LOG.info("Dataset: %s split, %d samples", split, len(indexed_dataset))
    with torch.no_grad():
        for x, y, index in progress_iter(loader, desc=f"Ensemble error audit {split}", unit="batch", colour="green"):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0).to(device)
            y_device = y.to(device)
            if inference_mode == "sliding_window":
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    ensemble_logits = ensemble_sliding_window_logits(models, x, method=method, window_size=window_size, overlap=window_overlap, window_batch_size=window_batch_size)
            else:
                member_logits = []
                member_probs = []
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    for model in models:
                        logits = _main_logits(model(x)).float()
                        if method == "mean_logit":
                            member_logits.append(logits)
                        else:
                            member_probs.append(torch.sigmoid(logits))
                if method == "mean_logit":
                    ensemble_logits = torch.stack(member_logits, dim=0).mean(dim=0)
                else:
                    ensemble_logits = _prob_to_logits(torch.stack(member_probs, dim=0).mean(dim=0))
            sweep_accumulator.update(y_device, ensemble_logits)
            prob_batch = torch.sigmoid(BinaryThresholdSweep._squeeze_logits(ensemble_logits)).detach().cpu().numpy()
            y_batch = y.detach().cpu().numpy()
            if y_batch.ndim == 4 and y_batch.shape[1] == 1:
                y_batch = y_batch[:, 0]
            idx_batch = index.detach().cpu().numpy().tolist() if isinstance(index, torch.Tensor) else list(index)
            for b, dataset_index in enumerate(idx_batch):
                dataset_index = int(dataset_index)
                image_path = Path(dataset.image_files[dataset_index])
                mask_path = Path(dataset.label_files[dataset_index])
                dem_path = Path(dataset.dem_files[dataset_index]) if getattr(dataset, "_include_dem", False) else None
                target = y_batch[b].astype(np.uint8)
                valid = target != 255
                target_fg = (target > 0) & valid
                prob = prob_batch[b]
                pred_base = (prob >= threshold) & valid
                pred = _remove_small_components(pred_base, min_component_area) & valid
                tp = int(np.count_nonzero(pred & target_fg)); fp = int(np.count_nonzero(pred & (~target_fg) & valid))
                tn = int(np.count_nonzero((~pred) & (~target_fg) & valid)); fn = int(np.count_nonzero((~pred) & target_fg))
                valid_pixels = int(np.count_nonzero(valid)); fg_pixels = int(np.count_nonzero(target_fg)); pred_pixels = int(np.count_nonzero(pred))
                fg_ratio = _safe_div(float(fg_pixels), float(valid_pixels)); pred_ratio = _safe_div(float(pred_pixels), float(valid_pixels))
                metrics = _metrics_from_counts(tp, tn, fp, fn)
                is_empty = fg_pixels == 0; pred_any = pred_pixels > 0
                if is_empty and pred_any: category = "false_positive_empty"
                elif is_empty and not pred_any: category = "true_negative_empty"
                elif (not is_empty) and not pred_any: category = "false_negative_missed"
                elif metrics["recall"] < 0.25: category = "false_negative_low_recall"
                elif metrics["iou"] < 0.20: category = "poor_overlap"
                elif metrics["iou"] >= 0.50: category = "true_positive_good"
                else: category = "partial_overlap"
                row = {"split": split, "index": dataset_index, "file": image_path.name, "event_id": _event_id_from_name(image_path.name),
                       "tile_row_offset": _tile_offsets_from_name(image_path.name)[0], "tile_col_offset": _tile_offsets_from_name(image_path.name)[1],
                       "image_path": str(image_path), "mask_path": str(mask_path), "dem_path": str(dem_path) if dem_path else "",
                       "threshold": threshold, "min_component_area": min_component_area, "valid_pixels": valid_pixels,
                       "fg_pixels": fg_pixels, "pred_pixels": pred_pixels, "fg_ratio": fg_ratio, "pred_ratio": pred_ratio,
                       "foreground_bin": _foreground_bin(fg_ratio), "is_empty": bool(is_empty), "pred_any": bool(pred_any),
                       "tp_pixels": tp, "tn_pixels": tn, "fp_pixels": fp, "fn_pixels": fn, "error_category": category, **metrics}
                rows.append(row)
                key = str(dataset_index)
                if category != "partial_overlap" or metrics["iou"] < 0.35:
                    overlay_payloads[key] = (pred.copy(), prob.copy())
                for item in _sweep_rows_for_sample(prob, target, thresholds, sweep_component_areas):
                    acc = sweep_parts[(float(item["threshold"]), int(item["min_component_area"]))]
                    for name in ["tp", "tn", "fp", "fn", "empty", "empty_fp", "nonempty", "nonempty_detected"]:
                        acc[name] += float(item[name])

    df = pd.DataFrame(rows)
    tile_csv = output_dir / "tile_error_metrics.csv"
    df.to_csv(tile_csv, index=False)
    _aggregate_metrics(df, ["event_id"]).sort_values(["f1", "iou"], ascending=True).to_csv(output_dir / "event_metrics.csv", index=False)
    _aggregate_metrics(df, ["foreground_bin"]).sort_values("foreground_bin").to_csv(output_dir / "foreground_bin_metrics.csv", index=False)
    df.groupby("error_category", dropna=False).agg(tiles=("file", "count"), mean_f1=("f1", "mean"), mean_iou=("iou", "mean"), fg_pixels=("fg_pixels", "sum"), pred_pixels=("pred_pixels", "sum"), fp_pixels=("fp_pixels", "sum"), fn_pixels=("fn_pixels", "sum")).reset_index().sort_values("tiles", ascending=False).to_csv(output_dir / "error_category_summary.csv", index=False)
    sweep_rows = []
    for (thr, area), acc in sorted(sweep_parts.items()):
        metrics = _metrics_from_counts(acc["tp"], acc["tn"], acc["fp"], acc["fn"])
        sweep_rows.append({"threshold": thr, "min_component_area": area, **metrics, "empty_fp_rate": _safe_div(acc["empty_fp"], acc["empty"]), "nonempty_tile_recall": _safe_div(acc["nonempty_detected"], acc["nonempty"]), "tp_pixels": acc["tp"], "tn_pixels": acc["tn"], "fp_pixels": acc["fp"], "fn_pixels": acc["fn"]})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(output_dir / "threshold_component_sweep.csv", index=False)
    selected = _select_overlay_rows(df, max_per_category=max_overlays_per_category)
    selected.to_csv(output_dir / "selected_overlay_tiles.csv", index=False)
    written = 0
    for _, row in selected.iterrows():
        key = str(int(row["index"]))
        if key not in overlay_payloads:
            continue
        pred_mask, prob = overlay_payloads[key]
        safe_file = Path(row["file"]).stem.replace("/", "_")
        out_path = output_dir / "overlays" / str(row["error_category"]) / f"{safe_file}.png"
        dem_path = Path(row["dem_path"]) if str(row.get("dem_path", "")) else None
        _write_overlay(row.to_dict(), pred_mask, prob, out_path, dem_path=dem_path)
        written += 1
    best_sweep = sweep_accumulator.best(threshold_metric)
    best_component = None
    if not sweep_df.empty:
        best_idx = sweep_df[threshold_metric].astype(float).idxmax(); best_component = sweep_df.loc[best_idx].to_dict()
    counts = {"tp": float(df["tp_pixels"].sum()), "tn": float(df["tn_pixels"].sum()), "fp": float(df["fp_pixels"].sum()), "fn": float(df["fn_pixels"].sum())}
    overall = _metrics_from_counts(counts["tp"], counts["tn"], counts["fp"], counts["fn"])
    summary = {"checkpoints": [str(p) for p in checkpoint_paths], "ensemble_method": method, "inference_mode": inference_mode, "split": split,
               "include_events": list(include_events or []), "exclude_events": list(exclude_events or []),
               "samples": int(len(df)), "audit_threshold": threshold, "audit_min_component_area": min_component_area,
               "modalities": modalities, "normalization_mode": norm_mode, "global_threshold_sweep_best": asdict(best_sweep),
               "operating_point_metrics": {**overall, **counts}, "empty_tiles": int(df["is_empty"].sum()),
               "empty_fp_tiles": int(((df["is_empty"]) & (df["pred_any"])).sum()),
               "empty_fp_rate": _safe_div(float(((df["is_empty"]) & (df["pred_any"])).sum()), float(df["is_empty"].sum())),
               "nonempty_tiles": int((~df["is_empty"]).sum()), "nonempty_detected_tiles": int(((~df["is_empty"]) & (df["pred_any"])).sum()),
               "nonempty_tile_recall": _safe_div(float(((~df["is_empty"]) & (df["pred_any"])).sum()), float((~df["is_empty"]).sum())),
               "category_counts": Counter(df["error_category"]).most_common(), "best_threshold_component_setting": best_component,
               "overlays_written": written}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=_json_default)
    LOG.info("Ensemble error audit written to: %s", output_dir)
    LOG.info("Operating point @ threshold %.2f: f1=%.4f | iou=%.4f | precision=%.4f | recall=%.4f | mcc=%.4f", threshold, overall["f1"], overall["iou"], overall["precision"], overall["recall"], overall["mcc"])
    return summary
