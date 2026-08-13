from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from floods.utils.console import progress_iter

from floods.evaluation import BinaryThresholdSweep, DEFAULT_THRESHOLDS, load_checkpoint_state
from floods.utils.common import get_logger
from floods.utils.gis import imread

LOG = get_logger(__name__)


def _normalization_mode(config: Any) -> str:
    """Return the configured input-normalization mode for audit metadata."""
    data_config = getattr(config, "data", None)
    return str(getattr(data_config, "normalization_mode", "fixed") or "fixed").lower()


class IndexedDataset(Dataset):
    """Wrap a dataset so audit rows can be joined back to tile file names."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, y, index


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _event_id_from_name(name: str) -> str:
    match = re.search(r"(EMSR\d+)", name)
    return match.group(1) if match else "unknown"




def _filter_dataset_by_events(dataset: Any, include_events: Optional[Iterable[str]] = None, exclude_events: Optional[Iterable[str]] = None) -> None:
    include = {str(v).upper() for v in (include_events or [])}
    exclude = {str(v).upper() for v in (exclude_events or [])}
    if not include and not exclude:
        return
    mask = []
    for path in dataset.label_files:
        event = _event_id_from_name(Path(path).name).upper()
        keep = True
        if include:
            keep = event in include
        if exclude and event in exclude:
            keep = False
        mask.append(keep)
    before = len(dataset)
    dataset.add_mask(mask)
    LOG.info("Event filter: kept %d/%d samples (include=%s exclude=%s)", len(dataset), before, sorted(include), sorted(exclude))
    if len(dataset) == 0:
        raise ValueError("Event filter removed every sample")


def _tile_offsets_from_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[-2]), int(parts[-1])
        except ValueError:
            pass
    return None, None


def _foreground_bin(ratio: float) -> str:
    if ratio <= 0.0:
        return "empty"
    if ratio < 0.005:
        return "tiny"
    if ratio < 0.02:
        return "small"
    if ratio < 0.10:
        return "medium"
    return "large"


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metrics_from_counts(tp: float, tn: float, fp: float, fn: float) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denom) if denom > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "mcc": mcc,
    }


def _import_pyplot():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area from a binary mask."""
    min_area = int(min_area or 0)
    if min_area <= 1 or not mask.any():
        return mask.astype(bool, copy=False)
    mask_u8 = mask.astype(np.uint8, copy=False)
    try:
        import cv2  # type: ignore
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        keep = np.zeros(n_labels, dtype=bool)
        if n_labels > 1:
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        return keep[labels]
    except Exception:
        try:
            from scipy import ndimage  # type: ignore
            labels, n_labels = ndimage.label(mask_u8)
            if n_labels == 0:
                return mask.astype(bool, copy=False)
            counts = np.bincount(labels.ravel())
            keep = counts >= min_area
            keep[0] = False
            return keep[labels]
        except Exception:
            # Last-resort fallback: leave the mask unchanged rather than failing the audit.
            return mask.astype(bool, copy=False)


def _read_display_arrays(image_path: Path, mask_path: Path, dem_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    sar = imread(image_path, channels_first=True).astype(np.float32)
    mask = imread(mask_path).squeeze().astype(np.uint8)
    dem = imread(dem_path).squeeze().astype(np.float32) if dem_path and dem_path.exists() else None
    vv = sar[0] if sar.ndim == 3 else sar.squeeze()
    vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.nanpercentile(vv, [1, 99]) if vv.size else (0.0, 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vv)), float(np.nanmax(vv) + 1e-6)
    vv_display = np.clip((vv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return vv_display, mask, dem




def _crop_to_common_shape(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Crop 2D arrays to a shared height and width for diagnostic plotting.

    Test rasters can be padded during evaluation so that model outputs align with
    the network stride. Raw GeoTIFFs remain at their original size. Overlay PNGs
    only need the common valid extent, so this function crops all arrays to the
    smallest shared spatial shape. Metrics are computed before this step and are
    not affected by overlay cropping.
    """
    if not arrays:
        return tuple()
    heights = [int(array.shape[0]) for array in arrays]
    widths = [int(array.shape[1]) for array in arrays]
    height = min(heights)
    width = min(widths)
    return tuple(array[:height, :width] for array in arrays)

def _write_overlay(row: Dict[str, Any], pred_mask: np.ndarray, prob: np.ndarray, output_path: Path, dem_path: Optional[Path] = None) -> None:
    try:
        plt = _import_pyplot()
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        vv_display, target, _ = _read_display_arrays(image_path, mask_path, dem_path=dem_path)
        pred_mask, prob, target, vv_display = _crop_to_common_shape(
            pred_mask.astype(bool), prob.astype(np.float32), target, vv_display
        )
        valid = target != 255
        target_fg = (target > 0) & valid
        pred_fg = pred_mask & valid
        fp = pred_fg & ~target_fg
        fn = target_fg & ~pred_fg
        tp = target_fg & pred_fg

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(2, 3, 1)
        ax.imshow(vv_display, cmap="gray")
        ax.set_title("Processed VV display")
        ax.axis("off")
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(target_fg, cmap="gray")
        ax.set_title("Ground truth flood")
        ax.axis("off")
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(pred_fg, cmap="gray")
        ax.set_title("Prediction")
        ax.axis("off")
        ax = fig.add_subplot(2, 3, 4)
        ax.imshow(vv_display, cmap="gray")
        ax.imshow(target_fg, alpha=0.35)
        ax.set_title("Truth overlay")
        ax.axis("off")
        ax = fig.add_subplot(2, 3, 5)
        ax.imshow(vv_display, cmap="gray")
        # RGB overlay: TP green, FP red, FN blue.
        overlay = np.zeros((*target.shape, 4), dtype=np.float32)
        overlay[tp] = (0.0, 1.0, 0.0, 0.45)
        overlay[fp] = (1.0, 0.0, 0.0, 0.45)
        overlay[fn] = (0.0, 0.25, 1.0, 0.55)
        ax.imshow(overlay)
        ax.set_title("TP green | FP red | FN blue")
        ax.axis("off")
        ax = fig.add_subplot(2, 3, 6)
        ax.imshow(prob, vmin=0.0, vmax=1.0)
        ax.set_title("Predicted probability")
        ax.axis("off")
        title = (f"{row['file']} | {row['error_category']} | F1={row['f1']:.3f} IoU={row['iou']:.3f} "
                 f"fg={row['fg_ratio']:.4f} pred={row['pred_ratio']:.4f}")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_path, dpi=135)
        plt.close(fig)
    except Exception as exc:
        LOG.warning("Could not write overlay for %s: %s", row.get("file"), exc)


def _select_overlay_rows(df: pd.DataFrame, max_per_category: int | None = None, **kwargs: Any) -> pd.DataFrame:
    """Select representative rows for diagnostic overlays.

    The primary argument is ``max_per_category``.  ``max_overlays_per_category``
    is accepted as a compatibility alias because ensemble and single-model audit
    commands expose that user-facing option name.
    """
    if max_per_category is None:
        max_per_category = kwargs.pop("max_overlays_per_category", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected overlay selection argument(s): {unknown}")
    max_per_category = int(max_per_category or 0)
    if max_per_category <= 0 or df.empty:
        return df.iloc[0:0]
    selected = []
    sort_specs = {
        "false_positive_empty": ["fp_pixels", "pred_ratio"],
        "false_negative_missed": ["fn_pixels", "fg_ratio"],
        "false_negative_low_recall": ["fn_pixels", "fg_ratio"],
        "poor_overlap": ["iou", "fg_ratio"],
        "true_positive_good": ["iou", "fg_ratio"],
        "true_negative_empty": ["valid_pixels"],
    }
    for category, group in df.groupby("error_category"):
        if category in {"true_positive_good", "true_negative_empty"}:
            group = group.sort_values(sort_specs.get(category, ["iou"]), ascending=False)
        elif category == "poor_overlap":
            group = group.sort_values(["iou", "fg_ratio"], ascending=[True, False])
        else:
            group = group.sort_values(sort_specs.get(category, ["fp_pixels"]), ascending=False)
        selected.append(group.head(max_per_category))
    return pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0]


def _aggregate_metrics(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, group in df.groupby(list(by), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tp = float(group["tp_pixels"].sum())
        tn = float(group["tn_pixels"].sum())
        fp = float(group["fp_pixels"].sum())
        fn = float(group["fn_pixels"].sum())
        metrics = _metrics_from_counts(tp, tn, fp, fn)
        row = {name: value for name, value in zip(by, keys)}
        row.update(metrics)
        row.update({
            "tiles": int(len(group)),
            "valid_pixels": int(group["valid_pixels"].sum()),
            "fg_pixels": int(group["fg_pixels"].sum()),
            "pred_pixels": int(group["pred_pixels"].sum()),
            "empty_tiles": int(group["is_empty"].sum()),
            "empty_fp_tiles": int(((group["is_empty"]) & (group["pred_any"])).sum()),
            "empty_fp_rate": _safe_div(float(((group["is_empty"]) & (group["pred_any"])).sum()), float(group["is_empty"].sum())),
            "nonempty_tiles": int((~group["is_empty"]).sum()),
            "nonempty_detected_tiles": int(((~group["is_empty"]) & (group["pred_any"])).sum()),
            "nonempty_tile_recall": _safe_div(float(((~group["is_empty"]) & (group["pred_any"])).sum()), float((~group["is_empty"]).sum())),
            "mean_fg_ratio": float(group["fg_ratio"].mean()),
            "mean_pred_ratio": float(group["pred_ratio"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _sweep_rows_for_sample(prob: np.ndarray,
                           target: np.ndarray,
                           thresholds: Sequence[float],
                           min_component_areas: Sequence[int]) -> List[Dict[str, float]]:
    valid = target != 255
    target_fg = (target > 0) & valid
    rows = []
    for threshold in thresholds:
        pred_base = (prob >= float(threshold)) & valid
        for area in min_component_areas:
            pred = _remove_small_components(pred_base, int(area)) & valid
            tp = float(np.count_nonzero(pred & target_fg))
            fp = float(np.count_nonzero(pred & (~target_fg) & valid))
            tn = float(np.count_nonzero((~pred) & (~target_fg) & valid))
            fn = float(np.count_nonzero((~pred) & target_fg))
            true_any = bool(np.any(target_fg))
            pred_any = bool(np.any(pred))
            row = {"threshold": float(threshold), "min_component_area": int(area), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                   "empty": not true_any, "empty_fp": (not true_any) and pred_any,
                   "nonempty": true_any, "nonempty_detected": true_any and pred_any}
            rows.append(row)
    return rows


def error_audit_checkpoint(config: "Any",
                           checkpoint_path: Path,
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
                           exclude_events: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Audit model errors on a processed split and write CSV/PNG artifacts.

    The audit is intentionally separate from training. It does not change model
    weights. It identifies whether poor performance is dominated by empty-tile
    false positives, missed foreground, poor overlap/boundaries, or specific
    events/foreground-ratio bins.
    """
    from floods.datasets.flood import FloodDataset, RGBFloodDataset
    from floods.normalization import describe_stats, load_normalization_stats
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.utils.ml import seed_everything, seed_worker
    from floods.eval_collate import pad_segmentation_batch

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
        # Keep the dataset object intact but only iterate the first N indices. This is useful for smoke tests.
        indices = list(range(min(int(max_samples), len(dataset))))
        indexed_dataset: Dataset = torch.utils.data.Subset(IndexedDataset(dataset), indices)
    else:
        indexed_dataset = IndexedDataset(dataset)
    loader = DataLoader(dataset=indexed_dataset,
                        batch_size=config.trainer.batch_size,
                        shuffle=False,
                        num_workers=config.trainer.num_workers,
                        worker_init_fn=seed_worker,
                        collate_fn=pad_segmentation_batch)

    model = prepare_model(config=config, num_classes=1, stage="eval")
    state = load_checkpoint_state(Path(checkpoint_path))
    model.load_state_dict(state, strict=not config.model.multibranch)
    model = model.to(device)
    model.eval()

    thresholds = list(thresholds or DEFAULT_THRESHOLDS)
    threshold = float(threshold)
    min_component_area = int(min_component_area or 0)
    sweep_component_areas = [int(v) for v in (sweep_component_areas or [0, 8, 16, 32, 64])]
    if min_component_area not in sweep_component_areas:
        sweep_component_areas = sorted(set(sweep_component_areas + [min_component_area]))

    sweep_accumulator = BinaryThresholdSweep(thresholds=thresholds, device=device)
    rows: List[Dict[str, Any]] = []
    overlay_payloads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    sweep_parts: Dict[Tuple[float, int], Dict[str, float]] = defaultdict(lambda: {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0,
                                                                                  "empty": 0.0, "empty_fp": 0.0,
                                                                                  "nonempty": 0.0, "nonempty_detected": 0.0})

    LOG.info("Auditing checkpoint: %s", checkpoint_path)
    LOG.info("Dataset: %s split, %d samples", split, len(indexed_dataset))
    with torch.no_grad():
        for x, y, index in progress_iter(loader, desc=f"Error audit {split}", unit="batch", colour="green"):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0).to(device)
            y_device = y.to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x)
            logits = BinaryThresholdSweep._main_prediction(out).detach().float()
            sweep_accumulator.update(y_device, logits)
            prob_batch = torch.sigmoid(BinaryThresholdSweep._squeeze_logits(logits)).detach().cpu().numpy()
            y_batch = y.detach().cpu().numpy()
            if y_batch.ndim == 4 and y_batch.shape[1] == 1:
                y_batch = y_batch[:, 0]
            idx_batch = index.detach().cpu().numpy().tolist() if isinstance(index, torch.Tensor) else list(index)

            for b, dataset_index in enumerate(idx_batch):
                # Subset returns original wrapper index as a scalar tensor, not Subset-relative index.
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
                tp = int(np.count_nonzero(pred & target_fg))
                fp = int(np.count_nonzero(pred & (~target_fg) & valid))
                tn = int(np.count_nonzero((~pred) & (~target_fg) & valid))
                fn = int(np.count_nonzero((~pred) & target_fg))
                valid_pixels = int(np.count_nonzero(valid))
                fg_pixels = int(np.count_nonzero(target_fg))
                pred_pixels = int(np.count_nonzero(pred))
                fg_ratio = _safe_div(float(fg_pixels), float(valid_pixels))
                pred_ratio = _safe_div(float(pred_pixels), float(valid_pixels))
                metrics = _metrics_from_counts(tp, tn, fp, fn)
                is_empty = fg_pixels == 0
                pred_any = pred_pixels > 0
                if is_empty and pred_any:
                    category = "false_positive_empty"
                elif is_empty and not pred_any:
                    category = "true_negative_empty"
                elif (not is_empty) and not pred_any:
                    category = "false_negative_missed"
                elif metrics["recall"] < 0.25:
                    category = "false_negative_low_recall"
                elif metrics["iou"] < 0.20:
                    category = "poor_overlap"
                elif metrics["iou"] >= 0.50:
                    category = "true_positive_good"
                else:
                    category = "partial_overlap"
                row = {
                    "split": split,
                    "index": dataset_index,
                    "file": image_path.name,
                    "event_id": _event_id_from_name(image_path.name),
                    "tile_row_offset": _tile_offsets_from_name(image_path.name)[0],
                    "tile_col_offset": _tile_offsets_from_name(image_path.name)[1],
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "dem_path": str(dem_path) if dem_path else "",
                    "threshold": threshold,
                    "min_component_area": min_component_area,
                    "valid_pixels": valid_pixels,
                    "fg_pixels": fg_pixels,
                    "pred_pixels": pred_pixels,
                    "fg_ratio": fg_ratio,
                    "pred_ratio": pred_ratio,
                    "foreground_bin": _foreground_bin(fg_ratio),
                    "is_empty": bool(is_empty),
                    "pred_any": bool(pred_any),
                    "tp_pixels": tp,
                    "tn_pixels": tn,
                    "fp_pixels": fp,
                    "fn_pixels": fn,
                    "error_category": category,
                    **metrics,
                }
                rows.append(row)
                key = f"{dataset_index}"
                if category != "partial_overlap" or metrics["iou"] < 0.35:
                    overlay_payloads[key] = (pred.copy(), prob.copy())

                for item in _sweep_rows_for_sample(prob, target, thresholds, sweep_component_areas):
                    k = (float(item["threshold"]), int(item["min_component_area"]))
                    acc = sweep_parts[k]
                    acc["tp"] += item["tp"]
                    acc["tn"] += item["tn"]
                    acc["fp"] += item["fp"]
                    acc["fn"] += item["fn"]
                    acc["empty"] += float(item["empty"])
                    acc["empty_fp"] += float(item["empty_fp"])
                    acc["nonempty"] += float(item["nonempty"])
                    acc["nonempty_detected"] += float(item["nonempty_detected"])

    df = pd.DataFrame(rows)
    tile_csv = output_dir / "tile_error_metrics.csv"
    df.to_csv(tile_csv, index=False)

    # Aggregates
    event_metrics = _aggregate_metrics(df, ["event_id"]).sort_values(["f1", "iou"], ascending=True)
    event_metrics.to_csv(output_dir / "event_metrics.csv", index=False)
    bin_metrics = _aggregate_metrics(df, ["foreground_bin"]).sort_values("foreground_bin")
    bin_metrics.to_csv(output_dir / "foreground_bin_metrics.csv", index=False)
    category_counts = df.groupby("error_category", dropna=False).agg(
        tiles=("file", "count"),
        mean_f1=("f1", "mean"),
        mean_iou=("iou", "mean"),
        fg_pixels=("fg_pixels", "sum"),
        pred_pixels=("pred_pixels", "sum"),
        fp_pixels=("fp_pixels", "sum"),
        fn_pixels=("fn_pixels", "sum"),
    ).reset_index().sort_values("tiles", ascending=False)
    category_counts.to_csv(output_dir / "error_category_summary.csv", index=False)

    sweep_rows = []
    for (thr, area), acc in sorted(sweep_parts.items()):
        metrics = _metrics_from_counts(acc["tp"], acc["tn"], acc["fp"], acc["fn"])
        sweep_rows.append({
            "threshold": thr,
            "min_component_area": area,
            **metrics,
            "empty_fp_rate": _safe_div(acc["empty_fp"], acc["empty"]),
            "nonempty_tile_recall": _safe_div(acc["nonempty_detected"], acc["nonempty"]),
            "tp_pixels": acc["tp"],
            "tn_pixels": acc["tn"],
            "fp_pixels": acc["fp"],
            "fn_pixels": acc["fn"],
        })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(output_dir / "threshold_component_sweep.csv", index=False)

    # Overlay exports
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
        best_idx = sweep_df[threshold_metric].astype(float).idxmax()
        best_component = sweep_df.loc[best_idx].to_dict()
    overall_counts = {
        "tp": float(df["tp_pixels"].sum()),
        "tn": float(df["tn_pixels"].sum()),
        "fp": float(df["fp_pixels"].sum()),
        "fn": float(df["fn_pixels"].sum()),
    }
    overall = _metrics_from_counts(overall_counts["tp"], overall_counts["tn"], overall_counts["fp"], overall_counts["fn"])
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "include_events": list(include_events or []),
        "exclude_events": list(exclude_events or []),
        "samples": int(len(df)),
        "audit_threshold": threshold,
        "audit_min_component_area": min_component_area,
        "modalities": modalities,
        "normalization_mode": norm_mode,
        "global_threshold_sweep_best": asdict(best_sweep),
        "operating_point_metrics": {**overall, **overall_counts},
        "empty_tiles": int(df["is_empty"].sum()),
        "empty_fp_tiles": int(((df["is_empty"]) & (df["pred_any"])).sum()),
        "empty_fp_rate": _safe_div(float(((df["is_empty"]) & (df["pred_any"])).sum()), float(df["is_empty"].sum())),
        "nonempty_tiles": int((~df["is_empty"]).sum()),
        "nonempty_detected_tiles": int(((~df["is_empty"]) & (df["pred_any"])).sum()),
        "nonempty_tile_recall": _safe_div(float(((~df["is_empty"]) & (df["pred_any"])).sum()), float((~df["is_empty"]).sum())),
        "category_counts": Counter(df["error_category"]).most_common(),
        "best_threshold_component_setting": best_component,
        "files": {
            "tile_error_metrics": str(tile_csv),
            "event_metrics": str(output_dir / "event_metrics.csv"),
            "foreground_bin_metrics": str(output_dir / "foreground_bin_metrics.csv"),
            "error_category_summary": str(output_dir / "error_category_summary.csv"),
            "threshold_component_sweep": str(output_dir / "threshold_component_sweep.csv"),
            "selected_overlay_tiles": str(output_dir / "selected_overlay_tiles.csv"),
            "overlays_dir": str(output_dir / "overlays"),
        },
        "overlays_written": written,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    LOG.info("Error audit written to: %s", output_dir)
    LOG.info("Operating point @ threshold %.2f, min_component_area=%d: f1=%.4f | iou=%.4f | precision=%.4f | recall=%.4f | mcc=%.4f",
             threshold, min_component_area, overall["f1"], overall["iou"], overall["precision"], overall["recall"], overall["mcc"])
    LOG.info("Empty-tile false positives: %d/%d (%.4f)", summary["empty_fp_tiles"], summary["empty_tiles"], summary["empty_fp_rate"])
    if best_component:
        LOG.info("Best threshold/component setting by %s: threshold=%.2f | min_area=%d | f1=%.4f | iou=%.4f | empty_fp=%.4f",
                 threshold_metric, float(best_component["threshold"]), int(best_component["min_component_area"]),
                 float(best_component["f1"]), float(best_component["iou"]), float(best_component["empty_fp_rate"]))
    return summary
