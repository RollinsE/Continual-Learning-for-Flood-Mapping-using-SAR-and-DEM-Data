from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from floods.error_audit import (
    IndexedDataset,
    _filter_dataset_by_events,
    _foreground_bin,
    _remove_small_components,
    _safe_div,
)
from floods.evaluation import BinaryThresholdSweep, load_checkpoint_state
from floods.utils.common import get_logger
from floods.utils.console import progress_iter
from floods.water_prior_audit import (
    _finalize_accumulator,
    _json_default,
    _setting_accumulator,
    _update_accumulator,
)

LOG = get_logger(__name__)

DEFAULT_FIXED_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
DEFAULT_LOW_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
DEFAULT_HIGH_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
DEFAULT_MIN_SEED_PIXELS = [1, 16, 64]
DEFAULT_COMPONENT_AREAS = [96]


@dataclass(frozen=True)
class HysteresisSetting:
    strategy: str
    low_threshold: float
    high_threshold: float
    min_seed_pixels: int = 0

    @property
    def key(self) -> str:
        if self.strategy == "fixed":
            return f"fixed_thr{self.low_threshold:.3f}"
        return (
            f"hyst_low{self.low_threshold:.3f}_high{self.high_threshold:.3f}_"
            f"seed{self.min_seed_pixels}"
        )


def _event_id_from_name(name: str) -> str:
    match = re.search(r"(EMSR\d+)", str(name), flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def build_hysteresis_settings(
    fixed_thresholds: Sequence[float],
    low_thresholds: Sequence[float],
    high_thresholds: Sequence[float],
    min_seed_pixels: Sequence[int],
) -> List[HysteresisSetting]:
    fixed = sorted({float(value) for value in fixed_thresholds})
    lows = sorted({float(value) for value in low_thresholds})
    highs = sorted({float(value) for value in high_thresholds})
    seeds = sorted({int(value) for value in min_seed_pixels})

    for threshold in [*fixed, *lows, *highs]:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Thresholds must lie in [0, 1]")
    if any(value < 1 for value in seeds):
        raise ValueError("Minimum seed pixels must be at least 1")

    settings: List[HysteresisSetting] = [
        HysteresisSetting("fixed", threshold, threshold, 0)
        for threshold in fixed
    ]
    for low in lows:
        for high in highs:
            if low >= high:
                continue
            for seed_pixels in seeds:
                settings.append(
                    HysteresisSetting(
                        strategy="hysteresis",
                        low_threshold=low,
                        high_threshold=high,
                        min_seed_pixels=seed_pixels,
                    )
                )
    return settings


def apply_hysteresis_threshold(
    probability: np.ndarray,
    valid: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    min_seed_pixels: int = 1,
    connectivity: int = 8,
) -> np.ndarray:
    """Grow low-threshold components only when they contain strong seed pixels.

    This retains the spatial extent available at ``low_threshold`` while requiring
    each retained component to contain at least ``min_seed_pixels`` pixels at or
    above ``high_threshold``. It is therefore less permissive than lowering one
    global threshold across the whole tile.
    """
    probability = np.asarray(probability, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if probability.shape != valid.shape:
        raise ValueError("Probability and valid-mask shapes must match")
    low = float(low_threshold)
    high = float(high_threshold)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Hysteresis requires 0 <= low_threshold < high_threshold <= 1")
    min_seed_pixels = int(min_seed_pixels)
    if min_seed_pixels < 1:
        raise ValueError("min_seed_pixels must be at least 1")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")

    low_mask = valid & np.isfinite(probability) & (probability >= low)
    if not np.any(low_mask):
        return np.zeros_like(valid, dtype=bool)
    high_mask = low_mask & (probability >= high)
    if not np.any(high_mask):
        return np.zeros_like(valid, dtype=bool)

    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, component_count = ndimage.label(low_mask, structure=structure)
    if component_count == 0:
        return np.zeros_like(valid, dtype=bool)

    seed_counts = np.bincount(
        labels[high_mask].reshape(-1),
        minlength=component_count + 1,
    )
    keep_labels = np.flatnonzero(seed_counts >= min_seed_pixels)
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(valid, dtype=bool)
    return np.isin(labels, keep_labels) & valid


def apply_setting(
    probability: np.ndarray,
    valid: np.ndarray,
    setting: HysteresisSetting,
    connectivity: int = 8,
) -> np.ndarray:
    if setting.strategy == "fixed":
        return valid & np.isfinite(probability) & (
            probability >= float(setting.low_threshold)
        )
    if setting.strategy == "hysteresis":
        return apply_hysteresis_threshold(
            probability,
            valid,
            low_threshold=setting.low_threshold,
            high_threshold=setting.high_threshold,
            min_seed_pixels=setting.min_seed_pixels,
            connectivity=connectivity,
        )
    raise ValueError(f"Unsupported strategy: {setting.strategy}")


def _sort_best(rows: pd.DataFrame) -> Mapping[str, Any]:
    if rows.empty:
        raise ValueError("Cannot select from an empty hysteresis sweep")
    sort_columns = ["f1", "iou", "mcc", "precision"]
    if "incremental_f1_gain_vs_best_endpoint" in rows.columns:
        sort_columns.insert(1, "incremental_f1_gain_vs_best_endpoint")
    return rows.sort_values(sort_columns, ascending=[False] * len(sort_columns)).iloc[0].to_dict()


def annotate_hysteresis_sweep(
    sweep: pd.DataFrame,
    reference: Mapping[str, Any],
    max_recall_drop: float,
    max_empty_fp_rate_increase: float,
) -> pd.DataFrame:
    if sweep.empty:
        raise ValueError("Hysteresis sweep is empty")
    annotated = sweep.copy()
    annotated["is_hysteresis_setting"] = annotated["strategy"].eq("hysteresis")
    minimum_recall = float(reference["recall"]) - float(max_recall_drop)
    maximum_empty_fp_rate = float(reference["empty_fp_rate"]) + float(
        max_empty_fp_rate_increase
    )
    annotated["recall_guard_eligible"] = annotated["recall"] >= minimum_recall
    annotated["empty_fp_guard_eligible"] = (
        annotated["empty_fp_rate"] <= maximum_empty_fp_rate
    )
    annotated["decision_guard_eligible"] = (
        annotated["recall_guard_eligible"] & annotated["empty_fp_guard_eligible"]
    )

    metric_columns = ["precision", "recall", "f1", "iou", "mcc"]
    for metric in metric_columns:
        annotated[f"{metric}_change_vs_reference"] = (
            annotated[metric].astype(float) - float(reference[metric])
        )

    fixed = annotated[annotated["strategy"].eq("fixed")].copy()
    if fixed.duplicated(["low_threshold", "min_component_area"]).any():
        raise RuntimeError("Duplicate fixed-threshold baseline rows found")
    fixed_lookup = fixed.set_index(["low_threshold", "min_component_area"])

    matched_low: List[float] = []
    matched_high: List[float] = []
    matched_low_recall: List[float] = []
    matched_high_recall: List[float] = []
    for row in annotated.itertuples(index=False):
        low_key = (float(row.low_threshold), int(row.min_component_area))
        high_key = (float(row.high_threshold), int(row.min_component_area))
        low_row = fixed_lookup.loc[low_key] if low_key in fixed_lookup.index else None
        high_row = fixed_lookup.loc[high_key] if high_key in fixed_lookup.index else None
        matched_low.append(float(low_row["f1"]) if low_row is not None else np.nan)
        matched_high.append(float(high_row["f1"]) if high_row is not None else np.nan)
        matched_low_recall.append(float(low_row["recall"]) if low_row is not None else np.nan)
        matched_high_recall.append(float(high_row["recall"]) if high_row is not None else np.nan)

    annotated["matched_fixed_low_f1"] = matched_low
    annotated["matched_fixed_high_f1"] = matched_high
    annotated["matched_fixed_low_recall"] = matched_low_recall
    annotated["matched_fixed_high_recall"] = matched_high_recall
    annotated["matched_best_endpoint_f1"] = np.nanmax(
        annotated[["matched_fixed_low_f1", "matched_fixed_high_f1"]].to_numpy(dtype=float),
        axis=1,
    )
    annotated["incremental_f1_gain_vs_best_endpoint"] = (
        annotated["f1"] - annotated["matched_best_endpoint_f1"]
    )
    return annotated


def choose_best_strategy(
    sweep: pd.DataFrame,
    *,
    strategy: str,
    guarded: bool,
) -> Optional[Mapping[str, Any]]:
    rows = sweep[sweep["strategy"].eq(strategy)]
    if guarded:
        rows = rows[rows["decision_guard_eligible"]]
    if rows.empty:
        return None
    return _sort_best(rows)


def hysteresis_recommendation(
    best_hysteresis_guarded: Optional[Mapping[str, Any]],
    best_fixed_guarded: Optional[Mapping[str, Any]],
) -> Tuple[str, Optional[float], Optional[float]]:
    if best_hysteresis_guarded is None:
        return "do_not_use_hysteresis_no_guard_eligible_setting", None, None
    if best_fixed_guarded is None:
        return "do_not_use_hysteresis_missing_fixed_comparator", None, None

    gain_over_best_fixed = float(best_hysteresis_guarded["f1"]) - float(
        best_fixed_guarded["f1"]
    )
    endpoint_gain = float(
        best_hysteresis_guarded["incremental_f1_gain_vs_best_endpoint"]
    )
    if gain_over_best_fixed >= 0.005 and endpoint_gain >= 0.001:
        recommendation = "proceed_to_full_validation_hysteresis_audit"
    elif gain_over_best_fixed >= 0.001 and endpoint_gain > 0.0:
        recommendation = "weak_hysteresis_gain_review_tile_tradeoffs"
    else:
        recommendation = "do_not_use_hysteresis_no_incremental_gain"
    return recommendation, gain_over_best_fixed, endpoint_gain


def audit_hysteresis_postprocess(
    config: Any,
    checkpoint_path: Path,
    processed_data_dir: Path,
    output_dir: Path,
    split: str = "val",
    include_events: Optional[Iterable[str]] = None,
    exclude_events: Optional[Iterable[str]] = None,
    fixed_thresholds: Optional[Sequence[float]] = None,
    low_thresholds: Optional[Sequence[float]] = None,
    high_thresholds: Optional[Sequence[float]] = None,
    min_seed_pixels: Optional[Sequence[int]] = None,
    min_component_areas: Optional[Sequence[int]] = None,
    reference_threshold: float = 0.50,
    reference_min_component_area: int = 96,
    max_recall_drop: float = 0.02,
    max_empty_fp_rate_increase: float = 0.0,
    connectivity: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Audit seeded hysteresis region growing without changing model weights."""
    from floods.eval_collate import pad_segmentation_batch
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.utils.ml import seed_everything, seed_worker

    seed_everything(config.seed, deterministic=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.data.path = str(processed_data_dir)

    dataset, modalities, _ = prepare_evaluation_dataset(config, split=split)
    _filter_dataset_by_events(
        dataset,
        include_events=include_events,
        exclude_events=exclude_events,
    )
    if max_samples is not None and int(max_samples) > 0:
        active_indices = list(range(min(int(max_samples), len(dataset))))
    else:
        active_indices = list(range(len(dataset)))

    indexed_dataset: Dataset = torch.utils.data.Subset(
        IndexedDataset(dataset), active_indices
    )
    loader = DataLoader(
        dataset=indexed_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        worker_init_fn=seed_worker,
        collate_fn=pad_segmentation_batch,
    )

    use_cuda = torch.cuda.is_available() and not bool(config.trainer.cpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = bool(config.trainer.amp and use_cuda)
    model = prepare_model(config=config, num_classes=1, stage="eval")
    state = load_checkpoint_state(Path(checkpoint_path))
    model.load_state_dict(state, strict=not config.model.multibranch)
    model = model.to(device)
    model.eval()

    fixed_values = sorted(
        {float(value) for value in (fixed_thresholds or DEFAULT_FIXED_THRESHOLDS)}
        | {float(value) for value in (low_thresholds or DEFAULT_LOW_THRESHOLDS)}
        | {float(value) for value in (high_thresholds or DEFAULT_HIGH_THRESHOLDS)}
        | {float(reference_threshold)}
    )
    lows = sorted({float(value) for value in (low_thresholds or DEFAULT_LOW_THRESHOLDS)})
    highs = sorted({float(value) for value in (high_thresholds or DEFAULT_HIGH_THRESHOLDS)})
    seeds = sorted({int(value) for value in (min_seed_pixels or DEFAULT_MIN_SEED_PIXELS)})
    component_areas = sorted(
        {int(value) for value in (min_component_areas or DEFAULT_COMPONENT_AREAS)}
        | {int(reference_min_component_area)}
    )
    settings = build_hysteresis_settings(fixed_values, lows, highs, seeds)

    accumulators: Dict[Tuple[str, int], Dict[str, float]] = {}
    event_accumulators: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    tile_rows: List[Dict[str, Any]] = []

    LOG.info("Auditing hysteresis postprocessing with checkpoint: %s", checkpoint_path)
    LOG.info(
        "Dataset: split=%s samples=%d settings=%d component_areas=%s",
        split,
        len(indexed_dataset),
        len(settings),
        component_areas,
    )

    with torch.no_grad():
        for x, y, index in progress_iter(
            loader,
            desc=f"Hysteresis audit {split}",
            unit="batch",
            colour="blue",
        ):
            x = (
                torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0)
                .clamp(-30.0, 30.0)
                .to(device)
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(x)
            logits = BinaryThresholdSweep._main_prediction(output).detach().float()
            probability_batch = (
                torch.sigmoid(BinaryThresholdSweep._squeeze_logits(logits))
                .detach()
                .cpu()
                .numpy()
            )
            target_batch = y.detach().cpu().numpy()
            if target_batch.ndim == 4 and target_batch.shape[1] == 1:
                target_batch = target_batch[:, 0]
            index_batch = (
                index.detach().cpu().numpy().tolist()
                if isinstance(index, torch.Tensor)
                else list(index)
            )

            for batch_index, dataset_index in enumerate(index_batch):
                dataset_index = int(dataset_index)
                image_path = Path(dataset.image_files[dataset_index])
                event_id = _event_id_from_name(image_path.name)
                target = target_batch[batch_index].astype(np.uint8)
                probability = probability_batch[batch_index].astype(np.float32)
                valid = target != 255

                for setting in settings:
                    base_prediction = apply_setting(
                        probability,
                        valid,
                        setting,
                        connectivity=connectivity,
                    )
                    for component_area in component_areas:
                        prediction = _remove_small_components(
                            base_prediction, int(component_area)
                        ) & valid
                        key = (setting.key, int(component_area))
                        accumulator = accumulators.setdefault(key, _setting_accumulator())
                        event_key = (event_id, setting.key, int(component_area))
                        event_accumulator = event_accumulators.setdefault(
                            event_key, _setting_accumulator()
                        )
                        # Occurrence is not used here. A zero array keeps the shared
                        # metric helper focused on segmentation counts.
                        placeholder = np.zeros_like(target, dtype=np.uint8)
                        tile_metrics = _update_accumulator(
                            accumulator,
                            prediction,
                            target,
                            placeholder,
                            101,
                        )
                        _update_accumulator(
                            event_accumulator,
                            prediction,
                            target,
                            placeholder,
                            101,
                        )
                        valid_pixels = int(tile_metrics["valid_pixels"])
                        fg_pixels = int(tile_metrics["fg_pixels"])
                        tile_rows.append(
                            {
                                "event_id": event_id,
                                "file": image_path.name,
                                "strategy": setting.strategy,
                                "setting_key": setting.key,
                                "low_threshold": float(setting.low_threshold),
                                "high_threshold": float(setting.high_threshold),
                                "min_seed_pixels": int(setting.min_seed_pixels),
                                "min_component_area": int(component_area),
                                "foreground_bin": _foreground_bin(
                                    _safe_div(fg_pixels, valid_pixels)
                                ),
                                **tile_metrics,
                            }
                        )

    setting_lookup = {setting.key: setting for setting in settings}
    sweep_rows: List[Dict[str, Any]] = []
    for (setting_key, component_area), accumulator in accumulators.items():
        setting = setting_lookup[setting_key]
        sweep_rows.append(
            {
                "strategy": setting.strategy,
                "setting_key": setting.key,
                "low_threshold": float(setting.low_threshold),
                "high_threshold": float(setting.high_threshold),
                "min_seed_pixels": int(setting.min_seed_pixels),
                "min_component_area": int(component_area),
                **_finalize_accumulator(accumulator),
            }
        )
    sweep = pd.DataFrame(sweep_rows)

    tile_metrics = pd.DataFrame(tile_rows)
    tile_metrics_path = output_dir / "tile_setting_metrics.csv"
    tile_metrics.to_csv(tile_metrics_path, index=False)

    event_rows: List[Dict[str, Any]] = []
    for (event_id, setting_key, component_area), accumulator in event_accumulators.items():
        setting = setting_lookup[setting_key]
        event_rows.append(
            {
                "event_id": event_id,
                "strategy": setting.strategy,
                "setting_key": setting.key,
                "low_threshold": float(setting.low_threshold),
                "high_threshold": float(setting.high_threshold),
                "min_seed_pixels": int(setting.min_seed_pixels),
                "min_component_area": int(component_area),
                **_finalize_accumulator(accumulator),
            }
        )
    event_metrics = pd.DataFrame(event_rows)
    event_metrics_path = output_dir / "event_setting_metrics.csv"
    event_metrics.to_csv(event_metrics_path, index=False)

    reference_rows = sweep[
        sweep["strategy"].eq("fixed")
        & np.isclose(sweep["low_threshold"], float(reference_threshold))
        & sweep["min_component_area"].eq(int(reference_min_component_area))
    ]
    if len(reference_rows) != 1:
        raise RuntimeError("Could not identify a unique fixed-threshold reference")
    reference = reference_rows.iloc[0].to_dict()
    sweep = annotate_hysteresis_sweep(
        sweep,
        reference,
        max_recall_drop=max_recall_drop,
        max_empty_fp_rate_increase=max_empty_fp_rate_increase,
    ).sort_values(["f1", "iou", "mcc"], ascending=[False, False, False])
    sweep_path = output_dir / "hysteresis_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    best_fixed_unconstrained = choose_best_strategy(
        sweep, strategy="fixed", guarded=False
    )
    best_fixed_guarded = choose_best_strategy(sweep, strategy="fixed", guarded=True)
    best_hysteresis_unconstrained = choose_best_strategy(
        sweep, strategy="hysteresis", guarded=False
    )
    best_hysteresis_guarded = choose_best_strategy(
        sweep, strategy="hysteresis", guarded=True
    )
    recommendation, gain_over_best_fixed, endpoint_gain = hysteresis_recommendation(
        best_hysteresis_guarded,
        best_fixed_guarded,
    )

    selections = [
        ("reference", reference),
        ("best_fixed_unconstrained", best_fixed_unconstrained),
        ("best_fixed_guarded", best_fixed_guarded),
        ("best_hysteresis_unconstrained", best_hysteresis_unconstrained),
        ("best_hysteresis_guarded", best_hysteresis_guarded),
    ]
    comparison_rows: List[Dict[str, Any]] = []
    for label, selection in selections:
        if selection is None:
            continue
        subset = event_metrics[
            event_metrics["setting_key"].eq(selection["setting_key"])
            & event_metrics["min_component_area"].eq(
                int(selection["min_component_area"])
            )
        ].copy()
        subset.insert(0, "selection", label)
        comparison_rows.extend(subset.to_dict("records"))
    event_comparison = pd.DataFrame(comparison_rows)
    event_comparison_path = output_dir / "event_comparison.csv"
    event_comparison.to_csv(event_comparison_path, index=False)

    summary = {
        "checkpoint": str(checkpoint_path),
        "processed_data_dir": str(processed_data_dir),
        "split": split,
        "include_events": list(include_events or []),
        "exclude_events": list(exclude_events or []),
        "sample_count": int(len(indexed_dataset)),
        "modalities": list(modalities),
        "reference_setting": reference,
        "best_fixed_unconstrained_setting": best_fixed_unconstrained,
        "best_fixed_guarded_setting": best_fixed_guarded,
        "best_hysteresis_unconstrained_setting": best_hysteresis_unconstrained,
        "best_hysteresis_guarded_setting": best_hysteresis_guarded,
        "hysteresis_f1_gain_over_best_fixed": gain_over_best_fixed,
        "hysteresis_incremental_f1_gain_vs_best_endpoint": endpoint_gain,
        "max_recall_drop": float(max_recall_drop),
        "max_empty_fp_rate_increase": float(max_empty_fp_rate_increase),
        "connectivity": int(connectivity),
        "recommendation": recommendation,
        "decision_note": (
            "Hysteresis is compared with the best guard-eligible fixed threshold, "
            "and separately with the better fixed threshold at its low/high endpoints."
        ),
        "files": {
            "hysteresis_sweep": str(sweep_path),
            "tile_setting_metrics": str(tile_metrics_path),
            "event_setting_metrics": str(event_metrics_path),
            "event_comparison": str(event_comparison_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )

    LOG.info(
        "Hysteresis reference: f1=%.4f precision=%.4f recall=%.4f empty_fp=%.4f",
        float(reference["f1"]),
        float(reference["precision"]),
        float(reference["recall"]),
        float(reference["empty_fp_rate"]),
    )
    if best_hysteresis_guarded is not None:
        LOG.info(
            "Best guarded hysteresis: %s area=%d f1=%.4f gain_vs_best_fixed=%+.4f endpoint_gain=%+.4f",
            best_hysteresis_guarded["setting_key"],
            int(best_hysteresis_guarded["min_component_area"]),
            float(best_hysteresis_guarded["f1"]),
            float(gain_over_best_fixed or 0.0),
            float(endpoint_gain or 0.0),
        )
    LOG.info("Hysteresis audit written to: %s", output_dir)
    return summary
