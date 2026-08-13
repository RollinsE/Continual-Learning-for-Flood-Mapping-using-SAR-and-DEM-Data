from __future__ import annotations

import json
import math
import re
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from floods.error_audit import (
    IndexedDataset,
    _filter_dataset_by_events,
    _foreground_bin,
    _metrics_from_counts,
    _remove_small_components,
    _safe_div,
)
from floods.evaluation import BinaryThresholdSweep, load_checkpoint_state
from floods.utils.common import get_logger
from floods.utils.console import progress_iter

LOG = get_logger(__name__)

JRC_COLLECTION_ID = "jrc-gsw"
JRC_ASSET_KEY = "occurrence"
JRC_TEMPORAL_SCOPE = "1984-2020"
JRC_SOURCE_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PRIOR_SCHEMA_VERSION = "1"

DEFAULT_MODEL_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
DEFAULT_OCCURRENCE_THRESHOLDS = [75, 90, 95, 99]
DEFAULT_PENALTY_STRENGTHS = [0.25, 0.50, 0.75, 1.00]
DEFAULT_COMPONENT_AREAS = [96]


@dataclass(frozen=True)
class PriorSetting:
    strategy: str
    occurrence_threshold: int = 0
    penalty_strength: float = 0.0

    @property
    def key(self) -> str:
        if self.strategy == "none":
            return "none"
        if self.strategy == "hard_exclude":
            return f"hard_occ{self.occurrence_threshold:03d}"
        return (
            f"soft_occ{self.occurrence_threshold:03d}_"
            f"strength{self.penalty_strength:.2f}"
        )


def _event_id_from_name(name: str) -> str:
    match = re.search(r"(EMSR\d+)", str(name), flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


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


def build_prior_settings(
    occurrence_thresholds: Sequence[int],
    penalty_strengths: Sequence[float],
    include_hard_exclusion: bool = True,
    include_soft_penalty: bool = True,
) -> List[PriorSetting]:
    settings = [PriorSetting(strategy="none")]
    thresholds = sorted({int(value) for value in occurrence_thresholds})
    strengths = sorted({float(value) for value in penalty_strengths})

    for threshold in thresholds:
        if threshold < 0 or threshold > 100:
            raise ValueError("Occurrence thresholds must be between 0 and 100")
        if include_hard_exclusion:
            settings.append(
                PriorSetting(
                    strategy="hard_exclude",
                    occurrence_threshold=threshold,
                    penalty_strength=1.0,
                )
            )
        if include_soft_penalty:
            for strength in strengths:
                if strength < 0.0 or strength > 1.0:
                    raise ValueError("Penalty strengths must be between 0 and 1")
                settings.append(
                    PriorSetting(
                        strategy="soft_linear",
                        occurrence_threshold=threshold,
                        penalty_strength=strength,
                    )
                )
    return settings


def apply_water_prior(
    probability: np.ndarray,
    occurrence: np.ndarray,
    setting: PriorSetting,
) -> np.ndarray:
    """Return a probability map adjusted by a JRC occurrence prior.

    JRC occurrence values are percentages in [0, 100], with 255 representing
    no data. No-data pixels are left unchanged so missing prior coverage never
    suppresses a model prediction.
    """
    probability = np.asarray(probability, dtype=np.float32)
    occurrence = np.asarray(occurrence)
    if probability.shape != occurrence.shape:
        raise ValueError(
            f"Probability and occurrence shapes differ: {probability.shape} != {occurrence.shape}"
        )

    adjusted = probability.copy()
    if setting.strategy == "none":
        return adjusted

    prior_valid = np.isfinite(occurrence) & (occurrence >= 0) & (occurrence <= 100)
    active = prior_valid & (occurrence >= int(setting.occurrence_threshold))

    if setting.strategy == "hard_exclude":
        adjusted[active] = 0.0
        return adjusted

    if setting.strategy == "soft_linear":
        weight = np.zeros_like(adjusted, dtype=np.float32)
        weight[active] = occurrence[active].astype(np.float32) / 100.0
        multiplier = 1.0 - float(setting.penalty_strength) * weight
        adjusted *= np.clip(multiplier, 0.0, 1.0)
        return adjusted

    raise ValueError(f"Unsupported prior strategy: {setting.strategy}")


def _bbox_intersects(first: Sequence[float], second: Sequence[float]) -> bool:
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )


def _transform_close(first: Any, second: Any, tolerance: float = 1e-10) -> bool:
    return max(
        abs(float(left) - float(right))
        for left, right in zip(tuple(first)[:6], tuple(second)[:6])
    ) <= tolerance


class PlanetaryComputerOccurrenceProvider:
    """Fetch and align JRC Global Surface Water occurrence COGs.

    Network imports are lazy so the rest of the package remains usable without
    Planetary Computer dependencies. Tests can inject a provider implementing
    ``prepare`` and ``prior_path_for`` without any network access.
    """

    def __init__(
        self,
        cache_dir: Path,
        offline: bool = False,
        allow_incomplete: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = bool(offline)
        self.allow_incomplete = bool(allow_incomplete)
        self._catalog = None
        self._items_by_event: Dict[str, List[Any]] = {}
        self._asset_handles: Dict[str, Any] = {}
        self._exit_stack = ExitStack()

    def close(self) -> None:
        self._exit_stack.close()
        self._asset_handles.clear()

    def __enter__(self) -> "PlanetaryComputerOccurrenceProvider":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def prior_path_for(self, split: str, image_path: Path) -> Path:
        return self.cache_dir / str(split) / f"{Path(image_path).stem}.tif"

    def _ensure_dependencies(self) -> Tuple[Any, Any]:
        try:
            import planetary_computer  # type: ignore
            import pystac_client  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Water-prior acquisition requires `planetary-computer` and `pystac-client`. "
                "Install the release requirements before running the audit."
            ) from exc
        return planetary_computer, pystac_client

    def _catalog_client(self) -> Any:
        if self._catalog is not None:
            return self._catalog
        if self.offline:
            raise RuntimeError("Offline prior mode cannot query missing JRC occurrence tiles")
        planetary_computer, pystac_client = self._ensure_dependencies()
        self._catalog = pystac_client.Client.open(
            JRC_SOURCE_URL,
            modifier=planetary_computer.sign_inplace,
        )
        return self._catalog

    @staticmethod
    def _target_metadata(image_path: Path) -> Dict[str, Any]:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(image_path) as src:
            bounds_4326 = transform_bounds(
                src.crs,
                "EPSG:4326",
                *src.bounds,
                densify_pts=21,
            )
            return {
                "width": int(src.width),
                "height": int(src.height),
                "crs": src.crs,
                "transform": src.transform,
                "bounds_4326": tuple(float(value) for value in bounds_4326),
                "profile": src.profile.copy(),
            }

    def _cache_is_valid(self, path: Path, target: Mapping[str, Any]) -> bool:
        if not path.exists():
            return False
        try:
            import rasterio

            with rasterio.open(path) as src:
                tags = src.tags()
                return (
                    src.count == 1
                    and src.width == int(target["width"])
                    and src.height == int(target["height"])
                    and src.crs == target["crs"]
                    and _transform_close(src.transform, target["transform"])
                    and tags.get("prior_schema_version") == PRIOR_SCHEMA_VERSION
                    and tags.get("jrc_collection") == JRC_COLLECTION_ID
                    and tags.get("jrc_asset") == JRC_ASSET_KEY
                )
        except Exception:
            return False

    def _search_event_items(self, event_id: str, bounds: Sequence[float]) -> List[Any]:
        if event_id in self._items_by_event:
            return self._items_by_event[event_id]
        catalog = self._catalog_client()
        search = catalog.search(
            collections=[JRC_COLLECTION_ID],
            bbox=[float(value) for value in bounds],
        )
        items = list(search.items())
        if not items:
            raise RuntimeError(
                f"No Planetary Computer {JRC_COLLECTION_ID} items overlap event {event_id}"
            )
        self._items_by_event[event_id] = items
        LOG.info("JRC occurrence catalogue: event=%s items=%d", event_id, len(items))
        return items

    def _open_asset(self, href: str) -> Any:
        if href in self._asset_handles:
            return self._asset_handles[href]
        import rasterio

        environment = rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MAX_RETRY="5",
            GDAL_HTTP_RETRY_DELAY="2",
        )
        self._exit_stack.enter_context(environment)
        handle = self._exit_stack.enter_context(rasterio.open(href))
        self._asset_handles[href] = handle
        return handle

    def _write_aligned_prior(
        self,
        output_path: Path,
        target: Mapping[str, Any],
        items: Sequence[Any],
    ) -> Dict[str, Any]:
        import rasterio
        from rasterio.warp import Resampling, reproject

        output_path.parent.mkdir(parents=True, exist_ok=True)
        destination = np.full(
            (int(target["height"]), int(target["width"])),
            255,
            dtype=np.uint8,
        )
        item_ids: List[str] = []
        asset_hrefs: List[str] = []

        for item in items:
            item_bbox = list(item.bbox or [])
            if len(item_bbox) != 4 or not _bbox_intersects(item_bbox, target["bounds_4326"]):
                continue
            if JRC_ASSET_KEY not in item.assets:
                continue
            asset = item.assets[JRC_ASSET_KEY]
            href = str(asset.href)
            source = self._open_asset(href)
            warped_values = np.full_like(destination, 255)
            reproject(
                source=rasterio.band(source, 1),
                destination=warped_values,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=255,
                dst_transform=target["transform"],
                dst_crs=target["crs"],
                dst_nodata=255,
                resampling=Resampling.nearest,
            )
            valid = warped_values <= 100
            destination[valid] = warped_values[valid]
            item_ids.append(str(item.id))
            asset_hrefs.append(href.split("?", 1)[0])

        valid_fraction = float(np.mean(destination <= 100))
        if valid_fraction < 0.99 and not self.allow_incomplete:
            raise RuntimeError(
                f"JRC occurrence coverage is incomplete for {output_path.stem}: "
                f"{valid_fraction:.4%}. Use --allow-incomplete-prior only for a diagnostic run."
            )

        profile = {
            "driver": "GTiff",
            "width": int(target["width"]),
            "height": int(target["height"]),
            "count": 1,
            "dtype": "uint8",
            "crs": target["crs"],
            "transform": target["transform"],
            "nodata": 255,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(destination, 1)
            dst.set_band_description(1, "jrc_surface_water_occurrence_percent")
            dst.update_tags(
                prior_schema_version=PRIOR_SCHEMA_VERSION,
                jrc_collection=JRC_COLLECTION_ID,
                jrc_asset=JRC_ASSET_KEY,
                jrc_temporal_scope=JRC_TEMPORAL_SCOPE,
                jrc_item_ids=json.dumps(sorted(set(item_ids))),
                jrc_asset_hrefs=json.dumps(sorted(set(asset_hrefs))),
                valid_fraction=f"{valid_fraction:.10f}",
                attribution="Source: EC JRC/Google; hosted by Microsoft Planetary Computer",
            )
        return {
            "prior_path": str(output_path),
            "valid_fraction": valid_fraction,
            "item_count": len(set(item_ids)),
            "item_ids": json.dumps(sorted(set(item_ids))),
            "reused": False,
        }

    def prepare(
        self,
        split: str,
        image_paths: Sequence[Path],
    ) -> pd.DataFrame:
        image_paths = [Path(path) for path in image_paths]
        metadata_by_path: Dict[Path, Dict[str, Any]] = {}
        event_bounds: Dict[str, List[float]] = {}

        for image_path in image_paths:
            target = self._target_metadata(image_path)
            metadata_by_path[image_path] = target
            event_id = _event_id_from_name(image_path.name)
            bounds = target["bounds_4326"]
            if event_id not in event_bounds:
                event_bounds[event_id] = list(bounds)
            else:
                aggregate = event_bounds[event_id]
                aggregate[0] = min(aggregate[0], bounds[0])
                aggregate[1] = min(aggregate[1], bounds[1])
                aggregate[2] = max(aggregate[2], bounds[2])
                aggregate[3] = max(aggregate[3], bounds[3])

        if not self.offline:
            for event_id, bounds in event_bounds.items():
                self._search_event_items(event_id, bounds)

        rows: List[Dict[str, Any]] = []
        for image_path in image_paths:
            target = metadata_by_path[image_path]
            event_id = _event_id_from_name(image_path.name)
            output_path = self.prior_path_for(split, image_path)
            if self._cache_is_valid(output_path, target):
                import rasterio

                with rasterio.open(output_path) as src:
                    tags = src.tags()
                    valid_fraction = float(tags.get("valid_fraction", 0.0))
                    item_ids = tags.get("jrc_item_ids", "[]")
                rows.append(
                    {
                        "event_id": event_id,
                        "file": image_path.name,
                        "image_path": str(image_path),
                        "prior_path": str(output_path),
                        "valid_fraction": valid_fraction,
                        "item_count": len(json.loads(item_ids)),
                        "item_ids": item_ids,
                        "reused": True,
                    }
                )
                continue
            if self.offline:
                raise FileNotFoundError(
                    f"Offline prior cache is missing or stale: {output_path}"
                )
            result = self._write_aligned_prior(
                output_path,
                target,
                self._items_by_event[event_id],
            )
            rows.append(
                {
                    "event_id": event_id,
                    "file": image_path.name,
                    "image_path": str(image_path),
                    **result,
                }
            )
        return pd.DataFrame(rows)


def _read_prior_into_shape(prior_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    import rasterio

    with rasterio.open(prior_path) as src:
        prior = src.read(1)
    if prior.shape[0] > shape[0] or prior.shape[1] > shape[1]:
        raise ValueError(f"Prior shape {prior.shape} exceeds padded model shape {shape}")
    padded = np.full(shape, 255, dtype=np.uint8)
    padded[: prior.shape[0], : prior.shape[1]] = prior
    return padded


def _setting_accumulator() -> Dict[str, float]:
    return {
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "empty": 0.0,
        "empty_fp": 0.0,
        "nonempty": 0.0,
        "nonempty_detected": 0.0,
        "prior_valid_pixels": 0.0,
        "prior_active_pixels": 0.0,
        "prior_active_truth_pixels": 0.0,
        "prior_active_background_pixels": 0.0,
    }


def _finalize_accumulator(acc: Mapping[str, float]) -> Dict[str, float]:
    metrics = _metrics_from_counts(acc["tp"], acc["tn"], acc["fp"], acc["fn"])
    return {
        **metrics,
        "tp_pixels": float(acc["tp"]),
        "tn_pixels": float(acc["tn"]),
        "fp_pixels": float(acc["fp"]),
        "fn_pixels": float(acc["fn"]),
        "empty_fp_rate": _safe_div(acc["empty_fp"], acc["empty"]),
        "nonempty_tile_recall": _safe_div(
            acc["nonempty_detected"], acc["nonempty"]
        ),
        "prior_valid_pixels": float(acc["prior_valid_pixels"]),
        "prior_active_pixels": float(acc["prior_active_pixels"]),
        "prior_active_truth_pixels": float(acc["prior_active_truth_pixels"]),
        "prior_active_background_pixels": float(acc["prior_active_background_pixels"]),
        "prior_active_truth_fraction": _safe_div(
            acc["prior_active_truth_pixels"], acc["prior_active_pixels"]
        ),
    }


def _update_accumulator(
    acc: Dict[str, float],
    pred: np.ndarray,
    target: np.ndarray,
    occurrence: np.ndarray,
    occurrence_threshold: int,
) -> Dict[str, float]:
    valid = target != 255
    target_fg = (target > 0) & valid
    pred = pred & valid
    tp = float(np.count_nonzero(pred & target_fg))
    fp = float(np.count_nonzero(pred & (~target_fg) & valid))
    tn = float(np.count_nonzero((~pred) & (~target_fg) & valid))
    fn = float(np.count_nonzero((~pred) & target_fg))
    true_any = bool(np.any(target_fg))
    pred_any = bool(np.any(pred))
    prior_valid = valid & (occurrence <= 100)
    prior_active = prior_valid & (occurrence >= int(occurrence_threshold))

    acc["tp"] += tp
    acc["tn"] += tn
    acc["fp"] += fp
    acc["fn"] += fn
    acc["empty"] += float(not true_any)
    acc["empty_fp"] += float((not true_any) and pred_any)
    acc["nonempty"] += float(true_any)
    acc["nonempty_detected"] += float(true_any and pred_any)
    acc["prior_valid_pixels"] += float(np.count_nonzero(prior_valid))
    acc["prior_active_pixels"] += float(np.count_nonzero(prior_active))
    acc["prior_active_truth_pixels"] += float(np.count_nonzero(prior_active & target_fg))
    acc["prior_active_background_pixels"] += float(
        np.count_nonzero(prior_active & (~target_fg) & valid)
    )
    return {
        "tp_pixels": tp,
        "tn_pixels": tn,
        "fp_pixels": fp,
        "fn_pixels": fn,
        **_metrics_from_counts(tp, tn, fp, fn),
        "is_empty": bool(not true_any),
        "pred_any": bool(pred_any),
        "valid_pixels": int(np.count_nonzero(valid)),
        "fg_pixels": int(np.count_nonzero(target_fg)),
        "pred_pixels": int(np.count_nonzero(pred)),
        "prior_active_pixels": int(np.count_nonzero(prior_active)),
        "prior_active_truth_pixels": int(np.count_nonzero(prior_active & target_fg)),
        "prior_active_background_pixels": int(
            np.count_nonzero(prior_active & (~target_fg) & valid)
        ),
    }


def _sort_best_setting(rows: pd.DataFrame) -> Mapping[str, Any]:
    if rows.empty:
        raise ValueError("Cannot choose a setting from an empty sweep")
    sort_columns = ["f1", "iou", "mcc", "precision"]
    if "incremental_f1_gain_vs_same_threshold" in rows.columns:
        sort_columns.insert(1, "incremental_f1_gain_vs_same_threshold")
    return rows.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
    ).iloc[0].to_dict()


def _annotate_sweep(
    sweep: pd.DataFrame,
    reference: Mapping[str, Any],
    max_recall_drop: float,
) -> pd.DataFrame:
    """Add decision columns that separate prior gains from threshold tuning."""
    if sweep.empty:
        raise ValueError("Water-prior sweep is empty")

    annotated = sweep.copy()
    minimum_recall = float(reference["recall"]) - float(max_recall_drop)
    annotated["is_prior_setting"] = annotated["strategy"].ne("none")
    annotated["recall_guard_eligible"] = annotated["recall"] >= minimum_recall

    metric_columns = ["precision", "recall", "f1", "iou", "mcc"]
    for metric in metric_columns:
        annotated[f"{metric}_change_vs_reference"] = (
            annotated[metric].astype(float) - float(reference[metric])
        )

    none_rows = annotated[annotated["strategy"].eq("none")].copy()
    duplicate_baselines = none_rows.duplicated(
        ["model_threshold", "min_component_area"], keep=False
    )
    if duplicate_baselines.any():
        raise RuntimeError(
            "The sweep contains duplicate no-prior baselines for the same "
            "model threshold and component area"
        )

    matched_columns = ["model_threshold", "min_component_area", *metric_columns]
    matched_none = none_rows[matched_columns].rename(
        columns={metric: f"matched_none_{metric}" for metric in metric_columns}
    )
    annotated = annotated.merge(
        matched_none,
        on=["model_threshold", "min_component_area"],
        how="left",
        validate="many_to_one",
    )

    for metric in metric_columns:
        annotated[f"incremental_{metric}_gain_vs_same_threshold"] = (
            annotated[metric].astype(float)
            - annotated[f"matched_none_{metric}"].astype(float)
        )

    return annotated


def _choose_best_setting(
    sweep: pd.DataFrame,
    reference: Mapping[str, Any],
    max_recall_drop: float,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if sweep.empty:
        raise ValueError("Water-prior sweep is empty")
    best_unconstrained = _sort_best_setting(sweep)
    minimum_recall = float(reference["recall"]) - float(max_recall_drop)
    guarded = sweep[sweep["recall"] >= minimum_recall]
    best_guarded = (
        _sort_best_setting(guarded) if not guarded.empty else best_unconstrained
    )
    return best_unconstrained, best_guarded


def _choose_best_strategy_setting(
    sweep: pd.DataFrame,
    *,
    use_prior: bool,
    reference: Mapping[str, Any],
    max_recall_drop: float,
) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    strategy_rows = sweep[
        sweep["strategy"].ne("none") if use_prior else sweep["strategy"].eq("none")
    ]
    if strategy_rows.empty:
        return None, None

    best_unconstrained = _sort_best_setting(strategy_rows)
    minimum_recall = float(reference["recall"]) - float(max_recall_drop)
    guarded = strategy_rows[strategy_rows["recall"] >= minimum_recall]
    best_guarded = (
        _sort_best_setting(guarded) if not guarded.empty else None
    )
    return best_unconstrained, best_guarded


def _water_prior_recommendation(
    best_prior_guarded: Optional[Mapping[str, Any]],
    best_no_prior_guarded: Optional[Mapping[str, Any]],
) -> Tuple[str, Optional[float], Optional[float]]:
    """Judge the prior independently of model-threshold tuning."""
    if best_prior_guarded is None:
        return (
            "do_not_use_water_prior_no_recall_eligible_prior_setting",
            None,
            None,
        )
    if best_no_prior_guarded is None:
        return (
            "do_not_use_water_prior_missing_no_prior_comparator",
            None,
            None,
        )

    gain_over_best_no_prior = (
        float(best_prior_guarded["f1"]) - float(best_no_prior_guarded["f1"])
    )
    incremental_gain = float(
        best_prior_guarded["incremental_f1_gain_vs_same_threshold"]
    )

    if gain_over_best_no_prior >= 0.005 and incremental_gain >= 0.001:
        recommendation = "proceed_with_water_prior_postprocessing"
    elif gain_over_best_no_prior >= 0.001 and incremental_gain > 0.0:
        recommendation = "weak_prior_gain_review_event_tradeoffs"
    else:
        recommendation = "do_not_use_water_prior_no_incremental_gain"

    return recommendation, gain_over_best_no_prior, incremental_gain


def audit_water_prior(
    config: Any,
    checkpoint_path: Path,
    processed_data_dir: Path,
    output_dir: Path,
    prior_cache_dir: Path,
    split: str = "val",
    include_events: Optional[Iterable[str]] = None,
    exclude_events: Optional[Iterable[str]] = None,
    model_thresholds: Optional[Sequence[float]] = None,
    occurrence_thresholds: Optional[Sequence[int]] = None,
    penalty_strengths: Optional[Sequence[float]] = None,
    min_component_areas: Optional[Sequence[int]] = None,
    reference_threshold: float = 0.50,
    reference_min_component_area: int = 96,
    max_recall_drop: float = 0.02,
    include_hard_exclusion: bool = True,
    include_soft_penalty: bool = True,
    offline_prior_cache: bool = False,
    allow_incomplete_prior: bool = False,
    max_samples: Optional[int] = None,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate JRC surface-water occurrence as a post-processing prior.

    This command does not change model weights. It aligns the JRC occurrence
    layer to each processed tile, runs the retained checkpoint once, and sweeps
    hard exclusion and soft probability penalties. The summary reports both an
    unconstrained best setting and a recall-guarded best setting.
    """
    from floods.eval_collate import pad_segmentation_batch
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.utils.ml import seed_everything, seed_worker

    seed_everything(config.seed, deterministic=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_cache_dir = Path(prior_cache_dir)
    prior_cache_dir.mkdir(parents=True, exist_ok=True)

    config.data.path = str(processed_data_dir)
    dataset, modalities, _ = prepare_evaluation_dataset(config, split=split)
    _filter_dataset_by_events(dataset, include_events=include_events, exclude_events=exclude_events)
    if max_samples is not None and int(max_samples) > 0:
        active_indices = list(range(min(int(max_samples), len(dataset))))
    else:
        active_indices = list(range(len(dataset)))
    image_paths = [Path(dataset.image_files[index]) for index in active_indices]

    owns_provider = provider is None
    provider = provider or PlanetaryComputerOccurrenceProvider(
        cache_dir=prior_cache_dir,
        offline=offline_prior_cache,
        allow_incomplete=allow_incomplete_prior,
    )
    try:
        prior_index = provider.prepare(split, image_paths)
        prior_index_path = output_dir / "aligned_water_prior_index.csv"
        prior_index.to_csv(prior_index_path, index=False)

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

        thresholds = sorted(
            {float(value) for value in (model_thresholds or DEFAULT_MODEL_THRESHOLDS)}
            | {float(reference_threshold)}
        )
        occurrence_thresholds = sorted(
            {int(value) for value in (occurrence_thresholds or DEFAULT_OCCURRENCE_THRESHOLDS)}
        )
        strengths = sorted(
            {float(value) for value in (penalty_strengths or DEFAULT_PENALTY_STRENGTHS)}
        )
        component_areas = sorted(
            {int(value) for value in (min_component_areas or DEFAULT_COMPONENT_AREAS)}
            | {int(reference_min_component_area)}
        )
        settings = build_prior_settings(
            occurrence_thresholds,
            strengths,
            include_hard_exclusion=include_hard_exclusion,
            include_soft_penalty=include_soft_penalty,
        )

        accumulators: Dict[Tuple[str, float, int], Dict[str, float]] = {}
        event_accumulators: Dict[Tuple[str, str, float, int], Dict[str, float]] = {}
        tile_rows: List[Dict[str, Any]] = []
        overlap_accumulators: Dict[int, Dict[str, float]] = {
            threshold: {
                "valid_pixels": 0.0,
                "active_pixels": 0.0,
                "flood_pixels": 0.0,
                "background_pixels": 0.0,
                "active_flood_pixels": 0.0,
                "active_background_pixels": 0.0,
            }
            for threshold in occurrence_thresholds
        }

        LOG.info("Auditing water prior with checkpoint: %s", checkpoint_path)
        LOG.info(
            "Dataset: split=%s samples=%d settings=%d thresholds=%d component_areas=%s",
            split,
            len(indexed_dataset),
            len(settings),
            len(thresholds),
            component_areas,
        )

        with torch.no_grad():
            for x, y, index in progress_iter(
                loader,
                desc=f"Water-prior audit {split}",
                unit="batch",
                colour="blue",
            ):
                x = (
                    torch.nan_to_num(
                        x.float(), nan=0.0, posinf=30.0, neginf=-30.0
                    )
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
                    prior_path = Path(provider.prior_path_for(split, image_path))
                    occurrence = _read_prior_into_shape(prior_path, target.shape)
                    valid = target != 255
                    target_fg = (target > 0) & valid
                    prior_valid = valid & (occurrence <= 100)

                    for occurrence_threshold in occurrence_thresholds:
                        active = prior_valid & (occurrence >= occurrence_threshold)
                        overlap = overlap_accumulators[occurrence_threshold]
                        overlap["valid_pixels"] += float(np.count_nonzero(prior_valid))
                        overlap["active_pixels"] += float(np.count_nonzero(active))
                        overlap["flood_pixels"] += float(np.count_nonzero(target_fg))
                        overlap["background_pixels"] += float(
                            np.count_nonzero((~target_fg) & valid)
                        )
                        overlap["active_flood_pixels"] += float(
                            np.count_nonzero(active & target_fg)
                        )
                        overlap["active_background_pixels"] += float(
                            np.count_nonzero(active & (~target_fg) & valid)
                        )

                    for setting in settings:
                        adjusted = apply_water_prior(probability, occurrence, setting)
                        setting_occurrence_threshold = (
                            int(setting.occurrence_threshold)
                            if setting.strategy != "none"
                            else 101
                        )
                        for model_threshold in thresholds:
                            base_prediction = (adjusted >= float(model_threshold)) & valid
                            for component_area in component_areas:
                                prediction = _remove_small_components(
                                    base_prediction, component_area
                                ) & valid
                                key = (setting.key, float(model_threshold), int(component_area))
                                accumulator = accumulators.setdefault(
                                    key, _setting_accumulator()
                                )
                                event_key = (
                                    event_id,
                                    setting.key,
                                    float(model_threshold),
                                    int(component_area),
                                )
                                event_accumulator = event_accumulators.setdefault(
                                    event_key, _setting_accumulator()
                                )
                                tile_metrics = _update_accumulator(
                                    accumulator,
                                    prediction,
                                    target,
                                    occurrence,
                                    setting_occurrence_threshold,
                                )
                                _update_accumulator(
                                    event_accumulator,
                                    prediction,
                                    target,
                                    occurrence,
                                    setting_occurrence_threshold,
                                )
                                valid_pixels = int(tile_metrics["valid_pixels"])
                                fg_pixels = int(tile_metrics["fg_pixels"])
                                tile_rows.append(
                                    {
                                        "event_id": event_id,
                                        "file": image_path.name,
                                        "strategy": setting.strategy,
                                        "setting_key": setting.key,
                                        "occurrence_threshold": int(
                                            setting.occurrence_threshold
                                        ),
                                        "penalty_strength": float(
                                            setting.penalty_strength
                                        ),
                                        "model_threshold": float(model_threshold),
                                        "min_component_area": int(component_area),
                                        "foreground_bin": _foreground_bin(
                                            _safe_div(fg_pixels, valid_pixels)
                                        ),
                                        **tile_metrics,
                                    }
                                )

        setting_lookup = {setting.key: setting for setting in settings}
        sweep_rows: List[Dict[str, Any]] = []
        for (setting_key, model_threshold, component_area), accumulator in accumulators.items():
            setting = setting_lookup[setting_key]
            sweep_rows.append(
                {
                    "strategy": setting.strategy,
                    "setting_key": setting.key,
                    "occurrence_threshold": int(setting.occurrence_threshold),
                    "penalty_strength": float(setting.penalty_strength),
                    "model_threshold": float(model_threshold),
                    "min_component_area": int(component_area),
                    **_finalize_accumulator(accumulator),
                }
            )
        sweep = pd.DataFrame(sweep_rows)

        tile_metrics = pd.DataFrame(tile_rows)
        tile_metrics_path = output_dir / "tile_setting_metrics.csv"
        tile_metrics.to_csv(tile_metrics_path, index=False)

        event_rows: List[Dict[str, Any]] = []
        for (event_id, setting_key, model_threshold, component_area), accumulator in event_accumulators.items():
            setting = setting_lookup[setting_key]
            event_rows.append(
                {
                    "event_id": event_id,
                    "strategy": setting.strategy,
                    "setting_key": setting.key,
                    "occurrence_threshold": int(setting.occurrence_threshold),
                    "penalty_strength": float(setting.penalty_strength),
                    "model_threshold": float(model_threshold),
                    "min_component_area": int(component_area),
                    **_finalize_accumulator(accumulator),
                }
            )
        event_metrics = pd.DataFrame(event_rows)
        event_metrics_path = output_dir / "event_setting_metrics.csv"
        event_metrics.to_csv(event_metrics_path, index=False)

        overlap_rows: List[Dict[str, Any]] = []
        for occurrence_threshold, accumulator in overlap_accumulators.items():
            overlap_rows.append(
                {
                    "occurrence_threshold": int(occurrence_threshold),
                    **accumulator,
                    "active_fraction_of_prior_valid": _safe_div(
                        accumulator["active_pixels"], accumulator["valid_pixels"]
                    ),
                    "flood_pixels_flagged_fraction": _safe_div(
                        accumulator["active_flood_pixels"], accumulator["flood_pixels"]
                    ),
                    "background_pixels_flagged_fraction": _safe_div(
                        accumulator["active_background_pixels"],
                        accumulator["background_pixels"],
                    ),
                    "flagged_pixels_truth_fraction": _safe_div(
                        accumulator["active_flood_pixels"], accumulator["active_pixels"]
                    ),
                }
            )
        overlap_df = pd.DataFrame(overlap_rows).sort_values("occurrence_threshold")
        overlap_path = output_dir / "prior_label_overlap.csv"
        overlap_df.to_csv(overlap_path, index=False)

        reference_rows = sweep[
            (sweep["strategy"] == "none")
            & np.isclose(sweep["model_threshold"], float(reference_threshold))
            & (sweep["min_component_area"] == int(reference_min_component_area))
        ]
        if len(reference_rows) != 1:
            raise RuntimeError(
                "Could not identify a unique baseline reference setting in the sweep"
            )
        reference = reference_rows.iloc[0].to_dict()
        sweep = _annotate_sweep(
            sweep,
            reference,
            max_recall_drop=max_recall_drop,
        ).sort_values(
            ["f1", "iou", "mcc"],
            ascending=[False, False, False],
        )
        sweep_path = output_dir / "water_prior_sweep.csv"
        sweep.to_csv(sweep_path, index=False)

        best_unconstrained, best_guarded = _choose_best_setting(
            sweep, reference, max_recall_drop=max_recall_drop
        )
        best_no_prior_unconstrained, best_no_prior_guarded = (
            _choose_best_strategy_setting(
                sweep,
                use_prior=False,
                reference=reference,
                max_recall_drop=max_recall_drop,
            )
        )
        best_prior_unconstrained, best_prior_guarded = (
            _choose_best_strategy_setting(
                sweep,
                use_prior=True,
                reference=reference,
                max_recall_drop=max_recall_drop,
            )
        )

        comparison_rows: List[Dict[str, Any]] = []
        selections = [
            ("reference", reference),
            ("best_overall_unconstrained", best_unconstrained),
            ("best_overall_recall_guarded", best_guarded),
            ("best_no_prior_unconstrained", best_no_prior_unconstrained),
            ("best_no_prior_recall_guarded", best_no_prior_guarded),
            ("best_prior_unconstrained", best_prior_unconstrained),
            ("best_prior_recall_guarded", best_prior_guarded),
        ]
        for label, selection in selections:
            if selection is None:
                continue
            subset = event_metrics[
                (event_metrics["setting_key"] == selection["setting_key"])
                & np.isclose(
                    event_metrics["model_threshold"],
                    float(selection["model_threshold"]),
                )
                & (
                    event_metrics["min_component_area"]
                    == int(selection["min_component_area"])
                )
            ].copy()
            subset.insert(0, "selection", label)
            comparison_rows.extend(subset.to_dict("records"))
        event_comparison = pd.DataFrame(comparison_rows)
        event_comparison_path = output_dir / "event_comparison.csv"
        event_comparison.to_csv(event_comparison_path, index=False)

        unconstrained_gain = float(best_unconstrained["f1"]) - float(reference["f1"])
        guarded_gain = float(best_guarded["f1"]) - float(reference["f1"])

        prior_unconstrained_gain = (
            float(best_prior_unconstrained["f1"]) - float(reference["f1"])
            if best_prior_unconstrained is not None
            else None
        )
        prior_guarded_gain = (
            float(best_prior_guarded["f1"]) - float(reference["f1"])
            if best_prior_guarded is not None
            else None
        )
        (
            recommendation,
            prior_gain_over_best_no_prior,
            selected_prior_incremental_gain,
        ) = _water_prior_recommendation(
            best_prior_guarded,
            best_no_prior_guarded,
        )

        summary = {
            "checkpoint": str(checkpoint_path),
            "processed_data_dir": str(processed_data_dir),
            "split": split,
            "include_events": list(include_events or []),
            "exclude_events": list(exclude_events or []),
            "sample_count": int(len(indexed_dataset)),
            "modalities": list(modalities),
            "prior_source": {
                "collection": JRC_COLLECTION_ID,
                "asset": JRC_ASSET_KEY,
                "temporal_scope": JRC_TEMPORAL_SCOPE,
                "host": "Microsoft Planetary Computer",
                "producer": "European Commission Joint Research Centre",
                "important_note": (
                    "The occurrence prior summarises 1984-2020. For events before 2020, "
                    "this audit is diagnostic and may contain future temporal information."
                ),
            },
            "reference_setting": reference,
            "best_unconstrained_setting": best_unconstrained,
            "best_recall_guarded_setting": best_guarded,
            "best_no_prior_unconstrained_setting": best_no_prior_unconstrained,
            "best_no_prior_recall_guarded_setting": best_no_prior_guarded,
            "best_prior_unconstrained_setting": best_prior_unconstrained,
            "best_prior_recall_guarded_setting": best_prior_guarded,
            "unconstrained_f1_gain": unconstrained_gain,
            "recall_guarded_f1_gain": guarded_gain,
            "threshold_tuning_f1_gain": (
                float(best_no_prior_guarded["f1"]) - float(reference["f1"])
                if best_no_prior_guarded is not None
                else None
            ),
            "prior_unconstrained_f1_gain_vs_reference": prior_unconstrained_gain,
            "prior_recall_guarded_f1_gain_vs_reference": prior_guarded_gain,
            "water_prior_f1_gain_over_best_no_prior": prior_gain_over_best_no_prior,
            "water_prior_incremental_f1_gain_at_selected_threshold": (
                selected_prior_incremental_gain
            ),
            "max_recall_drop": float(max_recall_drop),
            "recommendation": recommendation,
            "decision_note": (
                "The recommendation is based on the best recall-eligible prior "
                "setting versus the best recall-eligible no-prior setting. A gain "
                "caused only by changing the model threshold is not counted as a "
                "water-prior gain."
            ),
            "files": {
                "aligned_water_prior_index": str(prior_index_path),
                "water_prior_sweep": str(sweep_path),
                "tile_setting_metrics": str(tile_metrics_path),
                "event_setting_metrics": str(event_metrics_path),
                "event_comparison": str(event_comparison_path),
                "prior_label_overlap": str(overlap_path),
                "prior_cache_dir": str(prior_cache_dir),
            },
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )

        LOG.info(
            "Water-prior reference: f1=%.4f precision=%.4f recall=%.4f",
            float(reference["f1"]),
            float(reference["precision"]),
            float(reference["recall"]),
        )
        if best_prior_guarded is not None:
            LOG.info(
                "Best recall-guarded water prior: %s threshold=%.2f area=%d "
                "f1=%.4f gain_vs_best_no_prior=%+.4f matched_threshold_gain=%+.4f",
                best_prior_guarded["setting_key"],
                float(best_prior_guarded["model_threshold"]),
                int(best_prior_guarded["min_component_area"]),
                float(best_prior_guarded["f1"]),
                float(prior_gain_over_best_no_prior or 0.0),
                float(selected_prior_incremental_gain or 0.0),
            )
        else:
            LOG.info("No recall-eligible water-prior setting was found")
        LOG.info("Water-prior audit written to: %s", output_dir)
        return summary
    finally:
        if owns_provider and hasattr(provider, "close"):
            provider.close()
