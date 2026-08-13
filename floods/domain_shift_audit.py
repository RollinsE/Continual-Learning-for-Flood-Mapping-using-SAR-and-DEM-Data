from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from floods.derived_features import read_processed_modalities
from floods.modalities import canonicalize_modalities
from floods.normalization import load_normalization_stats
from floods.utils.common import get_logger
from floods.utils.console import progress_iter
from floods.utils.gis import imread

LOG = get_logger(__name__)

_EVENT_RE = re.compile(r"(EMSR\d+)")
_TILE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _event_id(name: str) -> str:
    match = _EVENT_RE.search(str(name))
    return match.group(1) if match else "unknown"


def _select_tiles(
    processed_data_dir: Path,
    split: str,
    *,
    include_events: Iterable[str] | None,
    exclude_events: Iterable[str] | None,
    max_tiles: int,
    seed: int,
) -> list[Path]:
    paths = sorted((Path(processed_data_dir) / split / "mask").glob("*.tif"))
    if not paths:
        raise FileNotFoundError(
            f"No mask tiles found for split '{split}' under {Path(processed_data_dir) / split / 'mask'}"
        )
    include = {str(x).upper() for x in (include_events or [])}
    exclude = {str(x).upper() for x in (exclude_events or [])}
    paths = [
        path
        for path in paths
        if (not include or _event_id(path.name).upper() in include)
        and _event_id(path.name).upper() not in exclude
    ]
    if not paths:
        detail = f" include_events={sorted(include)}" if include else ""
        raise ValueError(f"No tiles remain for split '{split}' after event filtering.{detail}")
    max_tiles = int(max_tiles or 0)
    if max_tiles <= 0 or len(paths) <= max_tiles:
        return paths

    rng = np.random.default_rng(seed)
    by_event: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_event[_event_id(path.name)].append(path)
    selected: list[Path] = []
    events = sorted(by_event)
    # Round-robin, event-stratified sampling prevents large events from consuming
    # the entire cap while remaining deterministic for a fixed seed.
    shuffled = {}
    for event in events:
        values = list(by_event[event])
        order = rng.permutation(len(values))
        shuffled[event] = [values[int(i)] for i in order]
    depth = 0
    while len(selected) < max_tiles:
        added = False
        for event in events:
            values = shuffled[event]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) >= max_tiles:
                    break
        if not added:
            break
        depth += 1
    return sorted(selected)


def _safe_quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"p05": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "p95": math.nan}
    q = np.quantile(values, _TILE_QUANTILES)
    return {
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def _sample(values: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    maximum = int(maximum or 0)
    if maximum > 0 and values.size > maximum:
        idx = rng.choice(values.size, size=maximum, replace=False)
        values = values[idx]
    return values.astype(np.float32, copy=False)


def _mask_geometry(mask: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(mask).squeeze()
    valid = mask != 255
    flood = (mask == 1) & valid
    valid_pixels = int(valid.sum())
    fg_pixels = int(flood.sum())
    valid_ratio = float(valid_pixels / mask.size) if mask.size else 0.0
    fg_ratio = float(fg_pixels / valid_pixels) if valid_pixels else 0.0
    if fg_pixels == 0:
        return {
            "valid_pixels": valid_pixels,
            "valid_ratio": valid_ratio,
            "fg_pixels": 0,
            "fg_ratio": 0.0,
            "empty_tile": 1,
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_fraction": 0.0,
            "median_component_pixels": 0.0,
            "boundary_pixels": 0,
            "boundary_area_ratio": 0.0,
        }

    labelled, count = ndimage.label(flood, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(labelled.reshape(-1))[1:]
    largest = int(component_sizes.max()) if component_sizes.size else 0
    median_component = float(np.median(component_sizes)) if component_sizes.size else 0.0
    eroded = ndimage.binary_erosion(flood, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary_pixels = int(np.count_nonzero(flood & ~eroded))
    return {
        "valid_pixels": valid_pixels,
        "valid_ratio": valid_ratio,
        "fg_pixels": fg_pixels,
        "fg_ratio": fg_ratio,
        "empty_tile": 0,
        "component_count": int(count),
        "largest_component_pixels": largest,
        "largest_component_fraction": float(largest / fg_pixels) if fg_pixels else 0.0,
        "median_component_pixels": median_component,
        "boundary_pixels": boundary_pixels,
        "boundary_area_ratio": float(boundary_pixels / fg_pixels) if fg_pixels else 0.0,
    }


def _load_clip_stats(
    path: Path | None,
    modalities: Sequence[str],
    normalization_mode: str,
) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization statistics file not found: {path}")
    # Validate channel availability using the same loader as model preparation.
    _, _, clip_min, clip_max = load_normalization_stats(path, modalities, mode=normalization_mode)
    return {
        modality: {"clip_min": float(lo), "clip_max": float(hi)}
        for modality, lo, hi in zip(modalities, clip_min, clip_max)
    }


def _extract_domain(
    processed_data_dir: Path,
    split: str,
    paths: Sequence[Path],
    *,
    domain: str,
    modalities: Sequence[str],
    clip_stats: dict[str, dict[str, float]],
    max_pixels_per_tile: int,
    max_pixels_per_class_per_tile: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], list[np.ndarray]]]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    samples: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for path in progress_iter(paths, desc=f"Domain audit {domain}", unit="tile", colour="green"):
        stem = path.stem
        mask = imread(path, channels_first=True).squeeze()
        image = read_processed_modalities(processed_data_dir, split, stem, modalities)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Spatial mismatch for {stem}: image={image.shape[:2]} mask={mask.shape}")
        valid = mask != 255
        flood = (mask == 1) & valid
        background = (~flood) & valid
        finite_all = np.all(np.isfinite(image), axis=-1)
        row: dict[str, float | int | str] = {
            "domain": domain,
            "split": split,
            "event_id": _event_id(path.name),
            "file": path.name,
        }
        row.update(_mask_geometry(mask))
        for index, modality in enumerate(modalities):
            channel = image[..., index].astype(np.float32, copy=False)
            channel_finite = np.isfinite(channel)
            group_masks = {
                "all": valid & channel_finite,
                "flood": flood & channel_finite,
                "background": background & channel_finite,
            }
            for stratum, group_mask in group_masks.items():
                values = channel[group_mask]
                maximum = max_pixels_per_tile if stratum == "all" else max_pixels_per_class_per_tile
                sampled = _sample(values, maximum, rng)
                if sampled.size:
                    samples[(modality, stratum)].append(sampled)
            values = channel[valid & channel_finite]
            prefix = f"{modality}_"
            if values.size:
                q = _safe_quantiles(values)
                row[prefix + "mean"] = float(np.mean(values))
                row[prefix + "std"] = float(np.std(values))
                for name, value in q.items():
                    row[prefix + name] = value
            else:
                row[prefix + "mean"] = math.nan
                row[prefix + "std"] = math.nan
                for name in ("p05", "p25", "p50", "p75", "p95"):
                    row[prefix + name] = math.nan
            for stratum, group_mask in (("flood", flood), ("background", background)):
                group_values = channel[group_mask & channel_finite]
                row[f"{modality}_{stratum}_mean"] = float(np.mean(group_values)) if group_values.size else math.nan
                row[f"{modality}_{stratum}_p50"] = float(np.median(group_values)) if group_values.size else math.nan
            if modality in clip_stats and values.size:
                lo = clip_stats[modality]["clip_min"]
                hi = clip_stats[modality]["clip_max"]
                row[prefix + "clip_low_rate"] = float(np.mean(values < lo))
                row[prefix + "clip_high_rate"] = float(np.mean(values > hi))
                row[prefix + "clip_total_rate"] = float(np.mean((values < lo) | (values > hi)))
        row["all_modalities_finite_ratio"] = float(np.mean(valid & finite_all)) if mask.size else 0.0
        rows.append(row)
    return pd.DataFrame(rows), samples


def _cap_concatenated(parts: Sequence[np.ndarray], maximum: int, rng: np.random.Generator) -> np.ndarray:
    if not parts:
        return np.asarray([], dtype=np.float32)
    values = np.concatenate(parts).astype(np.float32, copy=False)
    maximum = int(maximum or 0)
    if maximum > 0 and values.size > maximum:
        idx = rng.choice(values.size, size=maximum, replace=False)
        values = values[idx]
    return values


def _psi(reference: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    target = target[np.isfinite(target)]
    if reference.size == 0 or target.size == 0:
        return math.nan
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, int(bins) + 1)))
    if edges.size < 3:
        return 0.0
    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist = np.histogram(reference, bins=edges)[0].astype(np.float64)
    tgt_hist = np.histogram(target, bins=edges)[0].astype(np.float64)
    eps = 1e-6
    ref_prop = np.maximum(ref_hist / max(ref_hist.sum(), 1.0), eps)
    tgt_prop = np.maximum(tgt_hist / max(tgt_hist.sum(), 1.0), eps)
    return float(np.sum((tgt_prop - ref_prop) * np.log(tgt_prop / ref_prop)))


def _compare_values(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    feature: str,
    stratum: str,
    feature_type: str,
) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    target = target[np.isfinite(target)]
    if reference.size == 0 or target.size == 0:
        return {
            "feature_type": feature_type,
            "feature": feature,
            "stratum": stratum,
            "reference_count": int(reference.size),
            "target_count": int(target.size),
        }
    ref_q = np.quantile(reference, [0.05, 0.25, 0.5, 0.75, 0.95])
    tgt_q = np.quantile(target, [0.05, 0.25, 0.5, 0.75, 0.95])
    ks = ks_2samp(reference, target, alternative="two-sided", method="auto")
    wdist = float(wasserstein_distance(reference, target))
    ref_iqr = float(ref_q[3] - ref_q[1])
    pooled_std = float(math.sqrt((np.var(reference) + np.var(target)) / 2.0))
    scale = ref_iqr if ref_iqr > 1e-9 else (float(np.std(reference)) if np.std(reference) > 1e-9 else 1.0)
    effect = float((np.mean(target) - np.mean(reference)) / pooled_std) if pooled_std > 1e-9 else 0.0
    normalized_wasserstein = float(wdist / scale)
    psi = _psi(reference, target)
    if normalized_wasserstein >= 1.0 or float(ks.statistic) >= 0.40 or (math.isfinite(psi) and psi >= 0.50):
        level = "strong"
    elif normalized_wasserstein >= 0.50 or float(ks.statistic) >= 0.20 or (math.isfinite(psi) and psi >= 0.20):
        level = "moderate"
    else:
        level = "low"
    return {
        "feature_type": feature_type,
        "feature": feature,
        "stratum": stratum,
        "reference_count": int(reference.size),
        "target_count": int(target.size),
        "reference_mean": float(np.mean(reference)),
        "target_mean": float(np.mean(target)),
        "reference_std": float(np.std(reference)),
        "target_std": float(np.std(target)),
        "reference_p05": float(ref_q[0]),
        "target_p05": float(tgt_q[0]),
        "reference_p50": float(ref_q[2]),
        "target_p50": float(tgt_q[2]),
        "reference_p95": float(ref_q[4]),
        "target_p95": float(tgt_q[4]),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": wdist,
        "normalized_wasserstein": normalized_wasserstein,
        "standardized_mean_difference": effect,
        "psi": psi,
        "shift_level": level,
    }


def _domain_classifier(
    tile_features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    seed: int,
    reference_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = tile_features[tile_features["domain"] == "reference"].copy()
    target = tile_features[tile_features["domain"] == "target"].copy()
    if reference.empty or target.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    max_reference = min(len(reference), max(1, int(round(len(target) * float(reference_ratio)))))
    if len(reference) > max_reference:
        by_event = {event: group.index.to_numpy() for event, group in reference.groupby("event_id")}
        selected: list[int] = []
        depth = 0
        shuffled = {event: rng.permutation(indices) for event, indices in by_event.items()}
        while len(selected) < max_reference:
            added = False
            for event in sorted(shuffled):
                values = shuffled[event]
                if depth < len(values):
                    selected.append(int(values[depth]))
                    added = True
                    if len(selected) >= max_reference:
                        break
            if not added:
                break
            depth += 1
        reference = reference.loc[selected]
    frame = pd.concat([reference, target], ignore_index=True)
    y = (frame["domain"] == "target").astype(np.uint8).to_numpy()
    metrics_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    min_class = int(np.bincount(y).min()) if len(np.unique(y)) == 2 else 0
    n_splits = min(5, min_class)
    for set_name, columns in feature_sets.items():
        usable = [column for column in columns if column in frame.columns and not frame[column].isna().all()]
        if not usable or n_splits < 2:
            metrics_rows.append(
                {
                    "feature_set": set_name,
                    "features": len(usable),
                    "reference_tiles": int((y == 0).sum()),
                    "target_tiles": int((y == 1).sum()),
                    "roc_auc": math.nan,
                    "average_precision": math.nan,
                    "interpretation": "insufficient_data",
                }
            )
            continue
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        probability = cross_val_predict(
            pipeline,
            frame[usable],
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        auc = float(roc_auc_score(y, probability))
        ap = float(average_precision_score(y, probability))
        if auc >= 0.90:
            interpretation = "very_strong_domain_separation"
        elif auc >= 0.80:
            interpretation = "strong_domain_separation"
        elif auc >= 0.70:
            interpretation = "moderate_domain_separation"
        else:
            interpretation = "weak_domain_separation"
        metrics_rows.append(
            {
                "feature_set": set_name,
                "features": len(usable),
                "reference_tiles": int((y == 0).sum()),
                "target_tiles": int((y == 1).sum()),
                "cv_folds": n_splits,
                "roc_auc": auc,
                "average_precision": ap,
                "interpretation": interpretation,
            }
        )
        pipeline.fit(frame[usable], y)
        coefficients = pipeline.named_steps["model"].coef_[0]
        for feature, coefficient in zip(usable, coefficients):
            coefficient_rows.append(
                {
                    "feature_set": set_name,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "absolute_coefficient": float(abs(coefficient)),
                    "direction": "higher_in_target" if coefficient > 0 else "lower_in_target",
                }
            )
    coefficients = pd.DataFrame(coefficient_rows)
    if not coefficients.empty:
        coefficients = coefficients.sort_values(
            ["feature_set", "absolute_coefficient"], ascending=[True, False]
        )
    return pd.DataFrame(metrics_rows), coefficients


def _event_similarity(
    tile_features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = tile_features[tile_features["domain"] == "reference"].copy()
    target = tile_features[tile_features["domain"] == "target"].copy()
    if reference.empty or target.empty:
        return pd.DataFrame(), pd.DataFrame()
    all_features = []
    for columns in feature_sets.values():
        for column in columns:
            if column in tile_features.columns and column not in all_features and not tile_features[column].isna().all():
                all_features.append(column)
    event_medians = reference.groupby("event_id")[all_features].median(numeric_only=True)
    event_counts = reference.groupby("event_id").size().rename("tiles")
    target_vector = target[all_features].median(numeric_only=True)
    centre = event_medians.median(axis=0)
    scale = event_medians.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    target_z = (target_vector - centre) / scale
    z_rows = []
    for feature in all_features:
        z_rows.append(
            {
                "feature": feature,
                "target_median": float(target_vector.get(feature, math.nan)),
                "training_event_median": float(centre.get(feature, math.nan)),
                "training_event_std": float(scale.get(feature, math.nan)),
                "zscore": float(target_z.get(feature, math.nan)),
                "absolute_zscore": float(abs(target_z.get(feature, math.nan))),
            }
        )
    zscores = pd.DataFrame(z_rows).sort_values("absolute_zscore", ascending=False)

    distance_rows = []
    standardized_events = (event_medians - centre) / scale
    for event, event_vector in standardized_events.iterrows():
        row = {"event_id": event, "tiles": int(event_counts.loc[event])}
        for set_name, columns in feature_sets.items():
            usable = [column for column in columns if column in standardized_events.columns]
            if not usable:
                row[f"distance_{set_name}"] = math.nan
                continue
            delta = event_vector[usable].to_numpy(dtype=np.float64) - target_z[usable].to_numpy(dtype=np.float64)
            row[f"distance_{set_name}"] = float(np.sqrt(np.mean(delta ** 2)))
        distance_rows.append(row)
    similarity = pd.DataFrame(distance_rows)
    if "distance_combined" in similarity.columns:
        similarity = similarity.sort_values("distance_combined", ascending=True).reset_index(drop=True)
        similarity.insert(0, "combined_rank", np.arange(1, len(similarity) + 1))
    return similarity, zscores


def _write_plots(
    output_dir: Path,
    pixel_shift: pd.DataFrame,
    zscores: pd.DataFrame,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        if not pixel_shift.empty:
            view = pixel_shift[pixel_shift["stratum"] == "all"].copy()
            view = view.sort_values("normalized_wasserstein", ascending=False)
            plt.figure(figsize=(9, 5))
            plt.bar(view["feature"], view["normalized_wasserstein"])
            plt.ylabel("Wasserstein distance / training IQR")
            plt.xlabel("Input modality")
            plt.title("Target event input-distribution shift")
            plt.tight_layout()
            plt.savefig(output_dir / "input_distribution_shift.png", dpi=160)
            plt.close()
        if not zscores.empty:
            view = zscores.head(20).sort_values("zscore")
            plt.figure(figsize=(10, 7))
            plt.barh(view["feature"], view["zscore"])
            plt.xlabel("Target median z-score vs training-event medians")
            plt.title("Most unusual target-event tile features")
            plt.tight_layout()
            plt.savefig(output_dir / "target_feature_zscores.png", dpi=160)
            plt.close()
    except Exception as exc:
        LOG.warning("Domain-shift plots were not written: %s", exc)


def audit_domain_shift(
    processed_data_dir: Path,
    output_dir: Path,
    *,
    reference_split: str = "train",
    target_split: str = "val",
    target_events: Sequence[str],
    reference_events: Sequence[str] | None = None,
    exclude_reference_events: Sequence[str] | None = None,
    input_modalities: Sequence[str] = ("vv", "vh", "dem"),
    normalization_stats_path: Path | None = None,
    normalization_mode: str = "robust_percentile",
    max_reference_tiles: int = 0,
    max_target_tiles: int = 0,
    max_pixels_per_tile: int = 256,
    max_pixels_per_class_per_tile: int = 128,
    max_total_pixels_per_domain: int = 250_000,
    domain_classifier_reference_ratio: float = 4.0,
    seed: int = 42,
    write_plots: bool = True,
) -> dict:
    """Compare a target event with the training domain at pixel, tile and event levels.

    The audit deliberately separates sensor/terrain features from label geometry so a
    weak model result is not automatically described as generic domain shift.
    """
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modalities = canonicalize_modalities(input_modalities)
    target_events = [str(event).upper() for event in target_events]
    if not target_events:
        raise ValueError("At least one --target-events value is required")
    clip_stats = _load_clip_stats(normalization_stats_path, modalities, normalization_mode)

    reference_paths = _select_tiles(
        processed_data_dir,
        reference_split,
        include_events=reference_events,
        exclude_events=exclude_reference_events,
        max_tiles=max_reference_tiles,
        seed=seed,
    )
    target_paths = _select_tiles(
        processed_data_dir,
        target_split,
        include_events=target_events,
        exclude_events=None,
        max_tiles=max_target_tiles,
        seed=seed + 1,
    )
    LOG.info(
        "Domain-shift audit: reference=%s (%d tiles, %d events) | target=%s events=%s (%d tiles)",
        reference_split,
        len(reference_paths),
        len({_event_id(path.name) for path in reference_paths}),
        target_split,
        target_events,
        len(target_paths),
    )
    LOG.info("Input modalities: %s", modalities)
    if normalization_stats_path is not None:
        LOG.info(
            "Normalization clipping audit: mode=%s stats=%s",
            normalization_mode,
            normalization_stats_path,
        )

    reference_df, reference_samples = _extract_domain(
        processed_data_dir,
        reference_split,
        reference_paths,
        domain="reference",
        modalities=modalities,
        clip_stats=clip_stats,
        max_pixels_per_tile=max_pixels_per_tile,
        max_pixels_per_class_per_tile=max_pixels_per_class_per_tile,
        seed=seed,
    )
    target_df, target_samples = _extract_domain(
        processed_data_dir,
        target_split,
        target_paths,
        domain="target",
        modalities=modalities,
        clip_stats=clip_stats,
        max_pixels_per_tile=max_pixels_per_tile,
        max_pixels_per_class_per_tile=max_pixels_per_class_per_tile,
        seed=seed + 1,
    )
    tile_features = pd.concat([reference_df, target_df], ignore_index=True)
    tile_features.to_csv(output_dir / "tile_features.csv", index=False)

    rng = np.random.default_rng(seed + 2)
    pixel_rows = []
    for modality in modalities:
        for stratum in ("all", "flood", "background"):
            reference_values = _cap_concatenated(
                reference_samples.get((modality, stratum), []), max_total_pixels_per_domain, rng
            )
            target_values = _cap_concatenated(
                target_samples.get((modality, stratum), []), max_total_pixels_per_domain, rng
            )
            pixel_rows.append(
                _compare_values(
                    reference_values,
                    target_values,
                    feature=modality,
                    stratum=stratum,
                    feature_type="pixel_distribution",
                )
            )
    pixel_shift = pd.DataFrame(pixel_rows)
    if not pixel_shift.empty and "normalized_wasserstein" in pixel_shift.columns:
        pixel_shift = pixel_shift.sort_values(
            ["stratum", "normalized_wasserstein"], ascending=[True, False]
        )
    pixel_shift.to_csv(output_dir / "pixel_distribution_shift.csv", index=False)

    geometry_features = [
        "valid_ratio",
        "fg_ratio",
        "empty_tile",
        "component_count",
        "largest_component_pixels",
        "largest_component_fraction",
        "median_component_pixels",
        "boundary_area_ratio",
    ]
    clip_features = []
    deployable_sensor_features = ["all_modalities_finite_ratio"]
    label_conditioned_input_features = []
    for modality in modalities:
        deployable_sensor_features.extend(
            [
                f"{modality}_mean",
                f"{modality}_std",
                f"{modality}_p05",
                f"{modality}_p25",
                f"{modality}_p50",
                f"{modality}_p75",
                f"{modality}_p95",
            ]
        )
        label_conditioned_input_features.extend(
            [
                f"{modality}_flood_mean",
                f"{modality}_flood_p50",
                f"{modality}_background_mean",
                f"{modality}_background_p50",
            ]
        )
        if modality in clip_stats:
            clip_features.extend(
                [
                    f"{modality}_clip_low_rate",
                    f"{modality}_clip_high_rate",
                    f"{modality}_clip_total_rate",
                ]
            )
    deployable_sensor_features.extend(clip_features)
    # Stable aggregate name retained for existing consumers. Unlike
    # deployable_sensor_terrain, sensor_terrain includes mask-stratified input
    # summaries and therefore is not available at inference time.
    sensor_features = deployable_sensor_features + label_conditioned_input_features
    tile_shift_rows = []
    for feature in sensor_features + geometry_features:
        if feature not in tile_features.columns:
            continue
        tile_shift_rows.append(
            _compare_values(
                reference_df[feature].to_numpy(),
                target_df[feature].to_numpy(),
                feature=feature,
                stratum="tile",
                feature_type="tile_feature",
            )
        )
    tile_shift = pd.DataFrame(tile_shift_rows)
    if not tile_shift.empty and "normalized_wasserstein" in tile_shift.columns:
        tile_shift = tile_shift.sort_values("normalized_wasserstein", ascending=False)
    tile_shift.to_csv(output_dir / "tile_feature_shift.csv", index=False)

    feature_sets = {
        "deployable_sensor_terrain": deployable_sensor_features,
        "label_conditioned_input": label_conditioned_input_features,
        "sensor_terrain": sensor_features,
        "label_geometry": geometry_features,
        "combined": sensor_features + geometry_features,
    }
    classifier_metrics, classifier_coefficients = _domain_classifier(
        tile_features,
        feature_sets,
        seed=seed,
        reference_ratio=domain_classifier_reference_ratio,
    )
    classifier_metrics.to_csv(output_dir / "domain_classifier_metrics.csv", index=False)
    classifier_coefficients.to_csv(output_dir / "domain_classifier_coefficients.csv", index=False)

    event_similarity, target_zscores = _event_similarity(tile_features, feature_sets)
    event_similarity.to_csv(output_dir / "training_event_similarity.csv", index=False)
    target_zscores.to_csv(output_dir / "target_feature_zscores.csv", index=False)

    classifier_map = {
        row["feature_set"]: row
        for row in classifier_metrics.to_dict(orient="records")
    }
    deployable_sensor_auc = float(
        classifier_map.get("deployable_sensor_terrain", {}).get("roc_auc", math.nan)
    )
    sensor_auc = float(classifier_map.get("sensor_terrain", {}).get("roc_auc", math.nan))
    label_conditioned_input_auc = float(
        classifier_map.get("label_conditioned_input", {}).get("roc_auc", math.nan)
    )
    geometry_auc = float(classifier_map.get("label_geometry", {}).get("roc_auc", math.nan))
    combined_auc = float(classifier_map.get("combined", {}).get("roc_auc", math.nan))
    if math.isfinite(deployable_sensor_auc) and deployable_sensor_auc >= 0.80:
        sensor_diagnosis = "strong_sensor_terrain_domain_shift"
    elif math.isfinite(deployable_sensor_auc) and deployable_sensor_auc >= 0.70:
        sensor_diagnosis = "moderate_sensor_terrain_domain_shift"
    else:
        sensor_diagnosis = "weak_sensor_terrain_domain_shift"
    geometry_contribution = (
        math.isfinite(geometry_auc)
        and geometry_auc >= 0.75
    ) or (
        math.isfinite(sensor_auc)
        and math.isfinite(combined_auc)
        and combined_auc - sensor_auc >= 0.05
    )
    diagnosis = sensor_diagnosis + (
        "_with_label_geometry_shift" if geometry_contribution else "_without_large_additional_label_geometry_shift"
    )

    target_empty_rate = float(target_df["empty_tile"].mean())
    reference_empty_rate = float(reference_df["empty_tile"].mean())
    target_fg_median = float(target_df["fg_ratio"].median())
    reference_fg_median = float(reference_df["fg_ratio"].median())
    strongest_pixel = []
    if not pixel_shift.empty and "normalized_wasserstein" in pixel_shift.columns:
        strongest_pixel = (
            pixel_shift.sort_values("normalized_wasserstein", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )
    closest_events = event_similarity.head(10).to_dict(orient="records") if not event_similarity.empty else []
    top_unusual_features = target_zscores.head(15).to_dict(orient="records") if not target_zscores.empty else []
    summary = {
        "schema_version": 1,
        "processed_data_dir": str(processed_data_dir),
        "reference_split": reference_split,
        "target_split": target_split,
        "target_events": target_events,
        "reference_events_filter": list(reference_events or []),
        "exclude_reference_events": list(exclude_reference_events or []),
        "input_modalities": modalities,
        "normalization_stats_path": str(normalization_stats_path) if normalization_stats_path else None,
        "normalization_mode": normalization_mode,
        "reference_tiles": int(len(reference_df)),
        "target_tiles": int(len(target_df)),
        "reference_event_count": int(reference_df["event_id"].nunique()),
        "target_event_count": int(target_df["event_id"].nunique()),
        "reference_empty_tile_rate": reference_empty_rate,
        "target_empty_tile_rate": target_empty_rate,
        "reference_median_fg_ratio": reference_fg_median,
        "target_median_fg_ratio": target_fg_median,
        "domain_classifier": classifier_metrics.to_dict(orient="records"),
        "deployable_sensor_terrain_roc_auc": deployable_sensor_auc,
        "label_conditioned_input_roc_auc": label_conditioned_input_auc,
        "sensor_terrain_roc_auc": sensor_auc,
        "label_geometry_roc_auc": geometry_auc,
        "combined_roc_auc": combined_auc,
        "diagnosis": diagnosis,
        "strongest_pixel_distribution_shifts": strongest_pixel,
        "closest_training_events": closest_events,
        "most_unusual_target_features": top_unusual_features,
        "outputs": {
            "tile_features": str(output_dir / "tile_features.csv"),
            "pixel_distribution_shift": str(output_dir / "pixel_distribution_shift.csv"),
            "tile_feature_shift": str(output_dir / "tile_feature_shift.csv"),
            "domain_classifier_metrics": str(output_dir / "domain_classifier_metrics.csv"),
            "domain_classifier_coefficients": str(output_dir / "domain_classifier_coefficients.csv"),
            "training_event_similarity": str(output_dir / "training_event_similarity.csv"),
            "target_feature_zscores": str(output_dir / "target_feature_zscores.csv"),
        },
        "limitations": [
            "This is a distribution diagnostic, not a model-performance estimate.",
            "The domain classifier uses tile-level cross-validation and can reflect spatial correlation within an event.",
            "Label-conditioned input and label-geometry features use the ground-truth mask and are not available at deployment time.",
            "Strong distribution shift does not by itself establish annotation error or causal model failure.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if write_plots:
        _write_plots(output_dir, pixel_shift, target_zscores)

    LOG.info(
        "Domain classifier ROC-AUC | deployable sensor/terrain=%s | label-conditioned input=%s | label geometry=%s | combined=%s",
        f"{deployable_sensor_auc:.4f}" if math.isfinite(deployable_sensor_auc) else "n/a",
        f"{label_conditioned_input_auc:.4f}" if math.isfinite(label_conditioned_input_auc) else "n/a",
        f"{geometry_auc:.4f}" if math.isfinite(geometry_auc) else "n/a",
        f"{combined_auc:.4f}" if math.isfinite(combined_auc) else "n/a",
    )
    LOG.info(
        "Tile composition | reference empty=%.3f median_fg=%.5f | target empty=%.3f median_fg=%.5f",
        reference_empty_rate,
        reference_fg_median,
        target_empty_rate,
        target_fg_median,
    )
    if closest_events:
        LOG.info(
            "Closest training events by combined standardized distance: %s",
            ", ".join(str(row["event_id"]) for row in closest_events[:5]),
        )
    LOG.info("Domain-shift diagnosis: %s", diagnosis)
    LOG.info("Domain-shift audit outputs written to: %s", output_dir)
    return summary
