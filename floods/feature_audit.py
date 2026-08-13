from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from floods.derived_features import read_processed_modalities
from floods.modalities import canonicalize_modalities
from floods.utils.common import get_logger
from floods.utils.console import progress_iter
from floods.utils.gis import imread

LOG = get_logger(__name__)

DEFAULT_BASE_MODALITIES = ("vv", "vh", "dem")
DEFAULT_EXTENDED_MODALITIES = (
    "vv",
    "vh",
    "dem",
    "vv_vh_log_ratio",
    "dem_slope",
    "dem_tpi",
)


def _event_id(stem: str) -> str:
    match = re.search(r"(EMSR\d+)", str(stem))
    return match.group(1) if match else "unknown"


def _auc_from_ranks(values: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    values = np.asarray(values, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(values, method="average")
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)


def _cohens_d(flood: np.ndarray, background: np.ndarray) -> float:
    flood = np.asarray(flood, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if flood.size < 2 or background.size < 2:
        return float("nan")
    var_a = float(np.var(flood, ddof=1))
    var_b = float(np.var(background, ddof=1))
    pooled_num = (flood.size - 1) * var_a + (background.size - 1) * var_b
    pooled_den = flood.size + background.size - 2
    pooled = np.sqrt(max(pooled_num / max(pooled_den, 1), 0.0))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(flood) - np.mean(background)) / pooled)


def _feature_rows(
    features: np.ndarray,
    labels: np.ndarray,
    modalities: Sequence[str],
    *,
    event_id: str = "all",
) -> list[dict]:
    rows: list[dict] = []
    labels = np.asarray(labels, dtype=np.uint8)
    for idx, modality in enumerate(modalities):
        values = np.asarray(features[:, idx], dtype=np.float64)
        finite = np.isfinite(values)
        local_labels = labels[finite]
        values = values[finite]
        flood = values[local_labels == 1]
        background = values[local_labels == 0]
        if flood.size == 0 or background.size == 0:
            continue
        auc = _auc_from_ranks(values, local_labels)
        ks = ks_2samp(flood, background, alternative="two-sided", method="auto")
        rows.append(
            {
                "event_id": event_id,
                "feature": modality,
                "n_flood": int(flood.size),
                "n_background": int(background.size),
                "flood_mean": float(np.mean(flood)),
                "background_mean": float(np.mean(background)),
                "flood_median": float(np.median(flood)),
                "background_median": float(np.median(background)),
                "median_difference": float(np.median(flood) - np.median(background)),
                "cohens_d": _cohens_d(flood, background),
                "roc_auc": float(auc),
                "auc_separation": float(max(auc, 1.0 - auc)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return rows


def _sample_split(
    processed_data_dir: Path,
    split: str,
    modalities: Sequence[str],
    *,
    max_pixels_per_class_per_tile: int,
    max_total_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask_paths = sorted((Path(processed_data_dir) / split / "mask").glob("*.tif"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found under {Path(processed_data_dir) / split / 'mask'}")
    rng = np.random.default_rng(seed)
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    event_parts: list[np.ndarray] = []
    stem_parts: list[np.ndarray] = []
    max_per_class = max(int(max_pixels_per_class_per_tile), 1)

    for mask_path in progress_iter(mask_paths, desc=f"Feature audit {split}", unit="tile", colour="magenta"):
        stem = mask_path.stem
        mask = imread(mask_path, channels_first=True).squeeze().astype(np.uint8, copy=False)
        stack = read_processed_modalities(processed_data_dir, split, stem, modalities)
        if stack.shape[:2] != mask.shape:
            raise ValueError(f"Feature/mask shape mismatch for {stem}: {stack.shape[:2]} != {mask.shape}")
        flat_x = stack.reshape(-1, stack.shape[-1])
        flat_y = mask.reshape(-1)
        finite = np.all(np.isfinite(flat_x), axis=1)
        valid = (flat_y != 255) & finite & np.isin(flat_y, [0, 1])
        candidate_indices: list[np.ndarray] = []
        for label in (0, 1):
            indices = np.flatnonzero(valid & (flat_y == label))
            if indices.size > max_per_class:
                indices = rng.choice(indices, size=max_per_class, replace=False)
            if indices.size:
                candidate_indices.append(np.asarray(indices, dtype=np.int64))
        if not candidate_indices:
            continue
        selected = np.concatenate(candidate_indices)
        rng.shuffle(selected)
        feature_parts.append(flat_x[selected].astype(np.float32, copy=False))
        label_parts.append(flat_y[selected].astype(np.uint8, copy=False))
        event = _event_id(stem)
        event_parts.append(np.full(selected.size, event, dtype=object))
        stem_parts.append(np.full(selected.size, stem, dtype=object))

    if not feature_parts:
        raise RuntimeError(f"No valid pixels were available for feature audit split={split}")
    features = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    events = np.concatenate(event_parts, axis=0)
    stems = np.concatenate(stem_parts, axis=0)

    max_total = int(max_total_pixels or 0)
    if max_total > 0 and features.shape[0] > max_total:
        # Preserve both classes and all events as far as practical by sampling within event/class groups.
        frame = pd.DataFrame({"index": np.arange(features.shape[0]), "event": events, "label": labels})
        group_count = max(frame.groupby(["event", "label"]).ngroups, 1)
        target_per_group = max(max_total // group_count, 1)
        selected_parts: list[np.ndarray] = []
        for _, group in frame.groupby(["event", "label"], sort=False):
            indices = group["index"].to_numpy(dtype=np.int64)
            if indices.size > target_per_group:
                indices = rng.choice(indices, size=target_per_group, replace=False)
            selected_parts.append(indices)
        selected = np.concatenate(selected_parts)
        if selected.size > max_total:
            selected = rng.choice(selected, size=max_total, replace=False)
        rng.shuffle(selected)
        features = features[selected]
        labels = labels[selected]
        events = events[selected]
        stems = stems[selected]

    LOG.info(
        "Feature audit sample: split=%s | pixels=%d | flood=%d | background=%d | events=%d",
        split,
        features.shape[0],
        int((labels == 1).sum()),
        int((labels == 0).sum()),
        len(set(events.tolist())),
    )
    return features, labels, events, stems


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return model


def _score_model(name: str, model: Pipeline, features: np.ndarray, labels: np.ndarray) -> dict:
    probability = model.predict_proba(features)[:, 1]
    return {
        "model": name,
        "pixels": int(features.shape[0]),
        "flood_pixels": int((labels == 1).sum()),
        "background_pixels": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
    }


def audit_feature_separability(
    processed_data_dir: Path,
    output_dir: Path,
    *,
    fit_split: str = "train",
    eval_split: str = "val",
    base_modalities: Sequence[str] = DEFAULT_BASE_MODALITIES,
    extended_modalities: Sequence[str] = DEFAULT_EXTENDED_MODALITIES,
    max_pixels_per_class_per_tile: int = 128,
    max_total_pixels_per_split: int = 250_000,
    seed: int = 42,
) -> dict:
    """Compare original and derived channels before committing to a deep training run."""
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_modalities = canonicalize_modalities(base_modalities)
    extended_modalities = canonicalize_modalities(extended_modalities)
    missing_base = [m for m in base_modalities if m not in extended_modalities]
    if missing_base:
        raise ValueError(f"Extended modalities must contain every base modality; missing {missing_base}")

    fit_x, fit_y, fit_events, _ = _sample_split(
        processed_data_dir,
        fit_split,
        extended_modalities,
        max_pixels_per_class_per_tile=max_pixels_per_class_per_tile,
        max_total_pixels=max_total_pixels_per_split,
        seed=seed,
    )
    eval_x, eval_y, eval_events, _ = _sample_split(
        processed_data_dir,
        eval_split,
        extended_modalities,
        max_pixels_per_class_per_tile=max_pixels_per_class_per_tile,
        max_total_pixels=max_total_pixels_per_split,
        seed=seed + 1,
    )
    model_modalities: dict[str, list[str]] = {"base": list(base_modalities)}
    ratio_name = "vv_vh_log_ratio"
    terrain_names = [name for name in ("dem_slope", "dem_tpi") if name in extended_modalities]
    if ratio_name in extended_modalities:
        model_modalities["base_plus_ratio"] = list(base_modalities) + [ratio_name]
    if terrain_names:
        model_modalities["base_plus_terrain"] = list(base_modalities) + terrain_names
    model_modalities["extended"] = list(extended_modalities)

    model_indices = {
        model_name: [extended_modalities.index(name) for name in names]
        for model_name, names in model_modalities.items()
    }
    fitted_models = {
        model_name: _fit_logistic(fit_x[:, indices], fit_y)
        for model_name, indices in model_indices.items()
    }

    global_rows: list[dict] = []
    for model_name, model in fitted_models.items():
        row = _score_model(model_name, model, eval_x[:, model_indices[model_name]], eval_y)
        row["modalities"] = " ".join(model_modalities[model_name])
        global_rows.append(row)
    comparison = pd.DataFrame(global_rows)
    base_row_global = comparison.loc[comparison["model"] == "base"].iloc[0]
    base_auc = float(base_row_global["roc_auc"])
    base_ap = float(base_row_global["average_precision"])
    comparison["roc_auc_delta_vs_base"] = comparison["roc_auc"] - base_auc
    comparison["average_precision_delta_vs_base"] = comparison["average_precision"] - base_ap
    comparison.to_csv(output_dir / "logistic_comparison.csv", index=False)

    extended_row_global = comparison.loc[comparison["model"] == "extended"].iloc[0]
    extended_auc = float(extended_row_global["roc_auc"])
    extended_ap = float(extended_row_global["average_precision"])

    event_rows: list[dict] = []
    for event in sorted(set(eval_events.tolist())):
        event_mask = eval_events == event
        if len(np.unique(eval_y[event_mask])) < 2:
            continue
        local_rows: list[dict] = []
        for model_name, model in fitted_models.items():
            row = _score_model(
                model_name,
                model,
                eval_x[event_mask][:, model_indices[model_name]],
                eval_y[event_mask],
            )
            row["event_id"] = event
            row["modalities"] = " ".join(model_modalities[model_name])
            local_rows.append(row)
        event_base = next(row for row in local_rows if row["model"] == "base")
        for row in local_rows:
            row["roc_auc_delta_vs_base"] = row["roc_auc"] - event_base["roc_auc"]
            row["average_precision_delta_vs_base"] = row["average_precision"] - event_base["average_precision"]
        event_rows.extend(local_rows)
    event_comparison = pd.DataFrame(event_rows)
    event_comparison.to_csv(output_dir / "event_logistic_comparison.csv", index=False)

    univariate_global = pd.DataFrame(
        _feature_rows(eval_x, eval_y, extended_modalities, event_id="all")
    ).sort_values(["auc_separation", "ks_statistic"], ascending=False)
    univariate_global.to_csv(output_dir / "univariate_global.csv", index=False)

    univariate_event_rows: list[dict] = []
    for event in sorted(set(eval_events.tolist())):
        event_mask = eval_events == event
        if len(np.unique(eval_y[event_mask])) < 2:
            continue
        univariate_event_rows.extend(
            _feature_rows(eval_x[event_mask], eval_y[event_mask], extended_modalities, event_id=event)
        )
    univariate_by_event = pd.DataFrame(univariate_event_rows)
    if not univariate_by_event.empty:
        univariate_by_event = univariate_by_event.sort_values(
            ["event_id", "auc_separation", "ks_statistic"], ascending=[True, False, False]
        )
    univariate_by_event.to_csv(output_dir / "univariate_by_event.csv", index=False)

    emsr342_rows = event_comparison[
        (event_comparison.get("event_id") == "EMSR342") & (event_comparison.get("model") == "extended")
    ] if not event_comparison.empty else pd.DataFrame()
    emsr342_auc_delta = float(emsr342_rows["roc_auc_delta_vs_base"].iloc[0]) if not emsr342_rows.empty else None
    auc_delta = extended_auc - base_auc
    ap_delta = extended_ap - base_ap
    ratio_rows = comparison[comparison["model"] == "base_plus_ratio"]
    terrain_rows = comparison[comparison["model"] == "base_plus_terrain"]
    ratio_auc_delta = float(ratio_rows["roc_auc_delta_vs_base"].iloc[0]) if not ratio_rows.empty else None
    terrain_auc_delta = float(terrain_rows["roc_auc_delta_vs_base"].iloc[0]) if not terrain_rows.empty else None
    if auc_delta >= 0.01 and (emsr342_auc_delta is None or emsr342_auc_delta >= 0.005):
        recommendation = "proceed_to_controlled_training"
    elif auc_delta >= 0.003 or ap_delta >= 0.01:
        recommendation = "weak_gain_run_only_one_controlled_training_experiment"
    else:
        recommendation = "do_not_train_no_clear_added_separability"

    summary = {
        "schema_version": 1,
        "processed_data_dir": str(processed_data_dir),
        "fit_split": fit_split,
        "eval_split": eval_split,
        "base_modalities": base_modalities,
        "extended_modalities": extended_modalities,
        "model_modalities": model_modalities,
        "fit_pixels": int(fit_x.shape[0]),
        "eval_pixels": int(eval_x.shape[0]),
        "base_roc_auc": base_auc,
        "extended_roc_auc": extended_auc,
        "roc_auc_delta": auc_delta,
        "base_average_precision": base_ap,
        "extended_average_precision": extended_ap,
        "average_precision_delta": ap_delta,
        "ratio_only_roc_auc_delta": ratio_auc_delta,
        "terrain_only_roc_auc_delta": terrain_auc_delta,
        "emsr342_roc_auc_delta": emsr342_auc_delta,
        "recommendation": recommendation,
        "limitations": [
            "This is a sampled pixel-level diagnostic, not a segmentation benchmark.",
            "Derived channels can help spatial boundary learning even when univariate separation is modest.",
            "A positive audit result does not guarantee validation F1 improvement.",
        ],
    }
    (output_dir / "feature_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOG.info(
        "Feature audit result: base ROC-AUC=%.4f | extended ROC-AUC=%.4f | delta=%+.4f | recommendation=%s",
        base_auc,
        extended_auc,
        auc_delta,
        recommendation,
    )
    if emsr342_auc_delta is not None:
        LOG.info("EMSR342 extended-vs-base ROC-AUC delta: %+.4f", emsr342_auc_delta)
    LOG.info("Feature audit outputs written to: %s", output_dir)
    return summary
