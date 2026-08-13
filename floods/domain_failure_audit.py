from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from floods.utils.common import get_logger

LOG = get_logger(__name__)

_DEPLOYABLE_SUFFIXES = {
    "mean",
    "std",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "clip_low_rate",
    "clip_high_rate",
    "clip_total_rate",
}
_GEOMETRY_FEATURES = {
    "valid_ratio",
    "fg_ratio",
    "empty_tile",
    "component_count",
    "largest_component_pixels",
    "largest_component_fraction",
    "median_component_pixels",
    "boundary_area_ratio",
    "boundary_pixels",
    "fg_pixels",
}


def _stem(value: object) -> str:
    return Path(str(value)).stem


def _feature_group(feature: str) -> str:
    if feature in _GEOMETRY_FEATURES:
        return "label_geometry"
    if "_flood_" in feature or "_background_" in feature:
        return "label_conditioned_input"
    if feature == "all_modalities_finite_ratio":
        return "deployable_sensor_terrain"
    match = re.match(r"^(.+?)_(.+)$", feature)
    if match and match.group(2) in _DEPLOYABLE_SUFFIXES:
        return "deployable_sensor_terrain"
    return "other"


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "domain",
        "split",
        "event_id",
        "file",
        "stem",
        "index",
        "tile_row_offset",
        "tile_col_offset",
        "threshold",
        "min_component_area",
        "valid_pixels",
        "pred_pixels",
        "pred_ratio",
        "tp_pixels",
        "tn_pixels",
        "fp_pixels",
        "fn_pixels",
        "precision",
        "recall",
        "f1",
        "iou",
        "mcc",
        "is_empty",
        "pred_any",
    }
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _performance_correlations(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in features:
        x = pd.to_numeric(frame[feature], errors="coerce")
        for metric in ("recall", "iou", "f1"):
            y = pd.to_numeric(frame[metric], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 20 or int(x[valid].nunique()) < 3:
                continue
            rows.append(
                {
                    "feature_group": _feature_group(feature),
                    "feature": feature,
                    "metric": metric,
                    "samples": int(valid.sum()),
                    "spearman_rho": float(x[valid].corr(y[valid], method="spearman")),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["absolute_spearman_rho"] = result["spearman_rho"].abs()
        result = result.sort_values(
            ["metric", "absolute_spearman_rho"], ascending=[True, False]
        )
    return result


def _failure_group_shift(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in features:
        failed = pd.to_numeric(frame.loc[frame["failure"], feature], errors="coerce").dropna()
        other = pd.to_numeric(frame.loc[~frame["failure"], feature], errors="coerce").dropna()
        if len(failed) < 5 or len(other) < 5:
            continue
        pooled = math.sqrt((float(failed.var(ddof=1)) + float(other.var(ddof=1))) / 2.0)
        standardized_median_difference = (
            float((failed.median() - other.median()) / pooled) if pooled > 1e-12 else 0.0
        )
        rows.append(
            {
                "feature_group": _feature_group(feature),
                "feature": feature,
                "failure_samples": int(len(failed)),
                "other_samples": int(len(other)),
                "failure_mean": float(failed.mean()),
                "other_mean": float(other.mean()),
                "failure_median": float(failed.median()),
                "other_median": float(other.median()),
                "standardized_median_difference": standardized_median_difference,
                "absolute_standardized_median_difference": abs(standardized_median_difference),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            "absolute_standardized_median_difference", ascending=False
        )
    return result


def _failure_classifier(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = [feature for feature in features if feature in frame and not frame[feature].isna().all()]
    y = frame["failure"].astype(np.uint8).to_numpy()
    if len(np.unique(y)) < 2 or not usable:
        return pd.DataFrame(), pd.DataFrame()
    min_class = int(np.bincount(y).min())
    folds = min(5, min_class)
    if folds < 2:
        return pd.DataFrame(), pd.DataFrame()
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
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probability = cross_val_predict(
        pipeline,
        frame[usable],
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    metrics = pd.DataFrame(
        [
            {
                "feature_set": "deployable_sensor_terrain",
                "features": len(usable),
                "samples": int(len(frame)),
                "failure_samples": int(y.sum()),
                "other_samples": int((1 - y).sum()),
                "cv_folds": folds,
                "roc_auc": float(roc_auc_score(y, probability)),
                "average_precision": float(average_precision_score(y, probability)),
            }
        ]
    )
    pipeline.fit(frame[usable], y)
    coefficients = pipeline.named_steps["model"].coef_[0]
    coefficient_frame = pd.DataFrame(
        {
            "feature": usable,
            "standardized_coefficient": coefficients.astype(float),
        }
    )
    coefficient_frame["absolute_coefficient"] = coefficient_frame[
        "standardized_coefficient"
    ].abs()
    coefficient_frame["direction"] = np.where(
        coefficient_frame["standardized_coefficient"] > 0,
        "higher_in_failures",
        "lower_in_failures",
    )
    coefficient_frame = coefficient_frame.sort_values(
        "absolute_coefficient", ascending=False
    )
    return metrics, coefficient_frame


def _nearest_reference_analogues(
    reference: pd.DataFrame,
    failures: pd.DataFrame,
    features: Sequence[str],
    *,
    neighbours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    usable = [feature for feature in features if feature in reference and not reference[feature].isna().all()]
    if not usable or reference.empty or failures.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    reference_values = scaler.fit_transform(imputer.fit_transform(reference[usable]))
    failure_values = scaler.transform(imputer.transform(failures[usable]))
    k = min(max(1, int(neighbours)), len(reference))
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(reference_values)
    distances, indices = model.kneighbors(failure_values)

    # Reference self-neighbour distances provide a support threshold in the same space.
    if len(reference) >= 2:
        self_model = NearestNeighbors(n_neighbors=2, metric="euclidean")
        self_model.fit(reference_values)
        self_distances, _ = self_model.kneighbors(reference_values)
        reference_nn = self_distances[:, 1]
        support_p95 = float(np.quantile(reference_nn, 0.95))
        support_median = float(np.median(reference_nn))
    else:
        support_p95 = math.nan
        support_median = math.nan

    rows: list[dict] = []
    for target_position, (_, target_row) in enumerate(failures.reset_index(drop=True).iterrows()):
        for rank in range(k):
            reference_row = reference.iloc[int(indices[target_position, rank])]
            rows.append(
                {
                    "target_file": target_row["file_err"],
                    "target_event": target_row["event_id_err"],
                    "target_recall": float(target_row["recall"]),
                    "target_iou": float(target_row["iou"]),
                    "neighbour_rank": rank + 1,
                    "distance": float(distances[target_position, rank]),
                    "reference_file": reference_row["file"],
                    "reference_event": reference_row["event_id"],
                    "reference_fg_ratio": _safe_float(reference_row.get("fg_ratio")),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs, pd.DataFrame(), {}
    analogue_summary = (
        pairs.groupby(["reference_event", "reference_file"], as_index=False)
        .agg(
            matched_targets=("target_file", "nunique"),
            matches=("target_file", "size"),
            minimum_distance=("distance", "min"),
            mean_distance=("distance", "mean"),
            reference_fg_ratio=("reference_fg_ratio", "first"),
        )
        .sort_values(["matched_targets", "minimum_distance"], ascending=[False, True])
    )
    nearest = pairs[pairs["neighbour_rank"] == 1]
    event_summary = (
        pairs.groupby("reference_event", as_index=False)
        .agg(
            unique_reference_tiles=("reference_file", "nunique"),
            matched_targets=("target_file", "nunique"),
            matches=("target_file", "size"),
            minimum_distance=("distance", "min"),
            mean_distance=("distance", "mean"),
        )
        .sort_values(["matched_targets", "minimum_distance"], ascending=[False, True])
    )
    support_rate = (
        float(np.mean(nearest["distance"] <= support_p95))
        if math.isfinite(support_p95)
        else math.nan
    )
    diagnostics = {
        "features": len(usable),
        "neighbours_per_failure": k,
        "reference_self_nn_median": support_median,
        "reference_self_nn_p95": support_p95,
        "failure_nearest_distance_median": float(nearest["distance"].median()),
        "failure_nearest_distance_p95": float(nearest["distance"].quantile(0.95)),
        "failure_within_reference_support_rate": support_rate,
        "closest_reference_events": event_summary.head(10).to_dict(orient="records"),
    }
    return pairs, analogue_summary, diagnostics


def audit_domain_failure_link(
    tile_features_csv: Path,
    tile_error_metrics_csv: Path,
    output_dir: Path,
    *,
    target_events: Iterable[str] | None = None,
    max_recall: float = 0.25,
    neighbours: int = 5,
    seed: int = 42,
    write_plots: bool = True,
) -> dict:
    """Link target-event failures to domain features and training-domain coverage."""
    tile_features_csv = Path(tile_features_csv)
    tile_error_metrics_csv = Path(tile_error_metrics_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(tile_features_csv)
    errors = pd.read_csv(tile_error_metrics_csv)
    required_feature_columns = {"domain", "event_id", "file"}
    required_error_columns = {"event_id", "file", "recall", "iou", "f1", "fg_pixels"}
    if not required_feature_columns.issubset(features.columns):
        missing = sorted(required_feature_columns - set(features.columns))
        raise ValueError(f"tile_features.csv is missing required columns: {missing}")
    if not required_error_columns.issubset(errors.columns):
        missing = sorted(required_error_columns - set(errors.columns))
        raise ValueError(f"tile_error_metrics.csv is missing required columns: {missing}")
    features = features.copy()
    errors = errors.copy()
    features["stem"] = features["file"].map(_stem)
    errors["stem"] = errors["file"].map(_stem)
    events = {str(event).upper() for event in (target_events or [])}
    if events:
        errors = errors[errors["event_id"].astype(str).str.upper().isin(events)]
    target_features = features[features["domain"] == "target"].copy()
    if events:
        target_features = target_features[
            target_features["event_id"].astype(str).str.upper().isin(events)
        ]
    joined = errors.merge(
        target_features,
        on="stem",
        suffixes=("_err", "_feat"),
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("No target tiles matched between the feature and error CSV files")
    fg_pixels_column = "fg_pixels_err" if "fg_pixels_err" in joined.columns else "fg_pixels"
    joined = joined[pd.to_numeric(joined[fg_pixels_column], errors="coerce") > 0].copy()
    if joined.empty:
        raise ValueError("No non-empty target tiles remain after joining the audit files")
    joined["failure"] = pd.to_numeric(joined["recall"], errors="coerce") < float(max_recall)

    # Map feature-file columns to their post-merge names.
    feature_name_map: dict[str, str] = {}
    for column in _numeric_feature_columns(features):
        merged_name = column if column not in errors.columns else f"{column}_feat"
        if merged_name in joined.columns:
            feature_name_map[column] = merged_name
    analysis_frame = joined.rename(
        columns={merged: original for original, merged in feature_name_map.items()}
    )
    all_features = list(feature_name_map)
    deployable = [feature for feature in all_features if _feature_group(feature) == "deployable_sensor_terrain"]

    correlations = _performance_correlations(analysis_frame, all_features)
    failure_shift = _failure_group_shift(analysis_frame, all_features)
    classifier_metrics, classifier_coefficients = _failure_classifier(
        analysis_frame,
        deployable,
        seed=seed,
    )

    reference = features[features["domain"] == "reference"].copy()
    failures = analysis_frame[analysis_frame["failure"]].copy()
    analogue_pairs, analogue_summary, coverage = _nearest_reference_analogues(
        reference,
        failures,
        deployable,
        neighbours=neighbours,
    )

    analysis_frame.to_csv(output_dir / "joined_target_tile_metrics.csv", index=False)
    correlations.to_csv(output_dir / "feature_performance_correlations.csv", index=False)
    failure_shift.to_csv(output_dir / "failure_group_feature_shift.csv", index=False)
    classifier_metrics.to_csv(output_dir / "failure_classifier_metrics.csv", index=False)
    classifier_coefficients.to_csv(output_dir / "failure_classifier_coefficients.csv", index=False)
    analogue_pairs.to_csv(output_dir / "failure_training_analogue_pairs.csv", index=False)
    analogue_summary.to_csv(output_dir / "training_analogue_summary.csv", index=False)

    top_recall = (
        correlations[
            (correlations["metric"] == "recall")
            & (correlations["feature_group"] == "deployable_sensor_terrain")
        ]
        .head(15)
        .to_dict(orient="records")
        if not correlations.empty
        else []
    )
    top_failure_shift = (
        failure_shift[
            failure_shift["feature_group"] == "deployable_sensor_terrain"
        ]
        .head(15)
        .to_dict(orient="records")
        if not failure_shift.empty
        else []
    )
    classifier_auc = (
        float(classifier_metrics.iloc[0]["roc_auc"])
        if not classifier_metrics.empty
        else math.nan
    )
    support_rate = float(coverage.get("failure_within_reference_support_rate", math.nan))
    if math.isfinite(classifier_auc) and classifier_auc >= 0.80:
        failure_predictability = "strong"
    elif math.isfinite(classifier_auc) and classifier_auc >= 0.70:
        failure_predictability = "moderate"
    else:
        failure_predictability = "weak"
    if math.isfinite(support_rate) and support_rate >= 0.80:
        coverage_diagnosis = "training_contains_close_analogues"
    elif math.isfinite(support_rate) and support_rate >= 0.50:
        coverage_diagnosis = "training_coverage_is_partial"
    else:
        coverage_diagnosis = "training_lacks_close_analogues"
    diagnosis = f"{failure_predictability}_feature_failure_link__{coverage_diagnosis}"

    summary = {
        "schema_version": 1,
        "tile_features_csv": str(tile_features_csv),
        "tile_error_metrics_csv": str(tile_error_metrics_csv),
        "target_events": sorted(events),
        "nonempty_target_tiles": int(len(analysis_frame)),
        "failure_recall_threshold": float(max_recall),
        "failure_tiles": int(analysis_frame["failure"].sum()),
        "other_tiles": int((~analysis_frame["failure"]).sum()),
        "deployable_features": len(deployable),
        "failure_classifier_roc_auc": classifier_auc,
        "failure_classifier_average_precision": (
            float(classifier_metrics.iloc[0]["average_precision"])
            if not classifier_metrics.empty
            else math.nan
        ),
        "coverage": coverage,
        "diagnosis": diagnosis,
        "top_deployable_recall_correlations": top_recall,
        "top_deployable_failure_shifts": top_failure_shift,
        "outputs": {
            "joined_target_tile_metrics": str(output_dir / "joined_target_tile_metrics.csv"),
            "feature_performance_correlations": str(output_dir / "feature_performance_correlations.csv"),
            "failure_group_feature_shift": str(output_dir / "failure_group_feature_shift.csv"),
            "failure_classifier_metrics": str(output_dir / "failure_classifier_metrics.csv"),
            "failure_classifier_coefficients": str(output_dir / "failure_classifier_coefficients.csv"),
            "failure_training_analogue_pairs": str(output_dir / "failure_training_analogue_pairs.csv"),
            "training_analogue_summary": str(output_dir / "training_analogue_summary.csv"),
        },
        "limitations": [
            "Associations do not establish causality.",
            "The failure definition is tied to the supplied model audit and operating threshold.",
            "Nearest-neighbour coverage is measured in standardized tile-summary space, not raw-image space.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if write_plots:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            if top_recall:
                view = pd.DataFrame(top_recall).sort_values("spearman_rho")
                plt.figure(figsize=(10, 6))
                plt.barh(view["feature"], view["spearman_rho"])
                plt.xlabel("Spearman correlation with tile recall")
                plt.title("Target-event deployable features linked to recall")
                plt.tight_layout()
                plt.savefig(output_dir / "deployable_feature_recall_correlations.png", dpi=160)
                plt.close()
            if not analogue_pairs.empty:
                nearest = analogue_pairs[analogue_pairs["neighbour_rank"] == 1]
                plt.figure(figsize=(8, 5))
                plt.hist(nearest["distance"], bins=min(20, max(5, len(nearest) // 4)))
                threshold = coverage.get("reference_self_nn_p95")
                if threshold is not None and math.isfinite(float(threshold)):
                    plt.axvline(float(threshold), linestyle="--", label="training self-NN p95")
                    plt.legend()
                plt.xlabel("Nearest training-tile distance")
                plt.ylabel("Failing target tiles")
                plt.title("Training-domain coverage of target failures")
                plt.tight_layout()
                plt.savefig(output_dir / "failure_training_coverage.png", dpi=160)
                plt.close()
        except Exception as exc:  # pragma: no cover - plotting is best effort
            LOG.warning("Domain-failure plots were not written: %s", exc)

    LOG.info(
        "Domain-failure link | target nonempty=%d | failures=%d | deployable classifier ROC-AUC=%.4f",
        len(analysis_frame),
        int(analysis_frame["failure"].sum()),
        classifier_auc,
    )
    LOG.info(
        "Training analogue coverage | within support=%.4f | nearest median=%.4f | reference self-NN p95=%.4f",
        support_rate,
        float(coverage.get("failure_nearest_distance_median", math.nan)),
        float(coverage.get("reference_self_nn_p95", math.nan)),
    )
    LOG.info("Domain-failure diagnosis: %s", diagnosis)
    LOG.info("Domain-failure audit outputs written to: %s", output_dir)
    return summary
