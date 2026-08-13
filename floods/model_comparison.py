from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from floods.ensemble_evaluation import ensemble_evaluate_checkpoints
from floods.evaluation import evaluate_checkpoint
from floods.utils.common import get_logger

LOG = get_logger(__name__)


@dataclass
class SingleModelSpec:
    name: str
    config: Any
    checkpoint: Path


@dataclass
class EnsembleModelSpec:
    name: str
    configs: List[Any]
    checkpoints: List[Path]
    method: str = "mean_logit"


def _metric_value(row: Dict[str, Any], metric: str) -> float:
    for key in (f"best_{metric}", metric):
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _confusion_from_result(result: Dict[str, Any]) -> Dict[str, float]:
    cm = result.get("confusion_matrix") or {}
    return {
        "tp": float(cm.get("tp", result.get("best_tp", 0.0)) or 0.0),
        "tn": float(cm.get("tn", result.get("best_tn", 0.0)) or 0.0),
        "fp": float(cm.get("fp", result.get("best_fp", 0.0)) or 0.0),
        "fn": float(cm.get("fn", result.get("best_fn", 0.0)) or 0.0),
    }


def _flatten_result(name: str, model_type: str, result: Dict[str, Any]) -> Dict[str, Any]:
    cm = _confusion_from_result(result)
    return {
        "name": name,
        "type": model_type,
        "best_threshold": result.get("best_threshold"),
        "best_f1": result.get("best_f1"),
        "best_iou": result.get("best_iou"),
        "best_mcc": result.get("best_mcc"),
        "best_precision": result.get("best_precision"),
        "best_recall": result.get("best_recall"),
        "tp": cm["tp"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
    }


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    keys = [
        "name", "type", "best_threshold", "best_f1", "best_iou", "best_mcc",
        "best_precision", "best_recall", "tp", "tn", "fp", "fn",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _safe_label(name: str, max_chars: int = 48) -> str:
    text = str(name)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _safe_filename(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return value or "model"


def _plot_metric_bars(rows: Sequence[Dict[str, Any]], metric: str, output_path: Path) -> None:
    labels = [_safe_label(row["name"]) for row in rows]
    values = [_metric_value(row, metric) for row in rows]
    height = max(4.0, 0.45 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(9, height))
    positions = list(range(len(rows)))
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model comparison: best {metric.upper()}")
    ax.grid(axis="x", alpha=0.25)
    for pos, value in zip(positions, values):
        if value == value:
            ax.text(value, pos, f" {value:.4f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_threshold_curves(all_results: Sequence[Dict[str, Any]], output_path: Path, metric: str = "f1") -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for result in all_results:
        sweep = result.get("threshold_sweep") or []
        if not sweep:
            continue
        xs = [float(row["threshold"]) for row in sweep if metric in row]
        ys = [float(row[metric]) for row in sweep if metric in row]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=_safe_label(result["name"], 32))
        plotted += 1
    ax.set_xlabel("Threshold")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Threshold sweep: {metric.upper()}")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(name: str, result: Dict[str, Any], output_path: Path) -> None:
    cm = _confusion_from_result(result)
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion matrix: {_safe_label(name, 32)}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Background", "Flood"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Background", "Flood"])
    max_val = float(matrix.max()) if matrix.size else 0.0
    threshold = max_val * 0.5
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value:,.0f}", ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compare_models(single_models: Sequence[SingleModelSpec],
                   ensembles: Sequence[EnsembleModelSpec],
                   output_dir: Path,
                   split: str = "val",
                   thresholds: Optional[Iterable[float]] = None,
                   threshold_metric: str = "f1",
                   metric_mode: Optional[str] = None,
                   include_events: Optional[Iterable[str]] = None,
                   exclude_events: Optional[Iterable[str]] = None,
                   inference_mode: str = "direct",
                   window_size: int = 512,
                   window_overlap: int = 128,
                   window_batch_size: int = 1,
                   plot_metrics: Sequence[str] = ("f1", "iou", "mcc"),
                   write_confusion_matrices: bool = True) -> List[Dict[str, Any]]:
    """Evaluate any number of single checkpoints and ensembles and write tables/plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not single_models and not ensembles:
        raise ValueError("At least one --model or --ensemble must be supplied.")

    all_results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for spec in single_models:
        LOG.info("Comparing single model: %s", spec.name)
        metrics = evaluate_checkpoint(config=spec.config,
                                      checkpoint_path=spec.checkpoint,
                                      split=split,
                                      thresholds=thresholds,
                                      threshold_metric=threshold_metric,
                                      metric_mode=metric_mode,
                                      include_events=include_events,
                                      exclude_events=exclude_events,
                                      inference_mode=inference_mode,
                                      window_size=window_size,
                                      window_overlap=window_overlap,
                                      window_batch_size=window_batch_size)
        record = {"name": spec.name, "type": "single", "metrics": metrics, "threshold_sweep": metrics.get("threshold_sweep", [])}
        all_results.append(record)
        rows.append(_flatten_result(spec.name, "single", metrics))
        if write_confusion_matrices:
            _plot_confusion_matrix(spec.name, metrics, output_dir / f"confusion_matrix_{_safe_filename(spec.name)}.png")

    for spec in ensembles:
        LOG.info("Comparing ensemble: %s (%d members, %s)", spec.name, len(spec.checkpoints), spec.method)
        metrics = ensemble_evaluate_checkpoints(configs=spec.configs,
                                                checkpoint_paths=spec.checkpoints,
                                                split=split,
                                                thresholds=thresholds,
                                                threshold_metric=threshold_metric,
                                                metric_mode=metric_mode,
                                                include_events=include_events,
                                                exclude_events=exclude_events,
                                                ensemble_method=spec.method,
                                                inference_mode=inference_mode,
                                                window_size=window_size,
                                                window_overlap=window_overlap,
                                                window_batch_size=window_batch_size)
        record = {"name": spec.name, "type": "ensemble", "metrics": metrics, "threshold_sweep": metrics.get("threshold_sweep", []), "ensemble_method": spec.method}
        all_results.append(record)
        rows.append(_flatten_result(spec.name, "ensemble", metrics))
        if write_confusion_matrices:
            _plot_confusion_matrix(spec.name, metrics, output_dir / f"confusion_matrix_{_safe_filename(spec.name)}.png")

    rows = sorted(rows, key=lambda row: _metric_value(row, threshold_metric), reverse=True)
    with open(output_dir / "comparison_results.json", "w", encoding="utf-8") as handle:
        json.dump({"split": split,
                   "threshold_metric": threshold_metric,
                   "results": all_results,
                   "summary": rows}, handle, indent=2)
    _write_csv(rows, output_dir / "comparison_summary.csv")

    for metric in plot_metrics:
        metric = str(metric).lower()
        if metric in {"f1", "iou", "mcc", "precision", "recall"}:
            _plot_metric_bars(rows, metric, output_dir / f"comparison_best_{metric}.png")
    _plot_threshold_curves(all_results, output_dir / "threshold_sweep_f1.png", metric="f1")

    LOG.info("Model comparison written to: %s", output_dir)
    return all_results
