from __future__ import annotations

import logging

import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject



LOG = logging.getLogger(__name__)

def _read_band(path: Path, band: int = 1) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        return src.read(band).astype(np.float32), {
            "crs": src.crs,
            "transform": src.transform,
            "shape": (src.height, src.width),
            "nodata": src.nodata,
        }


def _read_to_grid(path: Path, ref: dict[str, Any], *, band: int = 1, categorical: bool = False) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.shape == ref["shape"] and src.crs == ref["crs"] and src.transform == ref["transform"]:
            return src.read(band)
        out = np.zeros(ref["shape"], dtype=src.dtypes[band - 1])
        reproject(
            source=rasterio.band(src, band), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref["transform"], dst_crs=ref["crs"],
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
        return out


def _resolve_path(value: str | None, candidate_dir: Path, patterns: list[str]) -> Path | None:
    if value:
        p = Path(value)
        if p.exists():
            return p
        local = candidate_dir / p.name
        if local.exists():
            return local
    for pattern in patterns:
        found = sorted(candidate_dir.glob(pattern))
        if found:
            return found[0]
    return None


def _pct(values: np.ndarray, q: float) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else None


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p10": _pct(values, 10), "p90": _pct(values, 90)}


def _diagnose(metrics: dict[str, Any], probability: np.ndarray, pred: np.ndarray, target: np.ndarray, threshold: float) -> list[str]:
    valid = target != 255
    tp = valid & (pred == 1) & (target == 1)
    fp = valid & (pred == 1) & (target == 0)
    fn = valid & (pred == 0) & (target == 1)
    notes: list[str] = []
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    if precision < 0.20 and recall >= 0.50:
        notes.append("severe_overprediction")
    elif precision < 0.35:
        notes.append("false_positive_dominant")
    if recall < 0.30:
        notes.append("missed_flood_dominant")
    if fp.any():
        fp_median = float(np.median(probability[fp]))
        if fp_median <= threshold + 0.10:
            notes.append("many_false_positives_near_threshold")
        elif fp_median >= 0.75:
            notes.append("confident_false_positives")
    if fn.any():
        fn_median = float(np.median(probability[fn]))
        if fn_median >= threshold - 0.10:
            notes.append("many_false_negatives_near_threshold")
        elif fn_median < 0.20:
            notes.append("confident_missed_flood")
    if int(metrics.get("fp", 0)) > 2 * max(1, int(metrics.get("tp", 0))):
        notes.append("fp_more_than_twice_tp")
    if not notes:
        notes.append("balanced_or_no_single_dominant_failure")
    return notes


def _norm(a: np.ndarray) -> np.ndarray:
    finite = a[np.isfinite(a)]
    if not finite.size:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _write_montage(path: Path, title: str, vv: np.ndarray, vh: np.ndarray, dem: np.ndarray | None,
                   probability: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    valid = target != 255
    err = np.zeros((*pred.shape, 3), dtype=np.float32)
    err[valid & (pred == 1) & (target == 1)] = (0, 1, 0)
    err[valid & (pred == 1) & (target == 0)] = (1, 0, 0)
    err[valid & (pred == 0) & (target == 1)] = (0, 0.4, 1)
    panels = [("VV", _norm(vv), "gray"), ("VH", _norm(vh), "gray")]
    if dem is not None:
        panels.append(("DEM", _norm(dem), "terrain"))
    panels.extend([("Probability", probability, "magma"), ("Ground truth", np.where(valid, target, 0), "gray"), ("Prediction", pred, "gray"), ("Errors: TP green / FP red / FN blue", err, None)])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax in axes.ravel(): ax.axis("off")
    for ax, (name, image, cmap) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def audit_deployment_errors(deployment_dirs: list[Path], output_dir: Path, max_montages: int = 30) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    metadata_files: list[Path] = []
    for root in deployment_dirs:
        metadata_files.extend(root.rglob("*_prediction_metadata.json"))

    for meta_path in sorted(set(metadata_files)):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = meta.get("evaluation_metrics")
        mask_path = _resolve_path(meta.get("mask_path"), meta_path.parent, ["*ground_truth*.tif", "*mask*.tif"])
        pred_path = _resolve_path(meta.get("output_mask"), meta_path.parent, ["*_flood_mask.tif"])
        prob_path = _resolve_path(meta.get("output_probability"), meta_path.parent, ["*_flood_probability.tif"])
        sar_path = _resolve_path(meta.get("sar_path"), meta_path.parent, ["*sar*.tif"])
        dem_path = _resolve_path(meta.get("dem_path"), meta_path.parent, ["*dem*.tif"])
        if not metrics or not mask_path or not pred_path or not prob_path or not sar_path:
            continue
        pred, ref = _read_band(pred_path)
        pred = (pred > 0).astype(np.uint8)
        probability = _read_to_grid(prob_path, ref).astype(np.float32)
        target = _read_to_grid(mask_path, ref, categorical=True).astype(np.uint8)
        vv, _ = _read_band(sar_path, 1)
        vv = _read_to_grid(sar_path, ref, band=1).astype(np.float32)
        with rasterio.open(sar_path) as src:
            vh_band = 2 if src.count >= 2 else 1
        vh = _read_to_grid(sar_path, ref, band=vh_band).astype(np.float32)
        dem = _read_to_grid(dem_path, ref).astype(np.float32) if dem_path else None
        valid = target != 255
        masks = {
            "tp": valid & (pred == 1) & (target == 1),
            "fp": valid & (pred == 1) & (target == 0),
            "fn": valid & (pred == 0) & (target == 1),
            "tn": valid & (pred == 0) & (target == 0),
        }
        threshold = float(meta.get("threshold", 0.5))
        diagnoses = _diagnose(metrics, probability, pred, target, threshold)
        item = {
            "candidate_id": meta.get("candidate_id", meta_path.parent.name),
            "source_directory": str(meta_path.parent),
            "metrics": metrics,
            "threshold": threshold,
            "diagnoses": diagnoses,
            "probability_by_error": {k: _stats(probability[m]) for k, m in masks.items()},
            "vv_by_error": {k: _stats(vv[m]) for k, m in masks.items()},
            "vh_by_error": {k: _stats(vh[m]) for k, m in masks.items()},
            "dem_by_error": {k: _stats(dem[m]) for k, m in masks.items()} if dem is not None else None,
        }
        details.append(item)
        rows.append({
            "candidate_id": item["candidate_id"], "f1": metrics.get("f1"), "iou": metrics.get("iou"),
            "precision": metrics.get("precision"), "recall": metrics.get("recall"), "mcc": metrics.get("mcc"),
            "tp": metrics.get("tp"), "fp": metrics.get("fp"), "fn": metrics.get("fn"), "tn": metrics.get("tn"),
            "flood_fraction": meta.get("flood_fraction"), "diagnosis": ";".join(diagnoses),
            "fp_probability_median": item["probability_by_error"]["fp"]["median"],
            "fn_probability_median": item["probability_by_error"]["fn"]["median"],
        })
        item["_arrays"] = (vv, vh, dem, probability, pred, target)

    rows.sort(key=lambda r: (float(r["f1"] or 0.0), -int(r["fp"] or 0)))
    ranked_ids = {r["candidate_id"] for r in rows[:max_montages]}
    for item in details:
        if item["candidate_id"] in ranked_ids:
            vv, vh, dem, probability, pred, target = item.pop("_arrays")
            montage = output_dir / "montages" / f"{item['candidate_id']}_error_forensics.png"
            _write_montage(montage, f"{item['candidate_id']} | {', '.join(item['diagnoses'])}", vv, vh, dem, probability, pred, target)
            item["montage"] = str(montage)
        else:
            item.pop("_arrays", None)

    csv_path = output_dir / "deployment_error_ranking.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    report = {
        "deployment_dirs": [str(p) for p in deployment_dirs], "candidates_analysed": len(rows),
        "ranking_csv": str(csv_path), "details": details,
    }
    (output_dir / "deployment_error_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table_rows = "".join(f"<tr><td>{html.escape(str(r['candidate_id']))}</td><td>{float(r['f1'] or 0):.4f}</td><td>{float(r['precision'] or 0):.4f}</td><td>{float(r['recall'] or 0):.4f}</td><td>{int(r['fp'] or 0):,}</td><td>{html.escape(str(r['diagnosis']))}</td></tr>" for r in rows)
    montage_html = "".join(f"<h3>{html.escape(str(d['candidate_id']))}</h3><p>{html.escape(', '.join(d['diagnoses']))}</p><img src='{Path(d['montage']).relative_to(output_dir).as_posix()}' style='max-width:100%;height:auto'>" for d in details if d.get("montage"))
    (output_dir / "deployment_error_audit.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Deployment error audit</title><style>body{{font-family:Arial;max-width:1400px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;text-align:left}}th{{background:#eee}}</style></head><body><h1>Deployment error forensic audit</h1><p>Ranked worst first. Diagnostics are evidence-based heuristics, not land-cover labels.</p><table><tr><th>Candidate</th><th>F1</th><th>Precision</th><th>Recall</th><th>FP</th><th>Diagnosis</th></tr>{table_rows}</table>{montage_html}</body></html>", encoding="utf-8")
    LOG.info("Deployment audit analysed %d labelled candidate(s).", len(rows))
    if rows:
        LOG.info("Deployment audit worst candidates:")
        for r in rows[:min(10, len(rows))]:
            LOG.info("  %s: F1=%.4f P=%.4f R=%.4f | %s", r["candidate_id"], float(r["f1"] or 0), float(r["precision"] or 0), float(r["recall"] or 0), r["diagnosis"])
    LOG.info("Deployment audit report: %s", output_dir / "deployment_error_audit.html")
    return report
