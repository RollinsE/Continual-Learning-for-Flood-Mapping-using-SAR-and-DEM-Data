from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd
import rasterio
from floods.utils.common import get_logger

LOG = get_logger(__name__)


def _event_id(path: Path) -> str:
    match = re.search(r"(EMSR\d+)", path.name)
    return match.group(1) if match else "unknown"


def _same_transform(a, b, precision: int = 9) -> bool:
    return tuple(round(float(v), precision) for v in a) == tuple(round(float(v), precision) for v in b)


def _raster_row(sar_path: Path, dem_path: Path, mask_path: Path) -> dict:
    with rasterio.open(sar_path) as sar, rasterio.open(dem_path) as dem, rasterio.open(mask_path) as mask:
        row = {
            "event_id": _event_id(sar_path),
            "stem": sar_path.stem,
            "sar_path": str(sar_path),
            "dem_path": str(dem_path),
            "mask_path": str(mask_path),
            "sar_shape": (sar.height, sar.width),
            "dem_shape": (dem.height, dem.width),
            "mask_shape": (mask.height, mask.width),
            "sar_crs": str(sar.crs),
            "dem_crs": str(dem.crs),
            "mask_crs": str(mask.crs),
            "sar_transform": tuple(float(v) for v in sar.transform)[:6],
            "dem_transform": tuple(float(v) for v in dem.transform)[:6],
            "mask_transform": tuple(float(v) for v in mask.transform)[:6],
            "sar_nodata": sar.nodata,
            "dem_nodata": dem.nodata,
            "mask_nodata": mask.nodata,
        }
        row["shape_aligned"] = row["sar_shape"] == row["dem_shape"] == row["mask_shape"]
        row["crs_aligned"] = row["sar_crs"] == row["dem_crs"] == row["mask_crs"]
        row["transform_aligned"] = _same_transform(sar.transform, dem.transform) and _same_transform(sar.transform, mask.transform)
        row["aligned"] = bool(row["shape_aligned"] and row["crs_aligned"] and row["transform_aligned"])
        return row


def audit_raw_alignment(raw_data_dir: Path, output_dir: Path, include_events: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Audit CRS, transform, shape, and nodata consistency across SAR/DEM/mask triplets."""
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include: Set[str] = {str(v).upper() for v in (include_events or [])}
    sar_files = sorted(Path(p) for p in glob(str(raw_data_dir / "*" / "s1_raw" / "*.tif")))
    rows = []
    for sar_path in sar_files:
        event = _event_id(sar_path).upper()
        if include and event not in include:
            continue
        root = sar_path.parents[1]
        dem_matches = sorted((root / "DEM").glob(f"{sar_path.stem}.tif"))
        mask_matches = sorted((root / "mask").glob(f"{sar_path.stem}.tif"))
        if not dem_matches or not mask_matches:
            rows.append({"event_id": event, "stem": sar_path.stem, "sar_path": str(sar_path), "missing_dem": not bool(dem_matches), "missing_mask": not bool(mask_matches), "aligned": False})
            continue
        rows.append(_raster_row(sar_path, dem_matches[0], mask_matches[0]))
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "raw_alignment_audit.csv", index=False)
    summary = {
        "raw_data_dir": str(raw_data_dir),
        "rasters": int(len(df)),
        "aligned": int(df.get("aligned", pd.Series(dtype=bool)).sum()) if not df.empty else 0,
        "misaligned": int((~df.get("aligned", pd.Series(dtype=bool))).sum()) if not df.empty else 0,
        "output_csv": str(output_dir / "raw_alignment_audit.csv"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    LOG.info("Raw alignment audit written to: %s", output_dir)
    if summary["misaligned"]:
        LOG.warning("Raw alignment audit found %d potentially misaligned triplets", summary["misaligned"])
    return df


def audit_code_quality(project_root: Path, output_dir: Path, banned_terms: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Scan public source/config/docs for development-history language."""
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    banned_terms = list(banned_terms or [
        "notebook_compat", "NotebookCompat", "quick fix", "hack", "SREENI",
        "It ain't much", "weightning", "Nbr of trainable", "just in case",
        "/content/mmflood_repo", "/content/mmflood_raw", "/content/mmflood_processed",
        "/content/drive/MyDrive", "MyDrive/mmflood_runs", "mmflood_processed_256",
        "preprocess_mmflood_256.yaml", "stats_train_256.yaml",
    ])
    allowed_parts = {"CHANGELOG.md", "deprecated", "pipeline_audit.py", "tests", "COLAB.md"}
    rows: List[dict] = []
    exts = {".py", ".yaml", ".yml", ".md", ".txt"}
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(project_root)
        if any(part in allowed_parts for part in rel.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, start=1):
            for term in banned_terms:
                if term in line:
                    rows.append({"file": str(rel), "line": line_no, "term": term, "text": line.strip()[:240]})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "code_quality_terms.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"project_root": str(project_root), "findings": int(len(df)), "terms": banned_terms}, file, indent=2)
    LOG.info("Code-quality audit written to: %s", output_dir)
    return df
