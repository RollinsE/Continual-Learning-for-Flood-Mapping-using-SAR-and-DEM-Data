from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import rasterio

from floods.utils.common import get_logger

LOG = get_logger(__name__)

_EVENT_RE = re.compile(r"(EMSR\d+)", re.IGNORECASE)
_AREA_RE = re.compile(r"(EMSR\d+-\d+)", re.IGNORECASE)
_SCENE_RE = re.compile(r"(EMSR\d+-\d+-\d+)", re.IGNORECASE)


@dataclass
class TileRecord:
    split: str
    tile: str
    event_id: str
    area_id: str
    scene_id: str
    valid_pixels: int
    flood_pixels: int
    ignore_pixels: int
    flood_fraction: float


def _extract_id(stem: str, pattern: re.Pattern[str], fallback: str) -> str:
    match = pattern.search(stem)
    return match.group(1).upper() if match else fallback


def _ids_from_stem(stem: str) -> tuple[str, str, str]:
    event = _extract_id(stem, _EVENT_RE, "UNKNOWN")
    area = _extract_id(stem, _AREA_RE, event)
    scene = _extract_id(stem, _SCENE_RE, area)
    return event, area, scene


def _mask_stats(mask_path: Path) -> tuple[int, int, int, float]:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    valid = mask != 255
    valid_pixels = int(valid.sum())
    flood_pixels = int(((mask == 1) & valid).sum())
    ignore_pixels = int((mask == 255).sum())
    fraction = float(flood_pixels / valid_pixels) if valid_pixels else 0.0
    return valid_pixels, flood_pixels, ignore_pixels, fraction


def _discover_records(root: Path, splits: Sequence[str]) -> List[TileRecord]:
    records: List[TileRecord] = []
    for split in splits:
        mask_dir = root / split / "mask"
        sar_dir = root / split / "sar"
        if not mask_dir.exists():
            LOG.warning("Skipping missing split: %s", split)
            continue
        masks = sorted(mask_dir.glob("*.tif"))
        if not masks:
            LOG.warning("No masks found for split: %s", split)
            continue
        missing_sar = [p.name for p in masks if not (sar_dir / p.name).exists()]
        if missing_sar:
            raise FileNotFoundError(f"{split}: {len(missing_sar)} mask(s) have no matching SAR tile; first={missing_sar[0]}")
        LOG.info("Auditing %s: %d tiles", split, len(masks))
        for idx, mask_path in enumerate(masks, start=1):
            event, area, scene = _ids_from_stem(mask_path.stem)
            valid, flood, ignore, fraction = _mask_stats(mask_path)
            records.append(TileRecord(split, mask_path.stem, event, area, scene, valid, flood, ignore, fraction))
            if idx % 500 == 0 or idx == len(masks):
                LOG.info("%s mask scan: %d/%d", split, idx, len(masks))
    if not records:
        raise ValueError(f"No processed mask tiles found under {root}")
    return records


def _overlap_map(records: Iterable[TileRecord], attr: str) -> Dict[str, List[str]]:
    membership: Dict[str, Set[str]] = defaultdict(set)
    for record in records:
        membership[getattr(record, attr)].add(record.split)
    return {key: sorted(splits) for key, splits in membership.items() if len(splits) > 1 and key != "UNKNOWN"}


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: 0.0 for k in ("mean", "median", "p90", "p95", "p99")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _split_summary(records: List[TileRecord]) -> Dict[str, dict]:
    output: Dict[str, dict] = {}
    for split in sorted({r.split for r in records}):
        rows = [r for r in records if r.split == split]
        fractions = [r.flood_fraction for r in rows]
        output[split] = {
            "tiles": len(rows),
            "events": len({r.event_id for r in rows}),
            "areas": len({r.area_id for r in rows}),
            "scenes": len({r.scene_id for r in rows}),
            "empty_tiles": sum(r.flood_pixels == 0 for r in rows),
            "empty_tile_fraction": float(sum(r.flood_pixels == 0 for r in rows) / len(rows)),
            "flood_fraction": _quantiles(fractions),
        }
    return output


def _group_summary(records: List[TileRecord], attr: str) -> List[dict]:
    groups: Dict[tuple[str, str], List[TileRecord]] = defaultdict(list)
    for record in records:
        groups[(record.split, getattr(record, attr))].append(record)
    rows: List[dict] = []
    for (split, group_id), items in sorted(groups.items()):
        total_valid = sum(x.valid_pixels for x in items)
        total_flood = sum(x.flood_pixels for x in items)
        rows.append({
            "split": split,
            "group_level": attr.removesuffix("_id"),
            "group_id": group_id,
            "tiles": len(items),
            "empty_tiles": sum(x.flood_pixels == 0 for x in items),
            "pixel_weighted_flood_fraction": float(total_flood / total_valid) if total_valid else 0.0,
            **{f"tile_{k}": v for k, v in _quantiles([x.flood_fraction for x in items]).items()},
        })
    return rows


def audit_training_data(processed_data_dir: Path,
                        output_dir: Path,
                        splits: Sequence[str] = ("train", "val", "test"),
                        fail_on_leakage: bool = False) -> dict:
    """Audit split isolation and target composition before further training.

    The audit checks overlap at three levels: EMSR event (EMSR342), mapped area
    (EMSR342-5), and source scene (EMSR342-5-3). Scene-level overlap is the most
    serious because tiles from the same source raster can otherwise appear in
    both training and evaluation splits.
    """
    root = Path(processed_data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = _discover_records(root, splits)

    overlaps = {
        "event": _overlap_map(records, "event_id"),
        "area": _overlap_map(records, "area_id"),
        "scene": _overlap_map(records, "scene_id"),
        "exact_tile": _overlap_map(records, "tile"),
    }
    summary = {
        "processed_data_dir": str(root),
        "splits": _split_summary(records),
        "overlap": overlaps,
        "interpretation": {
            "exact_tile_overlap": "definite leakage",
            "scene_overlap": "high-risk leakage: tiles from the same source scene occur in multiple splits",
            "area_overlap": "potential spatial leakage: mapped areas occur in multiple splits",
            "event_overlap": "event-level dependence: the same emergency event occurs in multiple splits",
        },
    }

    with (out / "training_data_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with (out / "training_data_tiles.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)
    group_rows = _group_summary(records, "event_id") + _group_summary(records, "area_id") + _group_summary(records, "scene_id")
    with (out / "training_data_groups.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    LOG.info("Training-data audit")
    for split, item in summary["splits"].items():
        ff = item["flood_fraction"]
        LOG.info(
            "%s: tiles=%d events=%d areas=%d scenes=%d empty=%d (%.1f%%) | flood fraction median=%.4f mean=%.4f p95=%.4f",
            split, item["tiles"], item["events"], item["areas"], item["scenes"], item["empty_tiles"],
            100.0 * item["empty_tile_fraction"], ff["median"], ff["mean"], ff["p95"],
        )
    LOG.info(
        "Cross-split overlap: exact_tiles=%d scenes=%d areas=%d events=%d",
        len(overlaps["exact_tile"]), len(overlaps["scene"]), len(overlaps["area"]), len(overlaps["event"]),
    )
    if overlaps["exact_tile"]:
        LOG.error("Definite leakage detected: identical tile names occur across splits.")
    if overlaps["scene"]:
        example = next(iter(overlaps["scene"].items()))
        LOG.warning("Scene-level overlap detected; example=%s splits=%s", example[0], example[1])
    elif overlaps["area"]:
        LOG.warning("No scene overlap, but mapped-area overlap exists across splits.")
    elif overlaps["event"]:
        LOG.warning("No scene/area overlap, but EMSR events occur across splits; this is not a strict unseen-event validation.")
    else:
        LOG.info("PASS: no event, area, scene, or exact-tile overlap across the requested splits.")
    LOG.info("Audit files written to: %s", out)

    leakage = bool(overlaps["exact_tile"] or overlaps["scene"])
    if fail_on_leakage and leakage:
        raise RuntimeError("Training-data audit failed because exact-tile or source-scene overlap was detected")
    return summary
