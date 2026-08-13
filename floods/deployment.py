from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np
import rasterio
import torch
import yaml
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject

from floods.config import TrainConfig
from floods.utils.common import get_logger

LOG = get_logger(__name__)
_DEPLOY_OUTPUT_MODE = "standard"


def _set_deploy_output_mode(mode: str | None) -> None:
    """Set deployment console verbosity for the current process."""
    global _DEPLOY_OUTPUT_MODE
    mode = str(mode or "standard").lower().strip()
    if mode not in {"concise", "standard", "verbose"}:
        mode = "standard"
    _DEPLOY_OUTPUT_MODE = mode


def _deploy_rank() -> int:
    return {"concise": 0, "standard": 1, "verbose": 2}.get(_DEPLOY_OUTPUT_MODE, 1)


def _deploy_log(message: str, *args: object, level: str = "standard") -> None:
    """Emit a human-readable deployment progress message.

    Levels:
    - summary: always shown, even in concise mode
    - standard: shown in standard and verbose modes
    - verbose: shown only in verbose mode
    """
    required = {"summary": 0, "standard": 1, "verbose": 2}.get(level, 1)
    if _deploy_rank() >= required:
        LOG.info("[DEPLOY] " + message, *args)


def _deploy_is_concise() -> bool:
    return _deploy_rank() == 0


RASTER_SUFFIXES = {".tif", ".tiff"}
DEM_TERMS = ("dem", "srtm", "elevation", "height")
MASK_TERMS = ("mask", "label", "truth", "gt", "groundtruth", "ground_truth")
OUTPUT_TERMS = ("prediction", "probability", "flood_mask", "overlay", "preview", "explanation", "uncertainty")


@dataclass
class SceneCandidate:
    candidate_id: str
    kind: str
    date: Optional[str]
    sar_path: Optional[Path] = None
    vv_path: Optional[Path] = None
    vh_path: Optional[Path] = None
    status: str = "ready"
    reason: str = ""
    crs: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    count: Optional[int] = None
    bounds: Optional[tuple[float, float, float, float]] = None
    res: Optional[tuple[float, float]] = None
    dem_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    mosaic_group: Optional[str] = None

    def to_row(self) -> dict[str, Any]:
        bounds = self.bounds or (None, None, None, None)
        res = self.res or (None, None)
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "date": self.date or "",
            "status": self.status,
            "reason": self.reason,
            "sar_path": str(self.sar_path) if self.sar_path else "",
            "vv_path": str(self.vv_path) if self.vv_path else "",
            "vh_path": str(self.vh_path) if self.vh_path else "",
            "dem_path": str(self.dem_path) if self.dem_path else "",
            "mask_path": str(self.mask_path) if self.mask_path else "",
            "mosaic_group": self.mosaic_group or "",
            "crs": self.crs or "",
            "height": self.height or "",
            "width": self.width or "",
            "count": self.count or "",
            "left": "" if bounds[0] is None else bounds[0],
            "bottom": "" if bounds[1] is None else bounds[1],
            "right": "" if bounds[2] is None else bounds[2],
            "top": "" if bounds[3] is None else bounds[3],
            "res_x": "" if res[0] is None else res[0],
            "res_y": "" if res[1] is None else res[1],
        }


def _load_trusted_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = yaml.load(text, Loader=yaml.FullLoader)
    return data or {}


def _load_train_config(path: Path) -> TrainConfig:
    return TrainConfig(**_load_trusted_yaml(Path(path)))


def _resolve_manifest_reference(manifest_path: Path, value: str | Path, *, role: str) -> Path:
    """Resolve an absolute or manifest-relative file reference.

    Deployment manifests are portable when their asset paths are relative to the
    manifest directory.  Absolute paths remain supported for legacy manifests.
    """
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Deployment manifest contains an empty {role} path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    return path.resolve()


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_file_record(path: Path, *, role: str) -> dict[str, Any]:
    path = Path(path)
    return {
        "role": role,
        "path": path,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _copy_bundle_file(source: Path, destination: Path, *, role: str) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Deployment {role} not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        _deploy_log("Bundling %s: %s -> %s", role, source, destination)
        with source.open("rb") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        shutil.copystat(source, destination)
    return _bundle_file_record(destination, role=role)


def _write_portable_normalization_stats(source: Path, destination: Path) -> dict[str, Any]:
    """Write deployment normalization statistics without local training paths.

    Normalization inference uses the fitted channel statistics, not the original
    processed-dataset location.  Training provenance fields that can contain
    machine-specific paths are therefore omitted from the portable copy while
    the original training artifact remains untouched.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Deployment normalization_stats not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Normalization statistics must contain a JSON object: {source}")

    portable_payload = dict(payload)
    portable_payload.pop("processed_data_dir", None)
    portable_payload.pop("preserve_channel_stats_from", None)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(portable_payload, indent=2), encoding="utf-8")
    _deploy_log(
        "Bundling normalization_stats as path-neutral deployment metadata: %s -> %s",
        source,
        destination,
    )
    return _bundle_file_record(destination, role="normalization_stats")


def _write_portable_config(
    source: Path,
    destination: Path,
    *,
    normalization_destination: Optional[Path],
    role: str,
) -> dict[str, Any]:
    """Write a deployment-only config without machine-specific training paths."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Deployment {role} not found: {source}")
    payload = _load_trusted_yaml(source)
    data = payload.setdefault("data", {})
    data["path"] = "."
    data["cache_dir"] = "cache"
    if normalization_destination is not None:
        relative_stats = os.path.relpath(normalization_destination, start=Path(destination).parent)
        data["normalization_stats_path"] = Path(relative_stats).as_posix()
    else:
        data["normalization_stats_path"] = None
    payload["output_folder"] = "outputs"
    payload["init_checkpoint"] = None
    payload["resume"] = False
    payload["resume_from"] = None

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _deploy_log("Bundling %s as path-neutral config: %s -> %s", role, source, destination)
    return _bundle_file_record(destination, role=role)


def _manifest_relative(path: Path, manifest_path: Path) -> str:
    relative = os.path.relpath(Path(path), start=Path(manifest_path).parent)
    return Path(relative).as_posix()


def _configured_normalization_stats(config_path: Path) -> tuple[Optional[Path], Optional[str]]:
    payload = _load_trusted_yaml(Path(config_path))
    data = payload.get("data") or {}
    mode = data.get("normalization_mode")
    value = data.get("normalization_stats_path")
    if not value:
        return None, str(mode) if mode else None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return path.resolve(), str(mode) if mode else None


def _write_bundle_readme(path: Path, manifest_name: str) -> Path:
    content = f"""# Flood Extent Mapping deployment bundle

This directory is self-contained: the manifest refers only to files inside this
bundle.  Move or copy the entire directory together.

Run prediction after installing the matching Flood Extent Mapping package:

```bash
floodmap predict-scene \
  --manifest {manifest_name} \
  --sar-path /path/to/vv_vh_scene.tif \
  --output-dir outputs \
  --write-probability \
  --write-html-report
```

The `deployment_bundle.json` inventory records file sizes and SHA-256 hashes for
the bundled model assets.
"""
    Path(path).write_text(content, encoding="utf-8")
    return Path(path)


def _read_manifest(path: Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "scene"


def _extract_date(path: Path) -> Optional[str]:
    text = path.as_posix()
    match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", text)
    return match.group(1) if match else None


def _format_candidate_id(*,
                         stem: str,
                         date: Optional[str] = None,
                         scene_id: Optional[str] = None,
                         candidate_prefix: Optional[str] = None,
                         candidate_name_template: Optional[str] = None,
                         index: Optional[int] = None,
                         kind: str = "scene") -> str:
    """Build a readable candidate identifier without forcing an ``undated_`` prefix.

    ``candidate_name_template`` may use ``{scene_id}``, ``{date}``, ``{stem}``,
    ``{index}`` and ``{kind}``.  Missing dates are rendered as an empty string so
    fallbacks remain clean, for example ``EMSR107-7-2`` instead of
    ``undated_EMSR107-7-2``.
    """
    values = {
        "scene_id": scene_id or "",
        "date": date or "",
        "stem": stem or "scene",
        "index": "" if index is None else f"{int(index):02d}",
        "kind": kind or "scene",
    }
    if candidate_name_template:
        try:
            return _safe_name(candidate_name_template.format(**values))
        except Exception as exc:
            raise ValueError(f"Invalid candidate name template {candidate_name_template!r}: {exc}") from exc
    parts = []
    if candidate_prefix:
        parts.append(candidate_prefix)
    elif scene_id:
        parts.append(scene_id)
    if date:
        parts.append(date)
    parts.append(stem or "scene")
    if index is not None and not str(stem).endswith(f"_{int(index):02d}"):
        parts.append(f"{int(index):02d}")
    return _safe_name("_".join(str(part) for part in parts if str(part)))


def _detect_polarization(path: Path) -> Optional[str]:
    text = f"_{path.stem.lower()}_"
    if re.search(r"[^a-z0-9]vv[^a-z0-9]", text):
        return "vv"
    if re.search(r"[^a-z0-9]vh[^a-z0-9]", text):
        return "vh"
    if "_vv" in text or "vv_" in text:
        return "vv"
    if "_vh" in text or "vh_" in text:
        return "vh"
    return None


def _looks_like_dem(path: Path) -> bool:
    name = path.as_posix().lower()
    return any(term in name for term in DEM_TERMS)


def _looks_like_mask(path: Path) -> bool:
    name = path.as_posix().lower()
    return any(term in name for term in MASK_TERMS)


def _looks_like_output(path: Path) -> bool:
    name = path.as_posix().lower()
    return any(term in name for term in OUTPUT_TERMS)


def _raster_info(path: Path) -> dict[str, Any]:
    with rasterio.open(str(path)) as src:
        return {
            "path": str(path),
            "count": int(src.count),
            "height": int(src.height),
            "width": int(src.width),
            "crs": str(src.crs) if src.crs else "",
            "transform": tuple(float(v) for v in src.transform)[:6],
            "bounds": tuple(float(v) for v in src.bounds),
            "res": tuple(float(v) for v in src.res),
        }


def _profile_bounds(profile: dict) -> tuple[float, float, float, float] | None:
    """Return raster bounds as (left, bottom, right, top) from a Rasterio profile."""
    if profile.get("bounds") is not None:
        try:
            b = profile["bounds"]
            return (float(b.left), float(b.bottom), float(b.right), float(b.top)) if hasattr(b, "left") else tuple(float(v) for v in b)  # type: ignore[return-value]
        except Exception:
            pass
    transform = profile.get("transform")
    height = profile.get("height")
    width = profile.get("width")
    if transform is None or height is None or width is None:
        return None
    try:
        left, bottom, right, top = array_bounds(int(height), int(width), transform)
        return (float(left), float(bottom), float(right), float(top))
    except Exception:
        return None


def _bounds_overlap_ratio(a: Any, b: Any) -> float:
    """Intersection-over-smaller-area for two bounds-like objects."""
    try:
        if hasattr(a, "left"):
            a_vals = (float(a.left), float(a.bottom), float(a.right), float(a.top))
        else:
            a_vals = tuple(float(v) for v in a)
        if hasattr(b, "left"):
            b_vals = (float(b.left), float(b.bottom), float(b.right), float(b.top))
        else:
            b_vals = tuple(float(v) for v in b)
        a_left, a_bottom, a_right, a_top = a_vals
        b_left, b_bottom, b_right, b_top = b_vals
        left = max(a_left, b_left)
        right = min(a_right, b_right)
        bottom = max(a_bottom, b_bottom)
        top = min(a_top, b_top)
        if right <= left or top <= bottom:
            return 0.0
        intersection = (right - left) * (top - bottom)
        a_area = max(0.0, (a_right - a_left) * (a_top - a_bottom))
        b_area = max(0.0, (b_right - b_left) * (b_top - b_bottom))
        smaller = min(a_area, b_area)
        return float(intersection / smaller) if smaller > 0 else 0.0
    except Exception:
        return 0.0


def _mask_alignment_info(mask_path: Path, profile: dict, shape: tuple[int, int]) -> dict[str, Any]:
    """Collect quick spatial checks for SAR/mask pairing before evaluation."""
    sar_crs = profile.get("crs")
    sar_transform = profile.get("transform")
    sar_bounds = _profile_bounds(profile)
    height, width = shape
    with rasterio.open(str(mask_path)) as src:
        mask_bounds = tuple(float(v) for v in src.bounds)
        crs_match = bool(src.crs == sar_crs)
        shape_match = bool(src.height == height and src.width == width)
        transform_match = bool(src.transform == sar_transform)
        overlap = _bounds_overlap_ratio(sar_bounds, mask_bounds) if sar_bounds else 0.0
        return {
            "mask_path": str(mask_path),
            "sar_crs": str(sar_crs) if sar_crs else "",
            "mask_crs": str(src.crs) if src.crs else "",
            "crs_match": crs_match,
            "shape_match": shape_match,
            "transform_match": transform_match,
            "bounds_overlap_ratio": float(overlap),
            "sar_shape": [int(height), int(width)],
            "mask_shape": [int(src.height), int(src.width)],
            "sar_bounds": list(sar_bounds) if sar_bounds else None,
            "mask_bounds": list(mask_bounds),
            "resampled_to_sar_grid": not (crs_match and shape_match and transform_match),
        }


def _candidate_sort_key(candidate: SceneCandidate) -> tuple[str, str]:
    return (candidate.date or "9999-99-99", candidate.candidate_id)


def _common_pair_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(?<!\d)20\d{2}[-_]?\d{2}[-_]?\d{2}(?!\d)", "", stem)
    stem = re.sub(r"(?<!\d)20\d{2}(?!\d)", "", stem)
    stem = re.sub(r"(^|[^a-z0-9])v[hv]($|[^a-z0-9])", "_", stem)
    stem = re.sub(r"[_\-.]+", "_", stem).strip("_")
    return stem or path.parent.name.lower()


def _inventory_keys() -> list[str]:
    return [
        "candidate_id", "kind", "date", "status", "reason",
        "sar_path", "vv_path", "vh_path", "dem_path", "mask_path", "mosaic_group",
        "crs", "height", "width", "count", "left", "bottom", "right", "top", "res_x", "res_y",
    ]


def _write_inventory_csv(candidates: Sequence[SceneCandidate], output_file: Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_row() for candidate in candidates]
    keys = _inventory_keys()
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_from_row(row: dict[str, Any]) -> SceneCandidate:
    def path_or_none(value: Any) -> Optional[Path]:
        text = str(value or "").strip()
        return Path(text) if text else None

    def int_or_none(value: Any) -> Optional[int]:
        try:
            text = str(value or "").strip()
            return int(text) if text else None
        except Exception:
            return None

    def float_or_none(value: Any) -> Optional[float]:
        try:
            text = str(value or "").strip()
            return float(text) if text else None
        except Exception:
            return None

    bounds_values = [float_or_none(row.get(key)) for key in ["left", "bottom", "right", "top"]]
    bounds = tuple(bounds_values) if all(value is not None for value in bounds_values) else None
    res_values = [float_or_none(row.get(key)) for key in ["res_x", "res_y"]]
    res = tuple(res_values) if all(value is not None for value in res_values) else None
    candidate_id = str(row.get("candidate_id") or "").strip()
    sar = path_or_none(row.get("sar_path"))
    if not candidate_id:
        source = sar or path_or_none(row.get("vv_path")) or path_or_none(row.get("vh_path"))
        candidate_id = _safe_name(source.stem if source else "scene")
    return SceneCandidate(
        candidate_id=_safe_name(candidate_id),
        kind=str(row.get("kind") or ("multiband_vv_vh" if sar else "separate_vv_vh")),
        date=str(row.get("date") or "").strip() or None,
        sar_path=sar,
        vv_path=path_or_none(row.get("vv_path")),
        vh_path=path_or_none(row.get("vh_path")),
        dem_path=path_or_none(row.get("dem_path")),
        mask_path=path_or_none(row.get("mask_path")),
        mosaic_group=str(row.get("mosaic_group") or "").strip() or None,
        status=str(row.get("status") or "ready"),
        reason=str(row.get("reason") or ""),
        crs=str(row.get("crs") or "") or None,
        height=int_or_none(row.get("height")),
        width=int_or_none(row.get("width")),
        count=int_or_none(row.get("count")),
        bounds=bounds,  # type: ignore[arg-type]
        res=res,  # type: ignore[arg-type]
    )


def _candidates_from_input_csv(input_csv: Path) -> list[SceneCandidate]:
    """Read explicitly supplied deployment candidates from a CSV file.

    Supported columns are the inventory columns plus optional ``dem_path``,
    ``mask_path`` and ``mosaic_group``.  ``mosaic_group`` lets users explicitly
    state which rows should be mosaicked together before prediction.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        candidates = [_candidate_from_row(row) for row in reader]
    if not candidates:
        raise ValueError(f"Input CSV contains no deployment candidates: {input_csv}")
    for candidate in candidates:
        if candidate.sar_path is None and (candidate.vv_path is None or candidate.vh_path is None):
            candidate.status = "not_ready"
            candidate.reason = (candidate.reason + " " if candidate.reason else "") + "Provide either sar_path or both vv_path and vh_path."
    return candidates


def _candidate_from_path(path: Path,
                         info: dict[str, Any],
                         scene_id: Optional[str] = None,
                         candidate_prefix: Optional[str] = None,
                         candidate_name_template: Optional[str] = None) -> SceneCandidate:
    date = _extract_date(path)
    candidate_id = _format_candidate_id(stem=path.stem,
                                        date=date,
                                        scene_id=scene_id,
                                        candidate_prefix=candidate_prefix,
                                        candidate_name_template=candidate_name_template,
                                        kind="multiband_vv_vh")
    return SceneCandidate(candidate_id=candidate_id,
                          kind="multiband_vv_vh",
                          date=date,
                          sar_path=path,
                          crs=info.get("crs"),
                          height=info.get("height"),
                          width=info.get("width"),
                          count=info.get("count"),
                          bounds=tuple(info.get("bounds")) if info.get("bounds") else None,
                          res=tuple(info.get("res")) if info.get("res") else None)


def discover_scene(scene_dir: Path,
                   output_file: Optional[Path] = None,
                   scene_id: Optional[str] = None,
                   candidate_prefix: Optional[str] = None,
                   candidate_name_template: Optional[str] = None) -> list[dict[str, Any]]:
    """Inventory an event folder and group deployable SAR candidates.

    The command accepts analysis-ready GeoTIFFs.  Candidate IDs no longer force an
    ``undated_`` prefix; if no acquisition date can be found the clean file stem is
    used and the ``date`` column is left empty.  The inventory also records bounds
    and pixel size so users can decide whether tiles are spatially adjacent.
    """
    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    raster_files = sorted(path for path in scene_dir.rglob("*") if path.suffix.lower() in RASTER_SUFFIXES and path.is_file())
    candidates: list[SceneCandidate] = []
    separate: dict[tuple[str, str], dict[str, list[tuple[Path, dict[str, Any]]]]] = {}

    for path in raster_files:
        if _looks_like_output(path) or _looks_like_dem(path) or _looks_like_mask(path):
            continue
        try:
            info = _raster_info(path)
        except Exception as exc:  # pragma: no cover - defensive inventory path
            candidates.append(SceneCandidate(candidate_id=_safe_name(path.stem), kind="unreadable", date=_extract_date(path), sar_path=path, status="not_ready", reason=str(exc)))
            continue
        pol = _detect_polarization(path)
        date = _extract_date(path)
        if int(info["count"]) >= 2 and pol is None:
            candidates.append(_candidate_from_path(path, info, scene_id, candidate_prefix, candidate_name_template))
            continue
        if pol in {"vv", "vh"}:
            key = (date or "", _common_pair_key(path))
            separate.setdefault(key, {"vv": [], "vh": []})[pol].append((path, info))
            continue
        if int(info["count"]) >= 2:
            candidates.append(_candidate_from_path(path, info, scene_id, candidate_prefix, candidate_name_template))

    for (date, key), group in sorted(separate.items()):
        vv_items = sorted(group.get("vv", []), key=lambda item: item[0].as_posix())
        vh_items = sorted(group.get("vh", []), key=lambda item: item[0].as_posix())
        pair_count = min(len(vv_items), len(vh_items))
        for index in range(pair_count):
            vv_path, vv_info = vv_items[index]
            vh_path, vh_info = vh_items[index]
            status = "ready"
            reason = ""
            if vv_info["height"] != vh_info["height"] or vv_info["width"] != vh_info["width"]:
                reason = "VV/VH shapes differ; VH will be resampled to VV grid during prediction."
            if vv_info["crs"] != vh_info["crs"]:
                reason = (reason + " " if reason else "") + "VV/VH CRS differs; reproject externally if this is not intentional."
            candidate_id = _format_candidate_id(stem=key,
                                                date=date or None,
                                                scene_id=scene_id,
                                                candidate_prefix=candidate_prefix,
                                                candidate_name_template=candidate_name_template,
                                                index=index + 1,
                                                kind="separate_vv_vh")
            candidates.append(SceneCandidate(candidate_id=candidate_id,
                                             kind="separate_vv_vh",
                                             date=date or None,
                                             vv_path=vv_path,
                                             vh_path=vh_path,
                                             status=status,
                                             reason=reason,
                                             crs=vv_info["crs"],
                                             height=vv_info["height"],
                                             width=vv_info["width"],
                                             count=2,
                                             bounds=tuple(vv_info.get("bounds")) if vv_info.get("bounds") else None,
                                             res=tuple(vv_info.get("res")) if vv_info.get("res") else None))
        if len(vv_items) != len(vh_items):
            missing = "VH" if len(vv_items) > len(vh_items) else "VV"
            leftovers = vv_items[pair_count:] if len(vv_items) > len(vh_items) else vh_items[pair_count:]
            for path, info in leftovers:
                candidates.append(SceneCandidate(candidate_id=_format_candidate_id(stem=f"{key}_{path.stem}",
                                                                                   date=date or None,
                                                                                   scene_id=scene_id,
                                                                                   candidate_prefix=candidate_prefix,
                                                                                   candidate_name_template=candidate_name_template,
                                                                                   kind="separate_vv_vh"),
                                                 kind="separate_vv_vh",
                                                 date=date or None,
                                                 vv_path=path if missing == "VH" else None,
                                                 vh_path=path if missing == "VV" else None,
                                                 status="not_ready",
                                                 reason=f"Missing matching {missing} raster.",
                                                 crs=info["crs"],
                                                 height=info["height"],
                                                 width=info["width"],
                                                 count=1,
                                                 bounds=tuple(info.get("bounds")) if info.get("bounds") else None,
                                                 res=tuple(info.get("res")) if info.get("res") else None))

    candidates = sorted(candidates, key=_candidate_sort_key)
    if output_file:
        _write_inventory_csv(candidates, Path(output_file))
    _deploy_log("Discovered %d deployable SAR candidate(s) in %s", sum(c.status == "ready" for c in candidates), scene_dir)
    return [candidate.to_row() for candidate in candidates]


def _discover_candidates(scene_dir: Path,
                         scene_id: Optional[str] = None,
                         candidate_prefix: Optional[str] = None,
                         candidate_name_template: Optional[str] = None) -> list[SceneCandidate]:
    return [_candidate_from_row(row) for row in discover_scene(scene_dir,
                                                               scene_id=scene_id,
                                                               candidate_prefix=candidate_prefix,
                                                               candidate_name_template=candidate_name_template)]


def _select_candidates(candidates: Sequence[SceneCandidate], selection: str = "all", sar_date: Optional[str] = None) -> list[SceneCandidate]:
    ready = [candidate for candidate in candidates if candidate.status == "ready"]
    if sar_date:
        selected = [candidate for candidate in ready if (candidate.date or "") == sar_date or (candidate.date or "").replace("-", "") == sar_date.replace("-", "")]
    else:
        selection_raw = str(selection or "all")
        selection_lower = selection_raw.lower()
        if selection_lower == "all":
            selected = ready
        elif selection_lower == "latest":
            dated = [candidate for candidate in ready if candidate.date]
            selected = [max(dated or ready, key=_candidate_sort_key)] if ready else []
        elif selection_lower == "earliest":
            dated = [candidate for candidate in ready if candidate.date]
            selected = [min(dated or ready, key=_candidate_sort_key)] if ready else []
        else:
            selected = [candidate for candidate in ready if candidate.candidate_id == selection_raw]
    if not selected:
        raise ValueError("No ready SAR candidate matched the requested selection. Run discover-scene to inspect available files.")
    return selected


def _candidate_stem_group(candidate: SceneCandidate) -> str:
    """Best-effort scene-level stem used only for mosaic planning.

    This deliberately keeps the automatic grouping conservative.  For names such
    as EMSR107-7-2 and EMSR107-7-3, the group becomes EMSR107-7.  For names with
    explicit tile suffixes, such as scene_tile_01, the group becomes scene.
    """
    source = candidate.sar_path or candidate.vv_path or candidate.vh_path
    stem = Path(source).stem if source else candidate.candidate_id
    stem = re.sub(r"(?i)([_-]?tile[_-]?\d+)$", "", stem)
    stem = re.sub(r"(?i)([_-]?part[_-]?\d+)$", "", stem)
    stem = re.sub(r"([_-]\d+)$", "", stem)
    return _safe_name(stem or candidate.candidate_id)


def _candidate_group_key(candidate: SceneCandidate, allow_undated: bool = False, use_name_group: bool = False) -> Optional[str]:
    """Return the conservative mosaic group key for a candidate."""
    if candidate.mosaic_group:
        return _safe_name(candidate.mosaic_group)
    if candidate.date:
        return _safe_name(candidate.date)
    if use_name_group:
        return _candidate_stem_group(candidate)
    if allow_undated:
        # Explicit opt-in: the user accepts that undated files should be treated as
        # belonging to one scene group if their grids are compatible.
        return "undated"
    return None


def _planned_mosaic_groups(candidates: Sequence[SceneCandidate], use_name_group: bool = True) -> dict[str, list[SceneCandidate]]:
    groups: dict[str, list[SceneCandidate]] = {}
    for candidate in candidates:
        key = _candidate_group_key(candidate, allow_undated=False, use_name_group=use_name_group)
        if key:
            groups.setdefault(key, []).append(candidate)
    return {key: group for key, group in groups.items() if len(group) > 1}


def _mosaic_group_readiness(group: Sequence[SceneCandidate]) -> tuple[bool, str, list[Path]]:
    paths: list[Path] = []
    for candidate in group:
        paths.extend(_candidate_raster_paths_for_mosaic(candidate))
    if len(paths) != len(group):
        return False, "only multiband VV/VH SAR candidates can be mosaicked automatically", paths
    compatible, reason, _ = _mosaic_compatibility(paths)
    return compatible, reason, paths


def _log_mosaic_plan(candidates: Sequence[SceneCandidate], *, mode: str, evaluating: bool) -> None:
    """Print a user-facing mosaic plan so users do not have to inspect CSVs first."""
    if len(candidates) < 2:
        return
    groups = _planned_mosaic_groups(candidates, use_name_group=True)
    if not groups:
        _deploy_log("Mosaic plan: keeping %d candidate(s) separate; no plausible same-scene/date groups found.", len(candidates), level="summary")
        return
    _deploy_log("Mosaic plan", level="summary")
    for key, group in sorted(groups.items()):
        compatible, reason, paths = _mosaic_group_readiness(group)
        names = ", ".join(candidate.candidate_id for candidate in group)
        if not compatible:
            _deploy_log("  %s: keep separate (%s) | candidates: %s", key, reason, names, level="summary")
            continue
        has_explicit_or_date = any(candidate.mosaic_group or candidate.date for candidate in group)
        if mode == "off" or mode == "plan":
            _deploy_log("  %s: compatible, but keeping separate because mosaic_mode=%s | candidates: %s", key, mode, names, level="summary")
        elif mode == "smart" and evaluating:
            _deploy_log("  %s: compatible; smart mode will mosaic SAR + matching masks if mask pairing is safe | candidates: %s", key, names, level="summary")
        elif mode == "smart":
            _deploy_log("  %s: compatible; smart mode will mosaic | candidates: %s", key, names, level="summary")
        elif mode == "auto" and evaluating and not has_explicit_or_date:
            _deploy_log("  %s: compatible, but keeping separate during labelled evaluation because the rasters are undated/name-grouped and masks are candidate-specific | candidates: %s", key, names, level="summary")
        elif mode == "auto" and not has_explicit_or_date:
            _deploy_log("  %s: compatible; auto mode will mosaic by shared scene stem | candidates: %s", key, names, level="summary")
        elif mode in {"auto", "force"}:
            _deploy_log("  %s: compatible; will mosaic | candidates: %s", key, names, level="summary")
        else:
            _deploy_log("  %s: compatible, but keeping separate | candidates: %s", key, names, level="summary")


def _candidate_raster_paths_for_mosaic(candidate: SceneCandidate) -> list[Path]:
    if candidate.sar_path and candidate.kind in {"multiband_vv_vh", "mosaic_multiband_vv_vh"}:
        return [Path(candidate.sar_path)]
    return []


def _mosaic_compatibility(paths: Sequence[Path]) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Check whether multiband SAR rasters can be mosaicked safely."""
    if len(paths) < 2:
        return False, "Need at least two rasters to mosaic.", None
    infos = []
    for path in paths:
        try:
            infos.append(_raster_info(Path(path)))
        except Exception as exc:
            return False, f"Could not read {path}: {exc}", None
    first = infos[0]
    for path, info in zip(paths[1:], infos[1:]):
        if info.get("count") != first.get("count") or int(info.get("count") or 0) < 2:
            return False, "All SAR rasters must have the same band count and at least VV/VH bands.", None
        if info.get("crs") != first.get("crs"):
            return False, "SAR rasters have different CRS.", None
        res_a = tuple(float(v) for v in first.get("res") or ())
        res_b = tuple(float(v) for v in info.get("res") or ())
        if len(res_a) != 2 or len(res_b) != 2 or not np.allclose(res_a, res_b, rtol=1e-6, atol=1e-9):
            return False, "SAR rasters have different pixel sizes.", None
    # Require bounds to be overlapping or plausibly adjacent. This avoids merging
    # unrelated rasters that merely share CRS/resolution.
    bounds = [info.get("bounds") for info in infos]
    if not all(bounds):
        return False, "Cannot verify bounds for all rasters.", None
    any_touch_or_overlap = False
    res_x, res_y = (float(first["res"][0]), float(first["res"][1]))
    tol_x = max(res_x * 2.0, 1e-9)
    tol_y = max(abs(res_y) * 2.0, 1e-9)
    for i in range(len(bounds)):
        a = tuple(float(v) for v in bounds[i])
        for j in range(i + 1, len(bounds)):
            b = tuple(float(v) for v in bounds[j])
            overlap = _bounds_overlap_ratio(a, b)
            horizontal_touch = abs(a[2] - b[0]) <= tol_x or abs(b[2] - a[0]) <= tol_x
            vertical_overlap = min(a[3], b[3]) > max(a[1], b[1])
            vertical_touch = abs(a[3] - b[1]) <= tol_y or abs(b[3] - a[1]) <= tol_y
            horizontal_overlap = min(a[2], b[2]) > max(a[0], b[0])
            if overlap > 0 or (horizontal_touch and vertical_overlap) or (vertical_touch and horizontal_overlap):
                any_touch_or_overlap = True
                break
        if any_touch_or_overlap:
            break
    if not any_touch_or_overlap:
        return False, "SAR rasters are neither overlapping nor adjacent within a two-pixel tolerance.", None
    return True, "compatible", first


def _mosaic_masks_to_profile(mask_paths: Sequence[Path],
                             output_path: Path,
                             reference_profile: dict[str, Any]) -> Path:
    """Mosaic ground-truth masks onto an already mosaicked SAR grid.

    The output keeps 255 as ignore/no-data outside the supplied mask footprints.
    Within footprints, value 1 remains flood and other valid values remain as
    supplied so existing MMFlood 0/1/2/255 semantics are preserved for the later
    evaluation reader.
    """
    height = int(reference_profile["height"])
    width = int(reference_profile["width"])
    dst_transform = reference_profile["transform"]
    dst_crs = reference_profile.get("crs")
    combined = np.full((height, width), 255, dtype=np.uint8)
    for mask_path in mask_paths:
        with rasterio.open(str(mask_path)) as src:
            tmp = np.full((height, width), 255, dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=255,
                resampling=Resampling.nearest,
            )
            valid = tmp != 255
            empty = combined == 255
            combined[valid & empty] = tmp[valid & empty]
            overlap = valid & ~empty
            both = overlap
            flood_any = both & ((combined == 1) | (tmp == 1))
            combined[flood_any] = 1
            remaining = both & ~flood_any
            combined[remaining] = tmp[remaining]
    profile = dict(reference_profile)
    for invalid_key in ("blockxsize", "blockysize", "tiled", "bounds", "res"):
        profile.pop(invalid_key, None)
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(combined, 1)
    return output_path


def _resolve_group_masks(group: Sequence[SceneCandidate],
                         mask_path: Optional[Path],
                         mask_dir: Optional[Path]) -> list[Path]:
    """Resolve one ground-truth mask per SAR candidate for safe mask mosaicking."""
    if mask_path:
        return [Path(mask_path)]
    if not mask_dir:
        return []
    masks: list[Path] = []
    seen: set[str] = set()
    for candidate in group:
        match = _find_mask(None, Path(mask_dir), candidate)
        if not match:
            return []
        key = str(Path(match).resolve())
        if key in seen:
            continue
        seen.add(key)
        masks.append(Path(match))
    return masks


def _mosaic_multiband_candidates(candidates: Sequence[SceneCandidate],
                                 output_dir: Path,
                                 allow_undated: bool = False,
                                 use_name_group: bool = False,
                                 evaluating: bool = False,
                                 mask_path: Optional[Path] = None,
                                 mask_dir: Optional[Path] = None,
                                 require_mask_mosaic: bool = False) -> list[SceneCandidate]:
    """Mosaic compatible selected SAR tiles into one candidate per group.

    When labelled evaluation is active, this function mosaics the matching mask
    tiles onto the same SAR mosaic grid.  If mask pairing cannot be made safe,
    the SAR group is kept separate rather than producing misleading metrics.
    """
    grouped: dict[str, list[SceneCandidate]] = {}
    ungrouped: list[SceneCandidate] = []
    for candidate in candidates:
        key = _candidate_group_key(candidate, allow_undated=allow_undated, use_name_group=use_name_group)
        if not key:
            ungrouped.append(candidate)
            continue
        grouped.setdefault(key, []).append(candidate)
    output: list[SceneCandidate] = list(ungrouped)
    mosaic_root = Path(output_dir) / "_mosaics"
    for key, group in sorted(grouped.items()):
        if len(group) < 2:
            output.extend(group)
            continue
        paths: list[Path] = []
        for candidate in group:
            paths.extend(_candidate_raster_paths_for_mosaic(candidate))
        if len(paths) != len(group):
            _deploy_log("Skipping mosaic group %s: only multiband VV/VH candidates can be mosaicked automatically.", key, level="summary")
            output.extend(group)
            continue
        compatible, reason, first_info = _mosaic_compatibility(paths)
        if not compatible or first_info is None:
            _deploy_log("Skipping mosaic group %s: %s", key, reason, level="summary")
            output.extend(group)
            continue

        mask_paths: list[Path] = []
        if evaluating:
            mask_paths = _resolve_group_masks(group, Path(mask_path) if mask_path else None, Path(mask_dir) if mask_dir else None)
            if not mask_paths and require_mask_mosaic:
                _deploy_log("Skipping mosaic group %s: labelled evaluation requested but matching masks could not be resolved for every candidate.", key, level="summary")
                output.extend(group)
                continue

        try:
            from rasterio.merge import merge
            srcs = [rasterio.open(str(path)) for path in paths]
            try:
                mosaic, transform = merge(srcs, indexes=[1, 2])
                profile = srcs[0].profile.copy()
            finally:
                for src in srcs:
                    src.close()
            profile.update(height=int(mosaic.shape[1]), width=int(mosaic.shape[2]), transform=transform, count=2)
            mosaic_path = mosaic_root / f"{_safe_name(key)}_mosaic.tif"
            mosaic_root.mkdir(parents=True, exist_ok=True)
            out_profile = profile.copy()
            for invalid_key in ("blockxsize", "blockysize", "tiled", "bounds", "res"):
                out_profile.pop(invalid_key, None)
            out_profile.update(driver="GTiff", count=2, dtype=str(mosaic.dtype), compress="deflate")
            with rasterio.open(str(mosaic_path), "w", **out_profile) as dst:
                dst.write(mosaic.astype(mosaic.dtype))
            with rasterio.open(str(mosaic_path)) as src:
                info = {
                    "crs": str(src.crs) if src.crs else "",
                    "height": int(src.height),
                    "width": int(src.width),
                    "count": int(src.count),
                    "bounds": tuple(float(v) for v in src.bounds),
                    "res": tuple(float(v) for v in src.res),
                }
                reference_profile = src.profile.copy()
            candidate_id = _safe_name(f"{key}_mosaic")
            mosaic_mask_path: Optional[Path] = None
            if evaluating and mask_paths:
                if len(mask_paths) == 1 and mask_path:
                    mosaic_mask_path = Path(mask_paths[0])
                    _deploy_log("Using scene-level ground-truth mask for mosaicked group %s: %s", key, mosaic_mask_path.name, level="summary")
                else:
                    mosaic_mask_path = mosaic_root / f"{_safe_name(key)}_mask_mosaic.tif"
                    _mosaic_masks_to_profile(mask_paths, mosaic_mask_path, reference_profile)
                    _deploy_log("Mosaicked %d matching ground-truth mask tile(s) for group %s: %s", len(mask_paths), key, mosaic_mask_path, level="summary")
            date_values = {candidate.date for candidate in group if candidate.date}
            output.append(SceneCandidate(candidate_id=candidate_id,
                                         kind="mosaic_multiband_vv_vh",
                                         date=next(iter(date_values)) if len(date_values) == 1 else None,
                                         sar_path=mosaic_path,
                                         status="ready",
                                         reason=f"Mosaicked {len(group)} compatible SAR tiles.",
                                         crs=info["crs"],
                                         height=info["height"],
                                         width=info["width"],
                                         count=info["count"],
                                         bounds=info["bounds"],
                                         res=info["res"],
                                         dem_path=group[0].dem_path,
                                         mask_path=mosaic_mask_path,
                                         mosaic_group=key))
            _deploy_log("Mosaicked %d compatible SAR tile(s) into candidate %s: %s", len(group), candidate_id, mosaic_path, level="summary")
        except Exception as exc:
            _deploy_log("Skipping mosaic group %s because mosaicking failed: %s", key, exc, level="summary")
            output.extend(group)
    return sorted(output, key=_candidate_sort_key)


def _find_dem(scene_dir: Optional[Path], dem_path: Optional[Path], dem_dir: Optional[Path] = None) -> Optional[Path]:
    if dem_path:
        return Path(dem_path)
    roots = [Path(root) for root in [dem_dir, scene_dir] if root]
    for root in roots:
        matches = sorted(path for path in root.rglob("*") if path.suffix.lower() in RASTER_SUFFIXES and _looks_like_dem(path))
        if matches:
            _deploy_log("Using discovered DEM: %s", matches[0])
            return matches[0]
    return None


def _find_mask(mask_path: Optional[Path], mask_dir: Optional[Path], candidate: Optional[SceneCandidate] = None) -> Optional[Path]:
    """Resolve an optional ground-truth mask for labelled deployment evaluation."""
    if mask_path:
        return Path(mask_path)
    if not mask_dir:
        return None
    root = Path(mask_dir)
    if not root.exists():
        return None
    matches = sorted(path for path in root.rglob("*") if path.suffix.lower() in RASTER_SUFFIXES and _looks_like_mask(path))
    if not matches:
        return None
    if candidate and candidate.mask_path:
        return Path(candidate.mask_path)
    if candidate:
        candidate_tokens = [candidate.candidate_id.lower()]
        if candidate.date:
            candidate_tokens.extend([candidate.date.lower(), candidate.date.replace("-", "")])
        source_stems = [path.stem.lower() for path in [candidate.sar_path, candidate.vv_path, candidate.vh_path] if path]
        candidate_tokens.extend(source_stems)
        for path in matches:
            lower = path.as_posix().lower()
            if any(token and token in lower for token in candidate_tokens):
                return path
    return matches[0]


def _read_mask_for_evaluation(mask_path: Path, profile: dict, shape: tuple[int, int]) -> np.ndarray:
    """Read a ground-truth mask on the SAR grid using conservative MMFlood semantics.

    Deployment evaluation treats value 1 as flood, values 0/2 as background and
    value 255 as ignore/no-data. If the mask grid differs from the SAR grid, the
    mask is reprojected/resampled with nearest-neighbour interpolation so the
    metrics are calculated on the prediction grid.
    """
    height, width = shape
    sar_transform = profile.get("transform")
    sar_crs = profile.get("crs")
    with rasterio.open(str(mask_path)) as src:
        same_grid = bool(src.height == height and src.width == width and src.transform == sar_transform and src.crs == sar_crs)
        if same_grid:
            raw = src.read(1)
        elif sar_transform is not None and sar_crs is not None and src.crs is not None:
            LOG.warning("Ground-truth mask grid differs from SAR grid; reprojecting/resampling mask to SAR grid with nearest-neighbour interpolation.")
            raw = np.full((height, width), 255, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=raw,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=sar_transform,
                dst_crs=sar_crs,
                dst_nodata=255,
                resampling=Resampling.nearest,
            )
        else:
            raise ValueError(
                "Ground-truth mask grid differs from the SAR grid, but CRS/transform metadata is incomplete. "
                "Evaluation was stopped rather than resizing the mask by array shape alone."
            )
    gt = np.zeros((height, width), dtype=np.uint8)
    gt[np.isclose(raw, 1)] = 1
    gt[np.isclose(raw, 255)] = 255
    return gt

def _compute_binary_metrics(pred: np.ndarray, target: np.ndarray, ignore_value: int = 255) -> dict[str, Any]:
    valid = target != ignore_value
    pred_bin = pred.astype(np.uint8) > 0
    target_bin = target.astype(np.uint8) > 0
    tp = int(np.logical_and(pred_bin, target_bin & valid).sum())
    tn = int(np.logical_and(~pred_bin, (~target_bin) & valid).sum())
    fp = int(np.logical_and(pred_bin, (~target_bin) & valid).sum())
    fn = int(np.logical_and(~pred_bin, target_bin & valid).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / (denom + 1e-8) if denom > 0 else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "valid_pixels": int(valid.sum()),
        "ignored_pixels": int((~valid).sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "mcc": float(mcc),
    }


def _save_confusion_matrix_png(path: Path, metrics: dict[str, Any]) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(matrix)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("Confusion matrix")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{int(value):,}", ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_error_overlay(path: Path, base: np.ndarray, pred: np.ndarray, target: np.ndarray) -> Path:
    base_norm = _normalise_for_display(base)
    rgb = np.dstack([base_norm, base_norm, base_norm])
    valid = target != 255
    tp = (pred == 1) & (target == 1) & valid
    fp = (pred == 1) & (target == 0) & valid
    fn = (pred == 0) & (target == 1) & valid
    rgb[tp] = [0.0, 0.85, 0.0]
    rgb[fp] = [1.0, 0.15, 0.15]
    rgb[fn] = [0.1, 0.3, 1.0]
    return _save_rgb(path, rgb)


def _pixel_area_km2(profile: dict) -> Optional[float]:
    transform = profile.get("transform")
    if transform is None:
        return None
    try:
        px_w = abs(float(transform.a if hasattr(transform, "a") else transform[0]))
        px_h = abs(float(transform.e if hasattr(transform, "e") else transform[4]))
        crs = profile.get("crs")
        if crs is not None and getattr(crs, "is_geographic", False):
            # Approximate at scene centre. Good enough for console summary only.
            height = int(profile.get("height") or 0)
            centre_y = float((transform * (0, height / 2))[1]) if hasattr(transform, "__mul__") else 0.0
            metres_per_degree_lat = 111_320.0
            metres_per_degree_lon = metres_per_degree_lat * max(0.01, np.cos(np.deg2rad(centre_y)))
            return (px_w * metres_per_degree_lon * px_h * metres_per_degree_lat) / 1_000_000.0
        return (px_w * px_h) / 1_000_000.0
    except Exception:
        return None


def write_deployment_manifest(output_file: Path,
                              configs: Sequence[Path],
                              checkpoints: Sequence[Path],
                              model_name: str = "flood_model",
                              ensemble_method: Optional[str] = None,
                              threshold: float = 0.5,
                              min_component_area: int = 0,
                              input_modalities: Optional[Iterable[str]] = None,
                              normalization_stats_path: Optional[str] = None,
                              normalization_mode: Optional[str] = None,
                              inference_mode: str = "sliding_window",
                              window_size: int = 512,
                              window_overlap: int = 128,
                              window_batch_size: int = 1,
                              window_blend: str = "uniform",
                              notes: Optional[str] = None,
                              copy_assets: bool = True,
                              assets_directory: str = "assets") -> Path:
    """Write a deployment manifest and, by default, a portable asset bundle.

    Portable manifests contain only paths relative to the manifest itself.  The
    corresponding configuration files, checkpoints and normalization statistics
    are copied into the deployment directory.  ``copy_assets=False`` retains the
    older reference-only behaviour for advanced workflows that intentionally keep
    model assets elsewhere.
    """
    if len(configs) != len(checkpoints):
        raise ValueError("configs and checkpoints must have the same length")
    if not configs:
        raise ValueError("At least one config/checkpoint pair is required")

    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    config_sources = [Path(value).expanduser().resolve() for value in configs]
    checkpoint_sources = [Path(value).expanduser().resolve() for value in checkpoints]
    for source in config_sources:
        if not source.is_file():
            raise FileNotFoundError(f"Deployment config not found: {source}")
    for source in checkpoint_sources:
        if not source.is_file():
            raise FileNotFoundError(f"Deployment checkpoint not found: {source}")

    configured_stats: list[Path] = []
    configured_modes: list[str] = []
    for config_source in config_sources:
        stats_path, mode = _configured_normalization_stats(config_source)
        if stats_path is not None:
            configured_stats.append(stats_path)
        if mode:
            configured_modes.append(mode)

    resolved_mode = str(normalization_mode or (configured_modes[0] if configured_modes else "robust_percentile"))
    stats_source: Optional[Path] = None
    if normalization_stats_path:
        candidate = Path(str(normalization_stats_path)).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        stats_source = candidate.resolve()
    elif configured_stats:
        existing = [path for path in configured_stats if path.is_file()]
        missing = [path for path in configured_stats if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Normalization statistics referenced by the training config were not found: "
                + ", ".join(str(path) for path in missing)
            )
        unique_by_hash: dict[str, Path] = {}
        for path in existing:
            unique_by_hash.setdefault(_sha256_file(path), path)
        if len(unique_by_hash) > 1:
            raise ValueError(
                "Ensemble configs reference different normalization-statistics files. "
                "Pass --normalization-stats-path explicitly to select the deployment asset."
            )
        stats_source = next(iter(unique_by_hash.values()))

    stats_required_modes = {"stats", "robust_percentile", "notebook_robust", "robust_minmax"}
    if resolved_mode.strip().lower().replace("-", "_") in stats_required_modes and stats_source is None:
        raise ValueError(
            f"normalization_mode='{resolved_mode}' requires normalization statistics, but none were supplied "
            "and none could be resolved from the training config."
        )
    if stats_source is not None and not stats_source.is_file():
        raise FileNotFoundError(f"Deployment normalization statistics not found: {stats_source}")

    mode = "ensemble" if len(config_sources) > 1 else "single"
    inventory_entries: list[dict[str, Any]] = []
    member_rows: list[dict[str, str]] = []
    bundled_stats_reference: Optional[str] = None

    if copy_assets:
        assets_path = Path(str(assets_directory))
        if assets_path.is_absolute() or ".." in assets_path.parts:
            raise ValueError("assets_directory must be a relative path contained inside the deployment bundle")
        assets_root = output_file.parent / assets_path
        stats_destination: Optional[Path] = None
        if stats_source is not None:
            stats_suffix = stats_source.suffix or ".json"
            stats_destination = assets_root / "normalization" / f"normalization_stats{stats_suffix}"
            stats_record = _write_portable_normalization_stats(stats_source, stats_destination)
            inventory_entries.append(stats_record)
            bundled_stats_reference = _manifest_relative(stats_destination, output_file)

        for index, (config_source, checkpoint_source) in enumerate(zip(config_sources, checkpoint_sources), start=1):
            config_suffix = config_source.suffix if config_source.suffix.lower() in {".yaml", ".yml", ".json"} else ".yaml"
            checkpoint_suffix = checkpoint_source.suffix or ".pth"
            config_destination = assets_root / "configs" / f"member_{index:02d}_config{config_suffix}"
            checkpoint_destination = assets_root / "checkpoints" / f"member_{index:02d}_checkpoint{checkpoint_suffix}"
            config_record = _write_portable_config(
                config_source,
                config_destination,
                normalization_destination=stats_destination,
                role=f"member_{index:02d}_config",
            )
            checkpoint_record = _copy_bundle_file(checkpoint_source, checkpoint_destination, role=f"member_{index:02d}_checkpoint")
            inventory_entries.extend([config_record, checkpoint_record])
            member_rows.append({
                "config": _manifest_relative(config_destination, output_file),
                "checkpoint": _manifest_relative(checkpoint_destination, output_file),
            })
    else:
        member_rows = [
            {"config": str(config_source), "checkpoint": str(checkpoint_source)}
            for config_source, checkpoint_source in zip(config_sources, checkpoint_sources)
        ]
        bundled_stats_reference = str(stats_source) if stats_source is not None else None

    manifest = {
        "schema_version": 2,
        "model_name": model_name,
        "mode": mode,
        "bundle": {
            "portable": bool(copy_assets),
            "path_base": "manifest_directory" if copy_assets else "absolute_or_working_environment",
            "inventory": "deployment_bundle.json" if copy_assets else None,
        },
        "members": member_rows,
        "ensemble_method": ensemble_method if mode == "ensemble" else None,
        "operating_point": {
            "threshold": float(threshold),
            "min_component_area": int(min_component_area),
        },
        "inputs": {
            "modalities": list(input_modalities or ["vv", "vh", "dem"]),
            "normalization_mode": resolved_mode,
            "normalization_stats_path": bundled_stats_reference,
            "sar_contract": "Analysis-ready GeoTIFF. Either one file with VV band 1 and VH band 2, or paired VV/VH files discovered from a scene folder.",
            "dem_contract": "Single-band DEM GeoTIFF aligned to SAR or resampleable to the SAR grid.",
        },
        "inference": {
            "mode": inference_mode,
            "window_size": int(window_size),
            "window_overlap": int(window_overlap),
            "window_batch_size": int(window_batch_size),
            "window_blend": str(window_blend),
        },
        "visual_outputs": {
            "previews": ["vv", "vh", "dem", "false_colour"],
            "prediction": ["probability_heatmap", "binary_mask_overlay"],
            "explanation": ["positive_evidence", "ensemble_uncertainty", "optional_per_modality_occlusion"],
        },
        "outputs": {
            "mask_values": {"0": "background", "1": "flood"},
            "probability": "Optional float32 GeoTIFF containing flood probability before thresholding.",
            "report": "Optional HTML report with input previews, prediction overlays, uncertainty and explanation panels.",
        },
        "notes": notes or "Use predict-scene with analysis-ready inputs that match the preprocessing and normalization settings used during validation.",
    }

    if output_file.suffix.lower() in {".yaml", ".yml"}:
        output_file.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    else:
        output_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if copy_assets:
        inventory_path = output_file.parent / "deployment_bundle.json"
        inventory_payload = {
            "schema_version": 1,
            "manifest": output_file.name,
            "model_name": model_name,
            "portable": True,
            "files": [
                {
                    "role": item["role"],
                    "path": _manifest_relative(item["path"], output_file),
                    "size_bytes": int(item["size_bytes"]),
                    "sha256": item["sha256"],
                }
                for item in inventory_entries
            ],
        }
        inventory_path.write_text(json.dumps(inventory_payload, indent=2), encoding="utf-8")
        _write_bundle_readme(output_file.parent / "DEPLOYMENT_README.md", output_file.name)
        total_bytes = sum(int(item["size_bytes"]) for item in inventory_entries)
        _deploy_log(
            "Portable deployment bundle written: %s | assets=%d | size=%.1f MiB",
            output_file.parent,
            len(inventory_entries),
            total_bytes / (1024 * 1024),
            level="summary",
        )
    else:
        _deploy_log("Reference-only deployment manifest written to: %s", output_file, level="summary")
    return output_file


def _read_band_to_reference_grid(
    path: Path,
    reference_profile: dict,
    height: int,
    width: int,
    *,
    resampling: Resampling,
    label: str,
) -> np.ndarray:
    """Read one raster band on a reference geospatial grid.

    When grids differ, this performs CRS-aware reprojection rather than an
    array-shape resize. Missing georeferencing is treated as an error because
    silently resizing geospatial rasters can pair the wrong ground locations.
    """
    path = Path(path)
    ref_crs = reference_profile.get("crs")
    ref_transform = reference_profile.get("transform")
    with rasterio.open(str(path)) as src:
        same_grid = (
            src.height == height
            and src.width == width
            and src.transform == ref_transform
            and src.crs == ref_crs
        )
        if same_grid:
            return src.read(1).astype(np.float32)
        if src.crs is None or ref_crs is None or ref_transform is None:
            raise ValueError(
                f"{label} grid differs from the SAR reference grid, but one raster lacks CRS/transform metadata; "
                "safe geospatial reprojection is not possible."
            )
        _deploy_log("%s grid differs from SAR reference; reprojecting onto the SAR grid.", label, level="standard")
        values = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=values,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return values


def _read_sar_arrays(candidate: SceneCandidate | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    if isinstance(candidate, Path):
        candidate = SceneCandidate(candidate_id=_safe_name(candidate.stem), kind="multiband_vv_vh", date=_extract_date(candidate), sar_path=candidate)
    if candidate.sar_path:
        with rasterio.open(str(candidate.sar_path)) as src:
            if src.count < 2:
                raise ValueError(f"SAR raster must have at least two bands. Got {src.count}: {candidate.sar_path}")
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)
            profile = src.profile.copy()
            profile["bounds"] = tuple(float(v) for v in src.bounds)
        return vv, vh, profile
    if not candidate.vv_path or not candidate.vh_path:
        raise ValueError(f"Candidate {candidate.candidate_id} is not a complete VV/VH pair")
    with rasterio.open(str(candidate.vv_path)) as vv_src:
        vv = vv_src.read(1).astype(np.float32)
        profile = vv_src.profile.copy()
        profile["bounds"] = tuple(float(v) for v in vv_src.bounds)
        height, width = vv_src.height, vv_src.width
    vh = _read_band_to_reference_grid(
        Path(candidate.vh_path),
        profile,
        height,
        width,
        resampling=Resampling.bilinear,
        label=f"VH ({candidate.candidate_id})",
    )
    return vv, vh, profile


def _build_tensor_from_arrays(vv: np.ndarray,
                              vh: np.ndarray,
                              dem_path: Optional[Path],
                              profile: dict,
                              modalities: Sequence[str],
                              normalization_mode: str,
                              normalization_stats_path: Path,
                              device: torch.device) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    from floods.modalities import canonicalize_modalities

    modalities = canonicalize_modalities(modalities)
    height, width = vv.shape
    raw_channels: dict[str, np.ndarray] = {"vv": vv.astype(np.float32), "vh": vh.astype(np.float32)}
    sar_crs = profile.get("crs")
    sar_transform = profile.get("transform")

    needs_dem = any(name in {"dem", "dem_slope", "dem_tpi"} for name in modalities)
    dem = None
    if needs_dem:
        if dem_path is None:
            raise ValueError("DEM-derived modality requested but no DEM was provided or discovered")
        dem = _read_band_to_reference_grid(
            Path(dem_path),
            profile,
            height,
            width,
            resampling=Resampling.bilinear,
            label="DEM",
        )
        raw_channels["dem"] = dem

    requested_derived = [
        name for name in modalities if name in {"vv_vh_log_ratio", "dem_slope", "dem_tpi"}
    ]
    if requested_derived:
        from floods.derived_features import derive_feature_channels, pixel_spacing_meters

        transform = profile.get("transform")
        if "dem_slope" in requested_derived:
            pixel_size_x, pixel_size_y = pixel_spacing_meters(
                transform,
                sar_crs,
                width,
                height,
            )
        else:
            pixel_size_x = pixel_size_y = 1.0
        raw_channels.update(
            derive_feature_channels(
                np.stack([vv, vh], axis=0),
                dem,
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                modalities=requested_derived,
            )
        )

    channels = [raw_channels[name] for name in modalities]
    image = np.stack(channels, axis=-1).astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=30.0, neginf=-30.0)
    from floods.normalization import describe_stats, load_normalization_stats
    from floods.inference_transforms import eval_transforms

    mean, std, clip_min, clip_max = load_normalization_stats(Path(normalization_stats_path), modalities, mode=normalization_mode)
    _deploy_log("Using deployment normalization stats (%s): %s", normalization_mode, describe_stats(Path(normalization_stats_path)), level="verbose")
    transform = eval_transforms(mean=mean, std=std, clip_min=clip_min, clip_max=clip_max, normalization_mode=normalization_mode)
    dummy_mask = np.zeros((height, width), dtype=np.uint8)
    transformed = transform(image=image, mask=dummy_mask)
    tensor = transformed["image"].unsqueeze(0).to(device)
    return tensor, raw_channels


def _build_scene_tensor(sar_path: Path,
                        dem_path: Optional[Path],
                        modalities: Sequence[str],
                        normalization_mode: str,
                        normalization_stats_path: Path,
                        device: torch.device) -> tuple[torch.Tensor, dict, dict[str, np.ndarray]]:
    vv, vh, profile = _read_sar_arrays(Path(sar_path))
    tensor, raw_channels = _build_tensor_from_arrays(vv, vh, dem_path, profile, modalities, normalization_mode, normalization_stats_path, device)
    return tensor, profile, raw_channels


def _load_models(configs: Sequence[TrainConfig], checkpoints: Sequence[Path], device: torch.device) -> list[torch.nn.Module]:
    models = []
    from floods.evaluation import load_checkpoint_state
    from floods.model_factory import prepare_model

    for index, (cfg, ckpt) in enumerate(zip(configs, checkpoints), start=1):
        # Deployment checkpoints already contain trained weights. Building the
        # architecture with pretrained=False avoids fragile Hugging Face downloads
        # and makes deployment deterministic/offline-friendly.
        cfg.model.pretrained = False
        _deploy_log("Loading model %d/%d without downloading pretrained encoder weights: decoder=%s encoder=%s", index, len(configs), cfg.model.decoder, cfg.model.encoder)
        model = prepare_model(config=cfg, num_classes=1, stage="eval")
        state = load_checkpoint_state(Path(ckpt))
        model.load_state_dict(state, strict=not cfg.model.multibranch)
        model = model.to(device)
        model.eval()
        models.append(model)
        _deploy_log("Loaded deployment model %d/%d from checkpoint: %s", index, len(configs), ckpt)
    return models


def _apply_component_filter(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    min_component_area = int(min_component_area or 0)
    if min_component_area <= 0:
        return mask.astype(np.uint8, copy=False)
    binary = mask.astype(np.uint8, copy=False)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            keep[labels == label_idx] = 1
    return keep


def _deployment_raster_profile(profile: dict, dtype: str, nodata: Optional[float | int]) -> dict:
    """Create a safe single-band GeoTIFF profile for deployment outputs.

    Rasterio ``profile`` dictionaries can be convenient containers for both real
    creation options and read-only metadata that we attach during deployment.
    GDAL is strict when opening a GeoTIFF for writing: informational keys such as
    ``bounds`` are forwarded as creation options and trigger warnings like
    ``driver GTiff does not support creation option BOUNDS``.  Keep only fields
    that are valid for writing a simple georeferenced single-band GeoTIFF.
    """
    out_profile = profile.copy()
    out_profile["driver"] = "GTiff"
    # Source rasters may carry block size metadata without tiled=True. GDAL rejects
    # BLOCKXSIZE/BLOCKYSIZE unless tiled output is explicitly enabled.  We also
    # remove read-only metadata that Rasterio/GDAL must not receive as GTiff
    # creation options.
    for key in (
        "blockxsize",
        "blockysize",
        "tiled",
        "bounds",
        "res",
    ):
        out_profile.pop(key, None)
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    return out_profile


def _write_mask(path: Path, profile: dict, mask: np.ndarray) -> None:
    # Do not set nodata=0 because 0 is the valid non-flood class.
    out_profile = _deployment_raster_profile(profile, dtype="uint8", nodata=None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", **out_profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def _write_float(path: Path, profile: dict, values: np.ndarray, nodata: Optional[float] = None) -> None:
    out_profile = _deployment_raster_profile(profile, dtype="float32", nodata=nodata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", **out_profile) as dst:
        dst.write(values.astype(np.float32), 1)


def _normalise_for_display(array: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _save_png(path: Path, array: np.ndarray, cmap: str = "gray", vmin: float = 0.0, vmax: float = 1.0) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), array, cmap=cmap, vmin=vmin, vmax=vmax)
    return path


def _save_rgb(path: Path, rgb: np.ndarray) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), np.clip(rgb, 0.0, 1.0))
    return path


def _overlay_heatmap(base: np.ndarray, values: np.ndarray, cmap: str = "magma", alpha: float = 0.55) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    base_norm = _normalise_for_display(base)
    base_rgb = np.dstack([base_norm, base_norm, base_norm])
    cmap_obj = plt.get_cmap(cmap)
    heat = cmap_obj(np.clip(values, 0.0, 1.0))[..., :3]
    weight = (np.clip(values, 0.0, 1.0) * alpha)[..., None]
    return base_rgb * (1.0 - weight) + heat * weight


def _overlay_mask(base: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base_norm = _normalise_for_display(base)
    rgb = np.dstack([base_norm, base_norm, base_norm])
    flood = mask.astype(bool)
    rgb[flood, 0] = rgb[flood, 0] * (1.0 - alpha) + alpha
    rgb[flood, 1] = rgb[flood, 1] * (1.0 - alpha)
    rgb[flood, 2] = rgb[flood, 2] * (1.0 - alpha)
    return rgb


def _save_top_crop(path: Path, base: np.ndarray, probability: np.ndarray, size: int = 512) -> Optional[Path]:
    if probability.size == 0:
        return None
    y, x = np.unravel_index(int(np.nanargmax(probability)), probability.shape)
    half = max(32, int(size) // 2)
    y0 = max(0, y - half)
    x0 = max(0, x - half)
    y1 = min(probability.shape[0], y0 + size)
    x1 = min(probability.shape[1], x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    crop = _overlay_heatmap(base[y0:y1, x0:x1], probability[y0:y1, x0:x1], cmap="magma", alpha=0.65)
    return _save_rgb(path, crop)


def _image_to_data_uri(path: Path) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _write_visual_report(report_path: Path,
                         title: str,
                         metadata: dict[str, Any],
                         image_paths: Sequence[tuple[str, Path]],
                         inventory_path: Optional[Path] = None) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for label, path in image_paths:
        try:
            src = _image_to_data_uri(path)
        except Exception:
            src = html.escape(path.as_posix())
        cards.append(f"<figure><img src='{src}' alt='{html.escape(label)}'><figcaption>{html.escape(label)}</figcaption></figure>")
    inventory_link = ""
    if inventory_path and inventory_path.exists():
        rel_inv = html.escape(inventory_path.relative_to(report_path.parent).as_posix() if inventory_path.is_relative_to(report_path.parent) else inventory_path.as_posix())
        inventory_link = f"<p><a href='{rel_inv}'>Scene inventory CSV</a></p>"
    flood_pixels = int(metadata.get("flood_pixels", 0))
    flood_fraction = metadata.get("flood_fraction")
    flood_area = metadata.get("flood_area_km2")
    threshold = metadata.get("threshold")
    output_mask = html.escape(str(metadata.get("output_mask") or ""))
    output_probability = html.escape(str(metadata.get("output_probability") or "Not written"))
    eval_metrics = metadata.get("evaluation_metrics") or None
    mask_alignment = metadata.get("mask_alignment") or None
    eval_html = ""
    if eval_metrics:
        eval_rows = [
            ("F1", f"{float(eval_metrics.get('f1', 0.0)):.4f}"),
            ("IoU", f"{float(eval_metrics.get('iou', 0.0)):.4f}"),
            ("Precision", f"{float(eval_metrics.get('precision', 0.0)):.4f}"),
            ("Recall", f"{float(eval_metrics.get('recall', 0.0)):.4f}"),
            ("MCC", f"{float(eval_metrics.get('mcc', 0.0)):.4f}"),
            ("Ground-truth flood pixels", f"{int(eval_metrics.get('tp', 0)) + int(eval_metrics.get('fn', 0)):,}"),
            ("Confusion matrix", (
                f"TP={int(eval_metrics.get('tp', 0)):,}<br>"
                f"TN={int(eval_metrics.get('tn', 0)):,}<br>"
                f"FP={int(eval_metrics.get('fp', 0)):,}<br>"
                f"FN={int(eval_metrics.get('fn', 0)):,}"
            )),
        ]
        if mask_alignment:
            eval_rows.extend([
                ("Mask file", f"<code>{html.escape(str(mask_alignment.get('mask_path') or ''))}</code>"),
                ("Mask CRS match", "yes" if mask_alignment.get("crs_match") else "no"),
                ("Bounds overlap", f"{float(mask_alignment.get('bounds_overlap_ratio') or 0.0):.3f}"),
                ("Mask resampled to SAR grid", "yes" if mask_alignment.get("resampled_to_sar_grid") else "no"),
                ("SAR shape", html.escape(str(mask_alignment.get("sar_shape") or ""))),
                ("Mask shape", html.escape(str(mask_alignment.get("mask_shape") or ""))),
            ])
        eval_table = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in eval_rows)
        eval_html = (
            "<section class='summary'><h2>Labelled-scene evaluation and mask alignment</h2>"
            "<table aria-label='Labelled scene evaluation metrics and mask alignment'><tbody>"
            f"{eval_table}"
            "</tbody></table>"
            "</section>"
        )
    summary_rows = [
        ("Threshold", html.escape(str(threshold))),
        ("Predicted flood pixels", f"{flood_pixels:,}"),
    ]
    if flood_fraction is not None:
        summary_rows.append(("Flood fraction", f"{float(flood_fraction) * 100:.2f}%"))
    if flood_area is not None:
        summary_rows.append(("Approx flood area", f"{float(flood_area):.3f} km²"))
    summary_rows.extend([
        ("Binary mask", f"<code>{output_mask}</code>"),
        ("Probability raster", f"<code>{output_probability}</code>"),
    ])
    summary_table = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in summary_rows)
    metadata_text = html.escape(json.dumps(metadata, indent=2))
    body = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{html.escape(title)}</title>
<style>
.floodmap-deployment-report {{
  --border:#dfe3e8;
  --muted:#5f6368;
  --blue:#1a73e8;
  --blue-bg:#eef6ff;
  box-sizing:border-box;
  font-family:Arial,sans-serif;
  margin:24px;
  color:#202124;
  line-height:1.45;
  background:#fff;
}}
.floodmap-deployment-report *,
.floodmap-deployment-report *::before,
.floodmap-deployment-report *::after {{ box-sizing:border-box; }}
.floodmap-deployment-report h1,
.floodmap-deployment-report h2 {{ line-height:1.2; }}
.floodmap-deployment-report .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px; }}
.floodmap-deployment-report figure {{ margin:0; border:1px solid var(--border); border-radius:8px; padding:10px; background:#fff; min-width:0; }}
.floodmap-deployment-report img {{ width:100%; height:auto; display:block; border-radius:4px; }}
.floodmap-deployment-report figcaption {{ margin-top:8px; font-weight:600; overflow-wrap: anywhere; }}
.floodmap-deployment-report pre {{ background:#f6f8fa; color:#202124; padding:12px; white-space:pre-wrap; overflow-wrap: anywhere; word-break:break-word; border-radius:6px; max-width:100%; }}
.floodmap-deployment-report code {{ color:#202124; overflow-wrap: anywhere; word-break:break-word; white-space:pre-wrap; }}
.floodmap-deployment-report table {{ border-collapse:collapse; width:100%; table-layout:fixed; }}
.floodmap-deployment-report th,
.floodmap-deployment-report td {{ border-bottom:1px solid var(--border); padding:8px 10px; vertical-align:top; text-align:left; }}
.floodmap-deployment-report th {{ width:220px; color:var(--muted); font-weight:700; }}
.floodmap-deployment-report td {{ overflow-wrap: anywhere; word-break:break-word; }}
.floodmap-deployment-report .note {{ background:#fff8e1; border-left:4px solid #f9ab00; padding:10px 12px; border-radius:4px; }}
.floodmap-deployment-report .summary {{ background:var(--blue-bg); border-left:4px solid var(--blue); padding:12px 14px; margin:16px 0; border-radius:4px; overflow-wrap: anywhere; }}
.floodmap-deployment-report .metadata-details summary {{ cursor:pointer; font-weight:700; margin:8px 0; }}
@media (max-width:700px) {{
  .floodmap-deployment-report {{ margin:12px; }}
  .floodmap-deployment-report th,
  .floodmap-deployment-report td {{ display:block; width:100%; }}
  .floodmap-deployment-report th {{ border-bottom:0; padding-bottom:2px; }}
  .floodmap-deployment-report td {{ padding-top:2px; }}
}}
</style>
</head>
<body>
<div class='floodmap-deployment-report' data-floodmap-report='deployment'>
<h1>{html.escape(title)}</h1>
<p class='note'>Explanation panels show model evidence in the prediction process. Probability and uncertainty maps explain model behaviour; they are not physical proof of why flooding occurred.</p>
{inventory_link}
<section class='summary'>
<h2>Final flood prediction</h2>
<table aria-label='Final flood prediction summary'>
<tbody>
{summary_table}
</tbody>
</table>
</section>
{eval_html}
<div class='grid'>
{''.join(cards)}
</div>
<details class='metadata-details'>
<summary>Prediction metadata</summary>
<pre>{metadata_text}</pre>
</details>
</div>
</body>
</html>
"""
    report_path.write_text(body, encoding="utf-8")
    return report_path


def _split_css_selector_list(selector_text: str) -> list[str]:
    """Split a CSS selector list on top-level commas."""
    selectors: list[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(selector_text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            selectors.append(selector_text[start:index])
            start = index + 1
    selectors.append(selector_text[start:])
    return selectors


def _scope_notebook_selector(selector: str, scope: str) -> str:
    selector = selector.strip()
    if not selector:
        return selector
    selector = re.sub(r"(?<![\w-]):root\b", scope, selector)
    selector = re.sub(r"(?<![\w-])(?:html|body)\b", scope, selector)
    selector = re.sub(
        rf"(?:{re.escape(scope)}\s+)+{re.escape(scope)}\b",
        scope,
        selector,
    )
    if scope in selector:
        return selector
    return f"{scope} {selector}"


def _scope_css_for_notebook(css_text: str, scope: str = ".floodmap-notebook-report") -> str:
    """Prefix CSS selectors so an inline report cannot style notebook output."""
    css = str(css_text)
    output: list[str] = []
    cursor = 0
    length = len(css)
    while cursor < length:
        open_brace = css.find("{", cursor)
        if open_brace < 0:
            output.append(css[cursor:])
            break

        prelude_region = css[cursor:open_brace]
        leading_match = re.match(r"\s*", prelude_region)
        leading = leading_match.group(0) if leading_match else ""
        prelude = prelude_region[len(leading):].strip()

        depth = 1
        index = open_brace + 1
        quote: Optional[str] = None
        escaped = False
        while index < length and depth:
            char = css[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif quote:
                if char == quote:
                    quote = None
            elif char in {"'", '"'}:
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            index += 1

        if depth:
            output.append(css[cursor:])
            break

        inner = css[open_brace + 1:index - 1]
        lower = prelude.lower()
        if lower.startswith(("@media", "@supports", "@container", "@layer", "@document")):
            rendered_prelude = prelude
            rendered_inner = _scope_css_for_notebook(inner, scope)
        elif lower.startswith(("@font-face", "@keyframes", "@-webkit-keyframes", "@page", "@property")):
            rendered_prelude = prelude
            rendered_inner = inner
        elif prelude.startswith("@"):
            rendered_prelude = prelude
            rendered_inner = inner
        else:
            rendered_prelude = ",\n".join(
                _scope_notebook_selector(item, scope)
                for item in _split_css_selector_list(prelude)
            )
            rendered_inner = inner

        output.append(f"{leading}{rendered_prelude} {{{rendered_inner}}}")
        cursor = index
    return "".join(output)


def _notebook_safe_report_fragment(report_html: str) -> str:
    """Return a CSS-isolated notebook fragment for current or legacy reports.

    Colab/Jupyter insert :class:`IPython.display.HTML` into the notebook DOM rather
    than an isolated browser document.  Every stylesheet is therefore rewritten
    beneath a dedicated wrapper before display.  This protects earlier console
    output even when an older report contains global selectors such as ``body``,
    ``pre``, ``code`` or ``*``.
    """
    source = str(report_html)
    scoped_styles: list[str] = []
    for match in re.finditer(
        r"<style(?:\s[^>]*)?>(.*?)</style>",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        scoped_styles.append(f"<style>{_scope_css_for_notebook(match.group(1))}</style>")
    body_match = re.search(
        r"<body(?:\s[^>]*)?>(.*?)</body>",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = body_match.group(1) if body_match else re.sub(
        r"<style(?:\s[^>]*)?>.*?</style>",
        "",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    wrapper = f"<div class='floodmap-notebook-report'>{body}</div>"
    return "\n".join([*scoped_styles, wrapper])


def _display_inline(report_path: Path, image_paths: Sequence[tuple[str, Path]]) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except Exception:
        _deploy_log("Inline display requested, but IPython display is not available. Report saved to %s", report_path)
        return
    if get_ipython() is None:
        _deploy_log("Inline display requested from a shell subprocess, so Colab cannot render it directly.")
        _deploy_log("Open the self-contained HTML report or run this in a notebook cell: from floods.deployment import display_deployment_outputs; display_deployment_outputs(r'%s')", report_path.parent.as_posix())
        return
    display(HTML(_notebook_safe_report_fragment(Path(report_path).read_text(encoding="utf-8"))))


def display_deployment_outputs(output_dir: str | Path) -> None:
    """Display saved deployment reports and core PNG previews inside a notebook kernel."""
    root = Path(output_dir)
    try:
        from IPython.display import HTML, display
    except Exception as exc:  # pragma: no cover - notebook helper
        raise RuntimeError("IPython display is not available in this environment") from exc
    reports = sorted(root.glob("**/*_report.html")) or sorted(root.glob("**/report.html"))
    if not reports and root.is_file() and root.suffix.lower() in {".html", ".htm"}:
        reports = [root]
    if not reports:
        raise FileNotFoundError(f"No deployment HTML reports found under {root}")
    for report in reports:
        LOG.info("Displaying deployment report: %s", report)
        display(HTML(_notebook_safe_report_fragment(report.read_text(encoding="utf-8"))))


def _infer_probability(models: Sequence[torch.nn.Module],
                       tensor: torch.Tensor,
                       method: str,
                       inference_mode: str,
                       window_size: int,
                       window_overlap: int,
                       window_batch_size: int,
                       window_blend: str = "uniform") -> tuple[np.ndarray, Optional[np.ndarray]]:
    member_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for model in models:
            from floods.sliding_window import main_logits, sliding_window_logits

            if inference_mode == "sliding_window":
                logits = sliding_window_logits(model, tensor, window_size=window_size, overlap=window_overlap, window_batch_size=window_batch_size, blend_mode=window_blend)
            else:
                logits = main_logits(model(tensor))
            member_probs.append(torch.sigmoid(logits.float()))
        if len(member_probs) == 1:
            prob = member_probs[0]
            uncertainty = None
        else:
            stack = torch.stack(member_probs, dim=0)
            if method == "mean_prob":
                prob = stack.mean(dim=0)
            elif method == "mean_logit":
                from floods.sliding_window import probability_to_logit

                logits = torch.stack([probability_to_logit(p) for p in member_probs], dim=0).mean(dim=0)
                prob = torch.sigmoid(logits)
            else:
                raise ValueError("ensemble_method must be mean_prob or mean_logit")
            uncertainty = stack.std(dim=0).squeeze().detach().cpu().numpy().astype(np.float32)
        probability = prob.squeeze().detach().cpu().numpy().astype(np.float32)
    return probability, uncertainty



def _window_starts_for_diagnostics(length: int, window_size: int, stride: int) -> list[int]:
    """Mirror deployment sliding-window placement without importing model code."""
    if length <= window_size:
        return [0]
    starts = list(range(0, max(length - window_size + 1, 1), stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _write_sliding_window_diagnostics(output_dir: Path,
                                      prefix: str,
                                      raw_channels: dict[str, np.ndarray],
                                      probability: np.ndarray,
                                      window_size: int,
                                      window_overlap: int) -> dict[str, Any]:
    """Write inference-grid diagnostics from the already computed probability map.

    This does not rerun inference. It records window placement, input validity,
    per-window prediction behaviour, coverage counts and discontinuities at the
    exact stitching boundaries used by sliding-window inference.
    """
    height, width = probability.shape
    window_size = int(window_size)
    overlap = int(window_overlap)
    stride = max(window_size - overlap, 1)
    padded_h = max(height, window_size)
    padded_w = max(width, window_size)
    rows = _window_starts_for_diagnostics(padded_h, window_size, stride)
    cols = _window_starts_for_diagnostics(padded_w, window_size, stride)
    diagnostics_dir = Path(output_dir) / "window_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    counts = np.zeros((padded_h, padded_w), dtype=np.uint16)
    records: list[dict[str, Any]] = []
    channel_names = [name for name in ("vv", "vh", "dem") if name in raw_channels]
    for index, (row, col) in enumerate((r, c) for r in rows for c in cols):
        row_end = min(row + window_size, height)
        col_end = min(col + window_size, width)
        valid_h = max(0, row_end - row)
        valid_w = max(0, col_end - col)
        counts[row:row + window_size, col:col + window_size] += 1
        rec: dict[str, Any] = {
            "window_index": index,
            "row": row,
            "col": col,
            "row_end": row_end,
            "col_end": col_end,
            "valid_height": valid_h,
            "valid_width": valid_w,
            "pad_bottom": max(0, row + window_size - height),
            "pad_right": max(0, col + window_size - width),
            "valid_area_fraction": float((valid_h * valid_w) / max(1, window_size * window_size)),
        }
        if valid_h and valid_w:
            prob_crop = probability[row:row_end, col:col_end]
            rec.update({
                "probability_min": float(np.nanmin(prob_crop)),
                "probability_max": float(np.nanmax(prob_crop)),
                "probability_mean": float(np.nanmean(prob_crop)),
                "probability_median": float(np.nanmedian(prob_crop)),
                "predicted_flood_fraction_at_0_45": float(np.mean(prob_crop >= 0.45)),
            })
            valid_stack = []
            for name in channel_names:
                crop = np.asarray(raw_channels[name][row:row_end, col:col_end], dtype=np.float32)
                finite = np.isfinite(crop)
                valid_stack.append(finite)
                vals = crop[finite]
                rec[f"{name}_finite_fraction"] = float(finite.mean())
                rec[f"{name}_zero_fraction"] = float(np.mean(crop[finite] == 0.0)) if vals.size else 1.0
                rec[f"{name}_min"] = float(vals.min()) if vals.size else None
                rec[f"{name}_max"] = float(vals.max()) if vals.size else None
                rec[f"{name}_mean"] = float(vals.mean()) if vals.size else None
                rec[f"{name}_std"] = float(vals.std()) if vals.size else None
            if valid_stack:
                rec["all_modalities_finite_fraction"] = float(np.logical_and.reduce(valid_stack).mean())
        records.append(rec)

    # Compare exact seam gradients with the full-scene gradient distribution.
    # A raw jump alone is not proof of a stitching artefact because a real image
    # boundary may coincide with a window start. Percentile and excess ratios
    # make the diagnostic interpretable.
    all_vertical_jumps = np.nanmean(np.abs(probability[:, 1:] - probability[:, :-1]), axis=0) if width > 1 else np.array([0.0])
    all_horizontal_jumps = np.nanmean(np.abs(probability[1:, :] - probability[:-1, :]), axis=1) if height > 1 else np.array([0.0])
    vertical_seams = []
    for col in sorted(set(cols[1:])):
        if 0 < col < width:
            vertical_seams.append({
                "col": col,
                "mean_abs_jump": float(np.nanmean(np.abs(probability[:, col] - probability[:, col - 1]))),
                "p95_abs_jump": float(np.nanpercentile(np.abs(probability[:, col] - probability[:, col - 1]), 95)),
                "scene_gradient_percentile": float(np.mean(all_vertical_jumps <= all_vertical_jumps[col - 1])),
                "median_excess_ratio": float(all_vertical_jumps[col - 1] / max(float(np.nanmedian(all_vertical_jumps)), 1e-9)),
            })
    horizontal_seams = []
    for row in sorted(set(rows[1:])):
        if 0 < row < height:
            horizontal_seams.append({
                "row": row,
                "mean_abs_jump": float(np.nanmean(np.abs(probability[row, :] - probability[row - 1, :]))),
                "p95_abs_jump": float(np.nanpercentile(np.abs(probability[row, :] - probability[row - 1, :]), 95)),
                "scene_gradient_percentile": float(np.mean(all_horizontal_jumps <= all_horizontal_jumps[row - 1])),
                "median_excess_ratio": float(all_horizontal_jumps[row - 1] / max(float(np.nanmedian(all_horizontal_jumps)), 1e-9)),
            })

    csv_path = diagnostics_dir / f"{prefix}_window_statistics.csv"
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    counts_crop = counts[:height, :width].astype(np.float32)
    count_path = _save_png(diagnostics_dir / f"{prefix}_overlap_count.png", counts_crop, cmap="viridis")
    base = _normalise_for_display(raw_channels.get("vv", probability))
    grid_rgb = np.dstack([base, base, base])
    grid_rgb = np.clip(grid_rgb * 255.0, 0, 255).astype(np.uint8)
    for row in rows:
        if row < height:
            cv2.line(grid_rgb, (0, row), (width - 1, row), (255, 255, 255), 1)
    for col in cols:
        if col < width:
            cv2.line(grid_rgb, (col, 0), (col, height - 1), (255, 255, 255), 1)
    grid_path = _save_rgb(diagnostics_dir / f"{prefix}_window_grid.png", grid_rgb)

    summary = {
        "height": height,
        "width": width,
        "window_size": window_size,
        "overlap": overlap,
        "stride": stride,
        "rows": rows,
        "cols": cols,
        "num_windows": len(records),
        "coverage_count_min": int(counts_crop.min()),
        "coverage_count_max": int(counts_crop.max()),
        "uncovered_pixels": int(np.sum(counts_crop == 0)),
        "vertical_seams": vertical_seams,
        "horizontal_seams": horizontal_seams,
        "max_vertical_mean_abs_jump": max((x["mean_abs_jump"] for x in vertical_seams), default=0.0),
        "max_horizontal_mean_abs_jump": max((x["mean_abs_jump"] for x in horizontal_seams), default=0.0),
        "window_statistics_csv": str(csv_path),
        "window_grid_png": str(grid_path),
        "overlap_count_png": str(count_path),
    }
    json_path = diagnostics_dir / f"{prefix}_window_diagnostics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["diagnostics_json"] = str(json_path)
    return summary


def _predict_candidate(candidate: SceneCandidate,
                       dem_path: Optional[Path],
                       models: Sequence[torch.nn.Module],
                       modalities: Sequence[str],
                       normalization_mode: str,
                       normalization_stats_path: Path,
                       method: str,
                       inference_mode: str,
                       window_size: int,
                       window_overlap: int,
                       window_batch_size: int,
                       window_blend: str,
                       threshold: float,
                       min_component_area: int,
                       output_dir: Path,
                       output_prefix: str,
                       output_mask: Optional[Path],
                       output_probability: Optional[Path],
                       mask_path: Optional[Path],
                       evaluate: bool,
                       write_probability: bool,
                       write_previews: bool,
                       write_overlay: bool,
                       write_uncertainty: bool,
                       write_html_report: bool,
                       display_inline: bool,
                       explain: bool,
                       explain_per_modality: bool,
                       write_window_diagnostics: bool,
                       device: torch.device,
                       manifest_path: Path) -> dict[str, Any]:
    _deploy_log("Reading SAR candidate %s", candidate.candidate_id)
    vv, vh, profile = _read_sar_arrays(candidate)
    _deploy_log("Building input stack: modalities=%s shape=%sx%s", ",".join(modalities), vv.shape[0], vv.shape[1])
    tensor, raw_channels = _build_tensor_from_arrays(vv, vh, dem_path, profile, modalities, normalization_mode, normalization_stats_path, device)
    _deploy_log("Running inference: mode=%s window=%s overlap=%s models=%d", inference_mode, window_size, window_overlap, len(models))
    probability, uncertainty = _infer_probability(models, tensor, method, inference_mode, window_size, window_overlap, window_batch_size, window_blend)
    _deploy_log("Applying threshold %.3f and component filter min_area=%d", threshold, min_component_area)
    pred = (probability >= threshold).astype(np.uint8)
    pred = _apply_component_filter(pred, min_component_area)
    pixel_area = _pixel_area_km2(profile)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_name(output_prefix or candidate.candidate_id)
    window_diagnostics = None
    if write_window_diagnostics and inference_mode == "sliding_window":
        window_diagnostics = _write_sliding_window_diagnostics(output_dir, prefix, raw_channels, probability, window_size, window_overlap)
        _deploy_log("Window diagnostics: windows=%d coverage=%d-%d max_seam_jump=%.4f", window_diagnostics["num_windows"], window_diagnostics["coverage_count_min"], window_diagnostics["coverage_count_max"], max(window_diagnostics["max_vertical_mean_abs_jump"], window_diagnostics["max_horizontal_mean_abs_jump"]), level="summary")
    elif write_window_diagnostics:
        _deploy_log("Window diagnostics requested, but inference mode is direct; no stitching grid exists.", level="summary")
    flood_mask_path = Path(output_mask) if output_mask else output_dir / f"{prefix}_flood_mask.tif"
    prob_path = Path(output_probability) if output_probability else output_dir / f"{prefix}_flood_probability.tif"
    _deploy_log("Writing binary flood mask GeoTIFF: %s", flood_mask_path)
    _write_mask(flood_mask_path, profile, pred)
    if write_probability or output_probability:
        _deploy_log("Writing flood probability GeoTIFF: %s", prob_path)
        _write_float(prob_path, profile, probability)
    uncertainty_path = None
    if write_uncertainty and uncertainty is not None:
        uncertainty_path = output_dir / f"{prefix}_ensemble_uncertainty.tif"
        _deploy_log("Writing ensemble uncertainty GeoTIFF: %s", uncertainty_path)
        _write_float(uncertainty_path, profile, uncertainty)

    target_mask = None
    evaluation_metrics = None
    mask_alignment = None
    evaluation_paths: dict[str, Optional[str]] = {
        "metrics_json": None,
        "confusion_matrix_json": None,
        "confusion_matrix_png": None,
        "error_overlay_png": None,
        "ground_truth_overlay_png": None,
    }
    if mask_path is not None:
        mask_alignment = _mask_alignment_info(Path(mask_path), profile, pred.shape)
        _deploy_log(
            "Mask pairing check: candidate=%s | mask=%s | crs_match=%s | bounds_overlap=%.3f | resampled=%s",
            candidate.candidate_id,
            Path(mask_path).name,
            "yes" if mask_alignment.get("crs_match") else "no",
            float(mask_alignment.get("bounds_overlap_ratio") or 0.0),
            "yes" if mask_alignment.get("resampled_to_sar_grid") else "no",
            level="summary",
        )
        _deploy_log("Reading ground-truth mask for labelled-scene evaluation: %s", mask_path)
        target_mask = _read_mask_for_evaluation(Path(mask_path), profile, pred.shape)
        evaluation_metrics = _compute_binary_metrics(pred, target_mask)
        metrics_path = output_dir / f"{prefix}_evaluation_metrics.json"
        cm_json_path = output_dir / f"{prefix}_confusion_matrix.json"
        metrics_path.write_text(json.dumps(evaluation_metrics, indent=2), encoding="utf-8")
        cm_json_path.write_text(json.dumps({"tn": evaluation_metrics["tn"], "fp": evaluation_metrics["fp"], "fn": evaluation_metrics["fn"], "tp": evaluation_metrics["tp"]}, indent=2), encoding="utf-8")
        evaluation_paths["metrics_json"] = str(metrics_path)
        evaluation_paths["confusion_matrix_json"] = str(cm_json_path)
        if evaluate or write_html_report or write_previews or write_overlay or display_inline:
            cm_png = _save_confusion_matrix_png(output_dir / f"{prefix}_confusion_matrix.png", evaluation_metrics)
            gt_overlay_png = _save_rgb(output_dir / "previews" / f"{prefix}_ground_truth_overlay.png", _overlay_mask(raw_channels["vv"], np.where(target_mask == 255, 0, target_mask).astype(np.uint8), alpha=0.45))
            error_png = _save_error_overlay(output_dir / "previews" / f"{prefix}_error_overlay.png", raw_channels["vv"], pred, target_mask)
            evaluation_paths["confusion_matrix_png"] = str(cm_png)
            evaluation_paths["ground_truth_overlay_png"] = str(gt_overlay_png)
            evaluation_paths["error_overlay_png"] = str(error_png)
        _deploy_log("Labelled-scene metrics: F1=%.4f IoU=%.4f P=%.4f R=%.4f MCC=%.4f",
                    evaluation_metrics["f1"], evaluation_metrics["iou"], evaluation_metrics["precision"], evaluation_metrics["recall"], evaluation_metrics["mcc"],
                    level="summary")

    image_paths: list[tuple[str, Path]] = []
    preview_dir = output_dir / "previews"
    if write_previews or write_overlay or write_html_report or display_inline or explain:
        _deploy_log("Creating visual previews and flood prediction overlays in %s", preview_dir)
        vv_png = _save_png(preview_dir / f"{prefix}_vv_input.png", _normalise_for_display(raw_channels["vv"]), cmap="gray")
        vh_png = _save_png(preview_dir / f"{prefix}_vh_input.png", _normalise_for_display(raw_channels["vh"]), cmap="gray")
        image_paths.extend([("Input VV", vv_png), ("Input VH", vh_png)])
        if "dem" in raw_channels:
            dem_png = _save_png(preview_dir / f"{prefix}_dem_input.png", _normalise_for_display(raw_channels["dem"]), cmap="terrain")
            image_paths.append(("Input DEM", dem_png))
            rgb = np.dstack([_normalise_for_display(raw_channels["vv"]), _normalise_for_display(raw_channels["vh"]), _normalise_for_display(raw_channels["dem"])])
            image_paths.append(("False-colour VV/VH/DEM", _save_rgb(preview_dir / f"{prefix}_false_colour.png", rgb)))
        else:
            rgb = np.dstack([_normalise_for_display(raw_channels["vv"]), _normalise_for_display(raw_channels["vh"]), np.zeros_like(raw_channels["vv"], dtype=np.float32)])
            image_paths.append(("False-colour VV/VH", _save_rgb(preview_dir / f"{prefix}_false_colour.png", rgb)))
        probability_png = _save_png(preview_dir / f"{prefix}_flood_probability_heatmap.png", probability, cmap="magma")
        image_paths.append(("Flood probability heatmap", probability_png))
        image_paths.append(("Flood probability overlay", _save_rgb(preview_dir / f"{prefix}_probability_overlay.png", _overlay_heatmap(raw_channels["vv"], probability, cmap="magma"))))
        image_paths.append(("Binary flood-mask overlay", _save_rgb(preview_dir / f"{prefix}_binary_mask_overlay.png", _overlay_mask(raw_channels["vv"], pred))))
        if target_mask is not None:
            target_png = _save_png(preview_dir / f"{prefix}_ground_truth_mask.png", np.where(target_mask == 255, 0, target_mask).astype(np.float32), cmap="gray")
            image_paths.append(("Ground-truth mask", target_png))
            if evaluation_paths.get("ground_truth_overlay_png"):
                image_paths.append(("SAR + ground-truth mask overlay", Path(evaluation_paths["ground_truth_overlay_png"])))
            if evaluation_paths.get("error_overlay_png"):
                image_paths.append(("Evaluation overlay: green=TP, red=FP, blue=FN", Path(evaluation_paths["error_overlay_png"])))
            if evaluation_paths.get("confusion_matrix_png"):
                image_paths.append(("Confusion matrix", Path(evaluation_paths["confusion_matrix_png"])))
        crop = _save_top_crop(preview_dir / f"{prefix}_top_prediction_crop.png", raw_channels["vv"], probability, size=min(512, max(probability.shape)))
        if crop:
            image_paths.append(("Zoom crop around strongest prediction", crop))
        if explain:
            positive = _save_rgb(preview_dir / f"{prefix}_positive_evidence.png", _overlay_heatmap(raw_channels["vv"], probability, cmap="inferno", alpha=0.70))
            image_paths.append(("Positive evidence map", positive))
        if write_uncertainty and uncertainty is not None:
            uncertainty_norm = _normalise_for_display(uncertainty, lower=0, upper=100)
            unc_png = _save_png(preview_dir / f"{prefix}_ensemble_uncertainty.png", uncertainty_norm, cmap="viridis")
            image_paths.append(("Ensemble uncertainty/disagreement", unc_png))
        if explain_per_modality:
            modality_contribs = _modality_occlusion_explanations(models, tensor, probability, modalities, method, inference_mode, window_size, window_overlap, window_batch_size)
            for modality, values in modality_contribs.items():
                values_norm = _normalise_for_display(np.clip(values, 0.0, None), lower=1, upper=99)
                path = _save_png(preview_dir / f"{prefix}_explanation_{modality}.png", values_norm, cmap="plasma")
                image_paths.append((f"Occlusion contribution: {modality.upper()}", path))

    metadata = {
        "manifest": str(manifest_path),
        "candidate_id": candidate.candidate_id,
        "candidate_kind": candidate.kind,
        "date": candidate.date,
        "sar_path": str(candidate.sar_path) if candidate.sar_path else None,
        "vv_path": str(candidate.vv_path) if candidate.vv_path else None,
        "vh_path": str(candidate.vh_path) if candidate.vh_path else None,
        "dem_path": str(dem_path) if dem_path else None,
        "output_mask": str(flood_mask_path),
        "output_probability": str(prob_path) if write_probability or output_probability else None,
        "output_uncertainty": str(uncertainty_path) if uncertainty_path else None,
        "threshold": float(threshold),
        "min_component_area": int(min_component_area),
        "flood_pixels": int(pred.sum()),
        "flood_fraction": float(pred.sum() / max(1, pred.size)),
        "flood_area_km2": float(pred.sum() * pixel_area) if pixel_area is not None else None,
        "mask_path": str(mask_path) if mask_path else None,
        "mask_alignment": mask_alignment,
        "evaluation_metrics": evaluation_metrics,
        "evaluation_outputs": evaluation_paths if evaluation_metrics else None,
        "window_diagnostics": window_diagnostics,
        "height": int(pred.shape[0]),
        "width": int(pred.shape[1]),
        "device": str(device),
        "visual_report": None,
    }
    metadata_path = output_dir / f"{prefix}_prediction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path = output_dir / f"{prefix}_report.html"
    if write_html_report or display_inline:
        _deploy_log("Writing self-contained HTML deployment report: %s", report_path)
        _write_visual_report(report_path, f"Flood Extent Mapping deployment report: {candidate.candidate_id}", metadata, image_paths)
        metadata["visual_report"] = str(report_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if display_inline:
        _display_inline(report_path, image_paths)
    area_text = f" | area_km2={metadata['flood_area_km2']:.3f}" if metadata.get("flood_area_km2") is not None else ""
    result_label = candidate.candidate_id if _deploy_is_concise() else str(flood_mask_path)
    _deploy_log("Prediction complete: %s | flood_pixels=%d | flood_fraction=%.2f%%%s", result_label, metadata["flood_pixels"], metadata["flood_fraction"] * 100.0, area_text, level="summary")
    return metadata


def _modality_occlusion_explanations(models: Sequence[torch.nn.Module],
                                     tensor: torch.Tensor,
                                     original_probability: np.ndarray,
                                     modalities: Sequence[str],
                                     method: str,
                                     inference_mode: str,
                                     window_size: int,
                                     window_overlap: int,
                                     window_batch_size: int) -> dict[str, np.ndarray]:
    contributions: dict[str, np.ndarray] = {}
    for channel_index, modality in enumerate(modalities):
        occluded = tensor.clone()
        occluded[:, channel_index] = 0.0
        occluded_probability, _ = _infer_probability(models, occluded, method, inference_mode, window_size, window_overlap, window_batch_size)
        contributions[str(modality).lower()] = original_probability - occluded_probability
    return contributions


def _load_deployment_context(manifest_path: Path, device: str) -> tuple[dict[str, Any], list[TrainConfig], list[torch.nn.Module], torch.device]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    _deploy_log("Loading deployment manifest: %s", manifest_path)
    manifest = _read_manifest(manifest_path)
    members = manifest.get("members") or []
    if not members:
        raise ValueError("Deployment manifest does not contain any members")

    config_paths = [
        _resolve_manifest_reference(manifest_path, item.get("config"), role=f"member {index} config")
        for index, item in enumerate(members, start=1)
    ]
    checkpoint_paths = [
        _resolve_manifest_reference(manifest_path, item.get("checkpoint"), role=f"member {index} checkpoint")
        for index, item in enumerate(members, start=1)
    ]
    missing = [path for path in [*config_paths, *checkpoint_paths] if not path.is_file()]
    if missing:
        portable = bool((manifest.get("bundle") or {}).get("portable"))
        hint = " Keep the entire deployment bundle together." if portable else ""
        raise FileNotFoundError(
            "Deployment asset(s) not found: " + ", ".join(str(path) for path in missing) + hint
        )

    configs: list[TrainConfig] = []
    for config_path in config_paths:
        config = _load_train_config(config_path)
        configured_stats = getattr(config.data, "normalization_stats_path", None)
        if configured_stats:
            configured_path = Path(str(configured_stats)).expanduser()
            if not configured_path.is_absolute():
                config.data.normalization_stats_path = str((config_path.parent / configured_path).resolve())
        configs.append(config)
    from floods.ensemble_evaluation import _ensure_compatible_configs

    _ensure_compatible_configs(configs)
    use_cuda = str(device).lower() != "cpu" and torch.cuda.is_available()
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    _deploy_log(
        "Model mode: %s | members=%d | device=%s | portable_bundle=%s",
        manifest.get("mode") or ("ensemble" if len(members) > 1 else "single"),
        len(members),
        torch_device,
        bool((manifest.get("bundle") or {}).get("portable")),
    )
    models = _load_models(configs, checkpoint_paths, torch_device)
    return manifest, configs, models, torch_device


def _deployment_settings(manifest: dict[str, Any], configs: Sequence[TrainConfig], manifest_path: Path) -> dict[str, Any]:
    inputs = manifest.get("inputs") or {}
    inference = manifest.get("inference") or {}
    operating_point = manifest.get("operating_point") or {}
    normalization_stats_path = inputs.get("normalization_stats_path") or configs[0].data.normalization_stats_path
    if not normalization_stats_path:
        raise ValueError("A deployment manifest must include normalization_stats_path, or the config must provide it")
    resolved_stats_path = _resolve_manifest_reference(
        Path(manifest_path), normalization_stats_path, role="normalization statistics"
    )
    if not resolved_stats_path.is_file():
        portable = bool((manifest.get("bundle") or {}).get("portable"))
        hint = " Keep the entire deployment bundle together." if portable else ""
        raise FileNotFoundError(f"Deployment normalization statistics not found: {resolved_stats_path}.{hint}")
    return {
        "modalities": inputs.get("modalities") or ["vv", "vh", "dem"],
        "normalization_mode": inputs.get("normalization_mode") or configs[0].data.normalization_mode or "robust_percentile",
        "normalization_stats_path": resolved_stats_path,
        "method": str(manifest.get("ensemble_method") or "mean_logit").lower().replace("-", "_"),
        "inference_mode": str(inference.get("mode") or "sliding_window").lower().replace("-", "_"),
        "window_size": int(inference.get("window_size") or 512),
        "window_overlap": int(inference.get("window_overlap") or 128),
        "window_batch_size": int(inference.get("window_batch_size") or 1),
        "window_blend": str(inference.get("window_blend") or "uniform").lower().replace("-", "_"),
        "threshold": float(operating_point.get("threshold", 0.5)),
        "min_component_area": int(operating_point.get("min_component_area", 0)),
    }


def predict_scene(manifest_path: Path,
                  sar_path: Optional[Path] = None,
                  dem_path: Optional[Path] = None,
                  mask_path: Optional[Path] = None,
                  mask_dir: Optional[Path] = None,
                  evaluate: bool = False,
                  output_mask: Optional[Path] = None,
                  output_probability: Optional[Path] = None,
                  device: str = "cuda",
                  scene_dir: Optional[Path] = None,
                  input_csv: Optional[Path] = None,
                  dem_dir: Optional[Path] = None,
                  output_dir: Optional[Path] = None,
                  output_prefix: Optional[str] = None,
                  scene_id: Optional[str] = None,
                  candidate_prefix: Optional[str] = None,
                  candidate_name_template: Optional[str] = None,
                  sar_selection: str = "all",
                  sar_date: Optional[str] = None,
                  mosaic_compatible_sar_tiles: bool = False,
                  mosaic_undated: bool = False,
                  mosaic_mode: str = "smart",
                  write_probability: bool = False,
                  write_previews: bool = False,
                  write_overlay: bool = False,
                  write_uncertainty: bool = False,
                  write_html_report: bool = False,
                  display_inline: bool = False,
                  explain: bool = False,
                  explain_per_modality: bool = False,
                  write_window_diagnostics: bool = False,
                  window_blend: Optional[str] = None,
                  output_mode: str = "standard",
                  prediction_only: bool = False) -> dict[str, Any]:
    """Run deployment inference on one direct SAR/DEM scene or a discovered scene folder."""
    _set_deploy_output_mode(output_mode)
    mosaic_mode = str(mosaic_mode or "smart").lower().strip()
    if mosaic_compatible_sar_tiles and mosaic_mode in {"smart", "plan"}:
        # The boolean flag means users want mosaicking where the planner judges it safe.
        mosaic_mode = "auto"
    if mosaic_mode not in {"smart", "off", "plan", "auto", "force"}:
        raise ValueError("mosaic_mode must be one of: smart, off, plan, auto, force")
    if prediction_only:
        write_previews = False
        write_overlay = False
        write_uncertainty = False
        write_html_report = False
        display_inline = False
        explain = False
        explain_per_modality = False
    _deploy_log("Process starting", level="summary")
    provided_inputs = [bool(sar_path), bool(scene_dir), bool(input_csv)]
    if sum(provided_inputs) != 1:
        raise ValueError("Provide exactly one of --sar-path, --scene-dir, or --input-csv")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest, configs, models, torch_device = _load_deployment_context(manifest_path, device)
    settings = _deployment_settings(manifest, configs, manifest_path)
    if window_blend is not None:
        settings["window_blend"] = str(window_blend).lower().replace("-", "_")
    if settings["window_blend"] not in {"uniform", "cosine"}:
        raise ValueError("window_blend must be uniform or cosine")
    _deploy_log("Operating point: threshold=%.3f min_component_area=%d method=%s", settings["threshold"], settings["min_component_area"], settings["method"])
    _deploy_log("Input modalities: %s", ", ".join(settings["modalities"]))

    resolved_dem = _find_dem(Path(scene_dir) if scene_dir else None, Path(dem_path) if dem_path else None, Path(dem_dir) if dem_dir else None)
    if resolved_dem:
        _deploy_log("Using DEM: %s", resolved_dem)
    elif not input_csv:
        _deploy_log("No DEM found/provided; this is only valid if the manifest modalities exclude DEM")
    if input_csv:
        _deploy_log("Reading deployment input CSV: %s", input_csv)
        discovered = _candidates_from_input_csv(Path(input_csv))
        candidates = _select_candidates(discovered, selection=sar_selection, sar_date=sar_date)
        base_output_dir = Path(output_dir or Path(input_csv).with_suffix(""))
        inventory_path = base_output_dir / "scene_inventory.csv"
        _write_inventory_csv(discovered, inventory_path)
    elif scene_dir:
        _deploy_log("Analysing SAR files in scene folder: %s", scene_dir)
        discovered = _discover_candidates(Path(scene_dir), scene_id=scene_id, candidate_prefix=candidate_prefix, candidate_name_template=candidate_name_template)
        ready_count = sum(1 for item in discovered if item.status == "ready")
        _deploy_log("SAR inventory complete: %d ready candidate(s), %d total inventory row(s)", ready_count, len(discovered))
        for item in discovered:
            detail = item.sar_path or item.vv_path or item.vh_path
            _deploy_log("Candidate: id=%s status=%s kind=%s date=%s path=%s", item.candidate_id, item.status, item.kind, item.date or "undated", detail)
        candidates = _select_candidates(discovered, selection=sar_selection, sar_date=sar_date)
        _deploy_log("Selected SAR candidate(s): %s", ", ".join(candidate.candidate_id for candidate in candidates))
        base_output_dir = Path(output_dir or Path(scene_dir) / "flood_predictions")
        inventory_path = base_output_dir / "scene_inventory.csv"
        _write_inventory_csv(discovered, inventory_path)
    else:
        _deploy_log("Using direct SAR file: %s", sar_path)
        direct_info = _raster_info(Path(sar_path))  # type: ignore[arg-type]
        direct_id = _format_candidate_id(stem=output_prefix or Path(sar_path).stem,
                                         date=_extract_date(Path(sar_path)),
                                         scene_id=scene_id,
                                         candidate_prefix=candidate_prefix,
                                         candidate_name_template=candidate_name_template,
                                         kind="multiband_vv_vh")
        candidates = [SceneCandidate(candidate_id=direct_id,
                                     kind="multiband_vv_vh",
                                     date=_extract_date(Path(sar_path)),
                                     sar_path=Path(sar_path),
                                     crs=direct_info.get("crs"),
                                     height=direct_info.get("height"),
                                     width=direct_info.get("width"),
                                     count=direct_info.get("count"),
                                     bounds=tuple(direct_info.get("bounds")) if direct_info.get("bounds") else None,
                                     res=tuple(direct_info.get("res")) if direct_info.get("res") else None)]  # type: ignore[arg-type]
        base_output_dir = Path(output_dir or (Path(output_mask).parent if output_mask else Path.cwd() / "flood_prediction"))
        inventory_path = None

    if len(candidates) > 1 and mosaic_mode != "off":
        _log_mosaic_plan(candidates, mode=mosaic_mode, evaluating=bool(evaluate or mask_path or mask_dir))
    should_mosaic = len(candidates) > 1 and mosaic_mode in {"smart", "auto", "force"}
    if should_mosaic:
        evaluating_now = bool(evaluate or mask_path or mask_dir)
        if mosaic_mode == "smart":
            use_name_group = True
            allow_undated_now = bool(mosaic_undated)
            require_mask_mosaic = evaluating_now
        else:
            use_name_group = mosaic_mode == "force" or (mosaic_mode == "auto" and not evaluating_now)
            allow_undated_now = bool(mosaic_undated or mosaic_mode == "force")
            require_mask_mosaic = evaluating_now and mosaic_mode != "force"
        candidates = _mosaic_multiband_candidates(candidates,
                                                 base_output_dir,
                                                 allow_undated=allow_undated_now,
                                                 use_name_group=use_name_group,
                                                 evaluating=evaluating_now,
                                                 mask_path=Path(mask_path) if mask_path else None,
                                                 mask_dir=Path(mask_dir) if mask_dir else None,
                                                 require_mask_mosaic=require_mask_mosaic)
        _write_inventory_csv(candidates, base_output_dir / "selected_candidates_after_mosaic.csv")

    results = []
    for index, candidate in enumerate(candidates, start=1):
        _deploy_log("Running prediction for candidate %d/%d: %s", index, len(candidates), candidate.candidate_id)
        sub_output_dir = base_output_dir if len(candidates) == 1 else base_output_dir / candidate.candidate_id
        single_output_mask = output_mask if len(candidates) == 1 else None
        single_output_probability = output_probability if len(candidates) == 1 else None
        prefix = output_prefix or candidate.candidate_id
        resolved_mask = candidate.mask_path or _find_mask(
            Path(mask_path) if mask_path else None,
            Path(mask_dir) if mask_dir else None,
            candidate,
        )
        if resolved_mask:
            _deploy_log("Using ground-truth mask for %s: %s", candidate.candidate_id, resolved_mask)
        candidate_dem = candidate.dem_path or resolved_dem
        if candidate.dem_path:
            _deploy_log("Using candidate-specific DEM for %s: %s", candidate.candidate_id, candidate.dem_path)
        result = _predict_candidate(candidate=candidate,
                                    dem_path=candidate_dem,
                                    models=models,
                                    modalities=settings["modalities"],
                                    normalization_mode=settings["normalization_mode"],
                                    normalization_stats_path=settings["normalization_stats_path"],
                                    method=settings["method"],
                                    inference_mode=settings["inference_mode"],
                                    window_size=settings["window_size"],
                                    window_overlap=settings["window_overlap"],
                                    window_batch_size=settings["window_batch_size"],
                                    window_blend=settings["window_blend"],
                                    threshold=settings["threshold"],
                                    min_component_area=settings["min_component_area"],
                                    output_dir=sub_output_dir,
                                    output_prefix=prefix if len(candidates) == 1 else candidate.candidate_id,
                                    output_mask=single_output_mask,
                                    output_probability=single_output_probability,
                                    mask_path=resolved_mask,
                                    evaluate=evaluate or resolved_mask is not None,
                                    write_probability=write_probability,
                                    write_previews=write_previews,
                                    write_overlay=write_overlay,
                                    write_uncertainty=write_uncertainty,
                                    write_html_report=write_html_report,
                                    display_inline=display_inline,
                                    explain=explain,
                                    explain_per_modality=explain_per_modality,
                                    write_window_diagnostics=write_window_diagnostics,
                                    device=torch_device,
                                    manifest_path=Path(manifest_path))
        results.append(result)

    summary = {
        "manifest": str(manifest_path),
        "scene_dir": str(scene_dir) if scene_dir else None,
        "input_csv": str(input_csv) if input_csv else None,
        "sar_path": str(sar_path) if sar_path else None,
        "dem_path": str(resolved_dem) if resolved_dem else None,
        "mask_path": str(mask_path) if mask_path else None,
        "mask_dir": str(mask_dir) if mask_dir else None,
        "evaluate": bool(evaluate or mask_path or mask_dir),
        "mosaic_mode": mosaic_mode,
        "output_dir": str(base_output_dir),
        "num_predictions": len(results),
        "predictions": results,
    }
    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_output_dir / "deployment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _deploy_log("Deployment summary JSON written to: %s", summary_path, level="standard")
    if len(results) > 1 and write_html_report:
        report_links: list[tuple[str, Path]] = []
        for item in results:
            report = item.get("visual_report")
            if report:
                report_links.append((item.get("candidate_id", "prediction"), Path(report)))
        html_summary = _write_deployment_summary_html(base_output_dir / "deployment_summary.html", summary, report_links, inventory_path)
        _deploy_log("Deployment summary HTML written to: %s", html_summary, level="standard")
    _print_deployment_result_table(summary)
    _deploy_log("Done. Outputs written to: %s", base_output_dir, level="summary")
    if display_inline:
        _deploy_log("For Colab display after a shell command, run: from floods.deployment import display_deployment_outputs; display_deployment_outputs(r'%s')", base_output_dir.as_posix())
    return summary


def _format_optional(value: Any, fmt: str = "{:.4f}") -> str:
    if value is None:
        return "-"
    try:
        return fmt.format(float(value))
    except Exception:
        return str(value)


def _short_report_label(path_value: Any) -> str:
    if not path_value:
        return "-"
    try:
        return Path(str(path_value)).name
    except Exception:
        return str(path_value)


def _global_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / (denom + 1e-8) if denom > 0 else 0.0
    return {"f1": float(f1), "iou": float(iou), "precision": float(precision), "recall": float(recall), "mcc": float(mcc)}


def _print_deployment_result_table(summary: dict[str, Any]) -> None:
    """Print a compact, user-facing result table after deployment."""
    predictions = summary.get("predictions") or []
    if not predictions:
        return
    has_eval = any(item.get("evaluation_metrics") for item in predictions)
    _deploy_log("Flood prediction results", level="summary")
    if has_eval:
        header = f"{'Candidate':<28} {'Flood px':>10} {'Area km²':>10} {'Flood %':>9} {'F1':>7} {'IoU':>7} {'P':>7} {'R':>7} {'MCC':>7}"
    else:
        header = f"{'Candidate':<28} {'Flood px':>10} {'Area km²':>10} {'Flood %':>9} {'Report':<32}"
    _deploy_log(header, level="summary")
    _deploy_log("-" * len(header), level="summary")
    report_lines: list[str] = []
    total_tp = total_tn = total_fp = total_fn = 0
    for item in predictions:
        candidate_full = str(item.get("candidate_id") or "prediction")
        candidate = candidate_full[:28]
        flood_pixels = int(item.get("flood_pixels") or 0)
        area = _format_optional(item.get("flood_area_km2"), "{:.3f}")
        flood_pct = _format_optional((item.get("flood_fraction") or 0) * 100.0, "{:.2f}")
        metrics = item.get("evaluation_metrics") or {}
        report_path = item.get("visual_report")
        if report_path:
            report_lines.append(f"  {candidate_full}: {report_path}")
        if has_eval:
            row = (f"{candidate:<28} {flood_pixels:>10,} {area:>10} {flood_pct:>8}% "
                   f"{_format_optional(metrics.get('f1')):>7} {_format_optional(metrics.get('iou')):>7} "
                   f"{_format_optional(metrics.get('precision')):>7} {_format_optional(metrics.get('recall')):>7} {_format_optional(metrics.get('mcc')):>7}")
            if metrics:
                total_tp += int(metrics.get("tp") or 0)
                total_tn += int(metrics.get("tn") or 0)
                total_fp += int(metrics.get("fp") or 0)
                total_fn += int(metrics.get("fn") or 0)
        else:
            report = _short_report_label(report_path)
            row = f"{candidate:<28} {flood_pixels:>10,} {area:>10} {flood_pct:>8}% {report:<32}"
        _deploy_log(row, level="summary")
    if has_eval and (total_tp + total_tn + total_fp + total_fn) > 0:
        overall = _global_metrics_from_counts(total_tp, total_tn, total_fp, total_fn)
        _deploy_log(
            "Overall labelled result: F1=%.4f IoU=%.4f P=%.4f R=%.4f MCC=%.4f | TP=%s FP=%s FN=%s TN=%s",
            overall["f1"], overall["iou"], overall["precision"], overall["recall"], overall["mcc"],
            f"{total_tp:,}", f"{total_fp:,}", f"{total_fn:,}", f"{total_tn:,}",
            level="summary",
        )
    if report_lines and not _deploy_is_concise():
        _deploy_log("Reports:\n%s", "\n".join(report_lines), level="summary")
    elif report_lines and has_eval:
        _deploy_log("Reports saved under: %s", summary.get("output_dir"), level="summary")
    _deploy_log("Main output directory: %s", summary.get("output_dir"), level="summary")

def deploy_scene_colab(**kwargs: Any) -> dict[str, Any]:
    """Run deployment from inside a Colab/Jupyter Python cell and display results inline.

    This is the preferred interactive deployment entry point because shell commands
    such as ``!floodmap predict-scene`` cannot reliably render IPython HTML/images.
    Keyword arguments match :func:`predict_scene`, with string paths accepted.
    """
    path_keys = {
        "manifest_path", "sar_path", "dem_path", "mask_path", "mask_dir", "output_mask",
        "output_probability", "scene_dir", "dem_dir", "output_dir",
    }
    if "manifest" in kwargs and "manifest_path" not in kwargs:
        kwargs["manifest_path"] = kwargs.pop("manifest")
    for key in path_keys:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = Path(kwargs[key])
    kwargs.setdefault("display_inline", True)
    kwargs.setdefault("write_html_report", True)
    kwargs.setdefault("write_previews", True)
    kwargs.setdefault("write_overlay", True)
    kwargs.setdefault("write_probability", True)
    kwargs.setdefault("output_mode", "concise")
    summary = predict_scene(**kwargs)
    try:
        display_deployment_outputs(summary["output_dir"])
    except Exception as exc:
        _deploy_log("Colab inline display could not be rendered automatically: %s", exc, level="summary")
        _deploy_log("Reports are still saved under: %s", summary.get("output_dir"), level="summary")
    return summary


def _write_deployment_summary_html(path: Path, summary: dict[str, Any], report_links: Sequence[tuple[str, Path]], inventory_path: Optional[Path]) -> Path:
    path = Path(path)
    items = []
    for label, report in report_links:
        rel = html.escape(report.relative_to(path.parent).as_posix() if report.is_relative_to(path.parent) else report.as_posix())
        items.append(f"<li><a href='{rel}'>{html.escape(label)}</a></li>")
    inventory = ""
    if inventory_path and inventory_path.exists():
        rel_inv = html.escape(inventory_path.relative_to(path.parent).as_posix() if inventory_path.is_relative_to(path.parent) else inventory_path.as_posix())
        inventory = f"<p><a href='{rel_inv}'>Scene inventory CSV</a></p>"
    body = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Flood Extent Mapping deployment summary</title>
<style>
.floodmap-deployment-summary {{ box-sizing:border-box; font-family:Arial,sans-serif; margin:24px; color:#202124; line-height:1.45; background:#fff; }}
.floodmap-deployment-summary *, .floodmap-deployment-summary *::before, .floodmap-deployment-summary *::after {{ box-sizing:border-box; }}
.floodmap-deployment-summary pre {{ background:#f6f8fa; color:#202124; padding:12px; white-space:pre-wrap; overflow-wrap: anywhere; word-break:break-word; border-radius:6px; }}
.floodmap-deployment-summary a {{ color:#1a73e8; }}
</style>
</head>
<body>
<div class='floodmap-deployment-summary' data-floodmap-report='summary'>
<h1>Flood Extent Mapping deployment summary</h1>
{inventory}
<ul>{''.join(items)}</ul>
<pre>{html.escape(json.dumps(summary, indent=2))}</pre>
</div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")
    return path
