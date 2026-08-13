from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import rasterio
from rasterio.io import MemoryFile


_POL_RE = re.compile(r"(?i)(?:^|[_\-.])(vv|vh)(?:$|[_\-.])")
_AUX_SUFFIX_RE = re.compile(r"(?i)(?:[_\-.](?:dem|srtm|elevation|mask|label|labels|truth|gt|groundtruth|ground_truth))+$")
_POL_SUFFIX_RE = re.compile(r"(?i)(?:[_\-.](?:vv|vh))+$")
_POL_PREFIX_RE = re.compile(r"(?i)^(?:vv|vh)[_\-.]+")


@dataclass(frozen=True)
class RasterUploadInfo:
    name: str
    count: int
    width: int
    height: int
    crs: Optional[str]
    transform: tuple[float, ...]
    descriptions: tuple[str, ...]
    tags: Mapping[str, str]
    polarization: Optional[str]


@dataclass(frozen=True)
class SarCandidateUpload:
    candidate_id: str
    kind: str
    sar_path: Optional[Path] = None
    vv_path: Optional[Path] = None
    vh_path: Optional[Path] = None


def _uploaded_bytes(uploaded) -> bytes:
    if hasattr(uploaded, "getbuffer"):
        return bytes(uploaded.getbuffer())
    if hasattr(uploaded, "getvalue"):
        return bytes(uploaded.getvalue())
    if hasattr(uploaded, "read"):
        position = uploaded.tell() if hasattr(uploaded, "tell") else None
        data = uploaded.read()
        if position is not None and hasattr(uploaded, "seek"):
            uploaded.seek(position)
        return data
    raise TypeError("Uploaded raster object does not expose file bytes")


def infer_polarization(name: str, descriptions: Sequence[str] = (), tags: Mapping[str, str] | None = None) -> Optional[str]:
    """Return vv/vh when the file identifies one polarization unambiguously."""
    evidence = [Path(name).stem]
    evidence.extend(str(item or "") for item in descriptions)
    if tags:
        evidence.extend(f"{key}={value}" for key, value in tags.items())
    hits: set[str] = set()
    for value in evidence:
        for match in _POL_RE.finditer(value):
            hits.add(match.group(1).lower())
        lowered = value.strip().lower()
        if lowered in {"vv", "vh"}:
            hits.add(lowered)
    return next(iter(hits)) if len(hits) == 1 else None


def inspect_upload(uploaded) -> RasterUploadInfo:
    data = _uploaded_bytes(uploaded)
    with MemoryFile(data) as mem:
        with mem.open() as src:
            descriptions = tuple(str(item or "") for item in src.descriptions)
            tags = {str(k): str(v) for k, v in src.tags().items()}
            return RasterUploadInfo(
                name=str(uploaded.name),
                count=int(src.count),
                width=int(src.width),
                height=int(src.height),
                crs=src.crs.to_string() if src.crs else None,
                transform=tuple(float(v) for v in src.transform)[:6],
                descriptions=descriptions,
                tags=tags,
                polarization=infer_polarization(str(uploaded.name), descriptions, tags),
            )


def canonical_tile_id(name: str, *, auxiliary: bool = False) -> str:
    """Return a stable tile key used for SAR/DEM/mask matching."""
    stem = Path(name).stem.strip()
    if auxiliary:
        stem = _AUX_SUFFIX_RE.sub("", stem)
    stem = _POL_PREFIX_RE.sub("", stem)
    stem = _POL_SUFFIX_RE.sub("", stem)
    return stem or Path(name).stem


def save_upload(uploaded, destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_uploaded_bytes(uploaded))
    return destination


def prepare_sar_candidates(uploaded_files: Sequence, root: Path) -> tuple[list[SarCandidateUpload], list[str], list[str]]:
    """Inspect SAR uploads and return deployable combined or VV/VH candidates.

    Users do not need to declare whether an upload is one-band or multiband.
    Two-band rasters are accepted directly. One-band rasters are paired only when
    VV/VH can be identified safely from the filename or GeoTIFF metadata.
    """
    root = Path(root)
    combined: list[tuple[object, RasterUploadInfo]] = []
    singles: dict[str, list[tuple[object, RasterUploadInfo]]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for uploaded in uploaded_files:
        try:
            info = inspect_upload(uploaded)
        except Exception as exc:
            errors.append(f"{getattr(uploaded, 'name', 'upload')}: could not read GeoTIFF metadata ({exc}).")
            continue
        if info.count >= 2:
            combined.append((uploaded, info))
        elif info.count == 1:
            key = canonical_tile_id(info.name)
            singles.setdefault(key, []).append((uploaded, info))
        else:
            errors.append(f"{info.name}: raster contains no readable bands.")

    candidates: list[SarCandidateUpload] = []
    seen_ids: set[str] = set()

    for uploaded, info in combined:
        candidate_id = canonical_tile_id(info.name)
        if candidate_id in seen_ids:
            errors.append(f"Duplicate SAR tile identifier: {candidate_id}.")
            continue
        path = save_upload(uploaded, root / "sar" / info.name)
        candidates.append(SarCandidateUpload(candidate_id=candidate_id, kind="multiband_vv_vh", sar_path=path))
        seen_ids.add(candidate_id)
        if info.count > 2:
            warnings.append(
                f"{info.name} contains {info.count} bands; deployment uses bands 1 and 2 as VV/VH unless the exported model documents another contract."
            )

    for key, items in sorted(singles.items()):
        by_pol: dict[str, list[tuple[object, RasterUploadInfo]]] = {"vv": [], "vh": []}
        unknown: list[tuple[object, RasterUploadInfo]] = []
        for item in items:
            pol = item[1].polarization
            if pol in by_pol:
                by_pol[pol].append(item)
            else:
                unknown.append(item)
        if len(by_pol["vv"]) == 1 and len(by_pol["vh"]) == 1 and not unknown:
            if key in seen_ids:
                errors.append(f"Duplicate SAR tile identifier: {key}.")
                continue
            vv_upload, vv_info = by_pol["vv"][0]
            vh_upload, vh_info = by_pol["vh"][0]
            vv_path = save_upload(vv_upload, root / "sar" / vv_info.name)
            vh_path = save_upload(vh_upload, root / "sar" / vh_info.name)
            candidates.append(SarCandidateUpload(candidate_id=key, kind="separate_vv_vh", vv_path=vv_path, vh_path=vh_path))
            seen_ids.add(key)
            continue

        names = ", ".join(info.name for _, info in items)
        if len(items) == 1:
            errors.append(
                f"{names} is a single-band SAR raster. Upload its matching VV/VH polarization file as well."
            )
        else:
            errors.append(
                f"Could not safely identify one VV and one VH raster for tile {key}: {names}. "
                "Use filenames or GeoTIFF band metadata that identify VV and VH."
            )

    return sorted(candidates, key=lambda item: item.candidate_id), errors, warnings


def stage_auxiliary_uploads(uploaded_files: Iterable, root: Path, kind: str) -> tuple[dict[str, Path], list[str]]:
    """Save DEM/mask uploads and index them by canonical tile ID, never upload order."""
    root = Path(root)
    mapping: dict[str, Path] = {}
    errors: list[str] = []
    for uploaded in uploaded_files or []:
        key = canonical_tile_id(str(uploaded.name), auxiliary=True)
        if key in mapping:
            errors.append(f"Duplicate {kind} tile identifier: {key}.")
            continue
        path = save_upload(uploaded, root / kind / str(uploaded.name))
        mapping[key] = path
    return mapping, errors


def write_candidate_csv(
    path: Path,
    candidates: Sequence[SarCandidateUpload],
    dem_paths: Mapping[str, Path],
    mask_paths: Mapping[str, Path],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["candidate_id", "kind", "sar_path", "vv_path", "vh_path", "dem_path", "mask_path", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "candidate_id": candidate.candidate_id,
                    "kind": candidate.kind,
                    "sar_path": str(candidate.sar_path or ""),
                    "vv_path": str(candidate.vv_path or ""),
                    "vh_path": str(candidate.vh_path or ""),
                    "dem_path": str(dem_paths.get(candidate.candidate_id, "")),
                    "mask_path": str(mask_paths.get(candidate.candidate_id, "")),
                    "status": "ready",
                }
            )
    return path
