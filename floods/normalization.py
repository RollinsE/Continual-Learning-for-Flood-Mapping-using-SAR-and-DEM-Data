from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from floods.utils.console import progress_iter

from floods.derived_features import read_processed_modalities
from floods.modalities import canonicalize_modalities
from floods.utils.common import get_logger
from floods.utils.gis import imread

LOG = get_logger(__name__)


@dataclass
class ChannelStats:
    channel: str
    count: int
    q_min: float
    q_max: float
    clip_min: float
    clip_max: float
    mean: float
    std: float
    raw_mean: float
    raw_std: float
    robust_mean: float
    robust_std: float
    raw_min: float
    raw_max: float


def _as_channels_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3:
        # project imread normally returns channels-first; GeoTIFF reads from rasterio are C,H,W.
        if arr.shape[0] <= 8 and arr.shape[-1] > 8:
            return np.moveaxis(arr, 0, -1)
        return arr
    raise ValueError(f"Unsupported raster shape: {arr.shape}")


def _sample_pixels(arr: np.ndarray, max_pixels: int, rng: np.random.Generator, valid_mask: np.ndarray | None = None) -> np.ndarray:
    arr = _as_channels_last(arr).astype(np.float32, copy=False)
    flat = arr.reshape(-1, arr.shape[-1])
    finite = np.all(np.isfinite(flat), axis=1)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask).reshape(-1).astype(bool)
        if valid_mask.shape[0] == finite.shape[0]:
            finite = finite & valid_mask
    flat = flat[finite]
    if flat.size == 0:
        return flat.reshape(0, arr.shape[-1])
    if max_pixels and len(flat) > max_pixels:
        idx = rng.choice(len(flat), size=max_pixels, replace=False)
        flat = flat[idx]
    return flat


def _read_modalities(processed_data_dir: Path, split: str, stem: str, modalities: Sequence[str]) -> np.ndarray:
    return read_processed_modalities(processed_data_dir, split, stem, modalities)


def _mask_valid_pixels(processed_data_dir: Path, split: str, stem: str) -> np.ndarray | None:
    mask_path = processed_data_dir / split / "mask" / f"{stem}.tif"
    if not mask_path.exists():
        return None
    mask = imread(mask_path, channels_first=True).squeeze()
    return mask.reshape(-1) != 255


def fit_normalization_stats(
    processed_data_dir: Path,
    output_file: Path,
    split: str = "train",
    input_modalities: Sequence[str] = ("vv", "vh"),
    q_min: float = 1.0,
    q_max: float = 99.0,
    max_pixels_per_file: int = 4096,
    seed: int = 1337,
    ignore_mask_255: bool = True,
    preserve_channel_stats_from: Path | None = None,
    include_events: Sequence[str] | None = None,
    exclude_events: Sequence[str] | None = None,
) -> dict:
    """Fit robust channel-normalization statistics from processed tiles.

    Statistics are fitted on one split, normally train, using sampled finite pixels.
    The output JSON stores raw clipping percentiles and mean/std after clipping.
    """
    processed_data_dir = Path(processed_data_dir)
    output_file = Path(output_file)
    modalities = canonicalize_modalities(input_modalities)
    sar_paths = sorted((processed_data_dir / split / "sar").glob("*.tif"))
    include = {str(v).upper() for v in (include_events or [])}
    exclude = {str(v).upper() for v in (exclude_events or [])}
    if include or exclude:
        import re
        filtered = []
        for path in sar_paths:
            match = re.search(r"(EMSR\d+)", path.name, flags=re.IGNORECASE)
            event = match.group(1).upper() if match else "UNKNOWN"
            if include and event not in include:
                continue
            if event in exclude:
                continue
            filtered.append(path)
        sar_paths = filtered
    if not sar_paths:
        raise FileNotFoundError(f"No SAR tiles found under {processed_data_dir / split / 'sar'}")

    rng = np.random.default_rng(seed)
    collected: List[np.ndarray] = []
    names = [p.stem for p in sar_paths]
    max_pixels_per_file = int(max_pixels_per_file or 0)
    for stem in progress_iter(names, desc=f"Fit normalization {split}", unit="tile", colour="green"):
        arr = _read_modalities(processed_data_dir, split, stem, modalities=modalities)
        valid = _mask_valid_pixels(processed_data_dir, split, stem) if ignore_mask_255 else None
        samples = _sample_pixels(arr, max_pixels=max_pixels_per_file, rng=rng, valid_mask=valid)
        if samples.size == 0:
            continue
        collected.append(samples)

    if not collected:
        raise RuntimeError("No finite pixels were available for normalization fitting")
    data = np.concatenate(collected, axis=0).astype(np.float32, copy=False)
    stats = []
    eps = 1e-6
    for i, name in enumerate(modalities):
        values = data[:, i]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise RuntimeError(f"No finite pixels available for channel {name}")
        cmin = float(np.percentile(values, q_min))
        cmax = float(np.percentile(values, q_max))
        if not math.isfinite(cmin) or not math.isfinite(cmax) or cmax <= cmin:
            cmin = float(np.nanmin(values))
            cmax = float(np.nanmax(values))
        clipped = np.clip(values, cmin, cmax)
        mean = float(np.mean(clipped))
        std = float(np.std(clipped))
        raw_mean = float(np.mean(values))
        raw_std = float(np.std(values))
        if not math.isfinite(std) or std < eps:
            std = 1.0
        if not math.isfinite(raw_std) or raw_std < eps:
            raw_std = 1.0
        scale = max(cmax - cmin, eps)
        robust = np.clip((values - cmin) / scale, 0.0, 1.0)
        robust_mean = float(np.mean(robust))
        robust_std = float(np.std(robust))
        if not math.isfinite(robust_std) or robust_std < eps:
            robust_std = 1.0
        stats.append(ChannelStats(
            channel=name,
            count=int(values.size),
            q_min=float(q_min),
            q_max=float(q_max),
            clip_min=cmin,
            clip_max=cmax,
            mean=mean,
            std=std,
            raw_mean=raw_mean,
            raw_std=raw_std,
            robust_mean=robust_mean,
            robust_std=robust_std,
            raw_min=float(np.nanmin(values)),
            raw_max=float(np.nanmax(values)),
        ).__dict__)

    preserved_channels: list[str] = []
    preserve_source = None
    if preserve_channel_stats_from is not None:
        preserve_path = Path(preserve_channel_stats_from)
        if not preserve_path.exists():
            raise FileNotFoundError(f"Preserved normalization stats file not found: {preserve_path}")
        with preserve_path.open("r", encoding="utf-8") as f:
            preserve_payload = json.load(f)
        preserve_by_channel = {
            str(item.get("channel", "")).strip().lower(): item
            for item in preserve_payload.get("channels", [])
            if item.get("channel")
        }
        merged_stats = []
        for item in stats:
            channel = str(item["channel"]).lower()
            if channel in preserve_by_channel:
                preserved = dict(preserve_by_channel[channel])
                preserved["channel"] = channel
                merged_stats.append(preserved)
                preserved_channels.append(channel)
            else:
                merged_stats.append(item)
        stats = merged_stats
        preserve_source = str(preserve_path)
        LOG.info(
            "Preserved existing normalization statistics for channels %s from %s",
            preserved_channels,
            preserve_path,
        )

    payload = {
        "schema_version": 2,
        "processed_data_dir": str(processed_data_dir),
        "split": split,
        "input_modalities": modalities,
        "files_used": len(sar_paths),
        "max_pixels_per_file": max_pixels_per_file,
        "seed": int(seed),
        "preserve_channel_stats_from": preserve_source,
        "preserved_channels": preserved_channels,
        "include_events": sorted(include),
        "exclude_events": sorted(exclude),
        "channels": stats,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOG.info("Normalization statistics written to: %s", output_file)
    for item in stats:
        LOG.info("%s: clip[%.4f, %.4f] mean=%.4f std=%.4f count=%d",
                 item["channel"], item["clip_min"], item["clip_max"], item["mean"], item["std"], item["count"])
    return payload


def load_normalization_stats(path: Path, input_modalities: Sequence[str], mode: str = "stats") -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    by_channel = {str(item["channel"]).lower(): item for item in payload.get("channels", [])}
    means, stds, clip_min, clip_max = [], [], [], []
    mode = str(mode or "stats").lower()
    for channel in canonicalize_modalities(input_modalities):
        if channel not in by_channel:
            raise KeyError(f"Normalization stats file {path} has no channel '{channel}'. Available: {sorted(by_channel)}")
        item = by_channel[channel]
        if mode in {"robust_percentile", "note" + "book_robust"}:
            # Robust percentile mode: min-max scale each clipped channel to [0, 1],
            # then normalize using the fitted training mean and standard deviation.
            # Older stats files fall back to clipped mean/std so they remain loadable.
            means.append(float(item.get("raw_mean", item.get("mean", 0.5))))
            stds.append(float(item.get("raw_std", item.get("std", 0.5))))
        elif mode == "robust_minmax":
            # Cleaner variant: robust min-max scale to [0, 1], then normalize
            # using mean/std of that scaled distribution.
            means.append(float(item.get("robust_mean", item.get("mean", 0.5))))
            stds.append(float(item.get("robust_std", item.get("std", 0.5))))
        else:
            means.append(float(item["mean"]))
            stds.append(float(item["std"]))
        clip_min.append(float(item["clip_min"]))
        clip_max.append(float(item["clip_max"]))
    return tuple(means), tuple(stds), tuple(clip_min), tuple(clip_max)

def describe_stats(path: Path) -> str:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    lines = [f"split={payload.get('split')} modalities={payload.get('input_modalities')} files={payload.get('files_used')}"]
    for item in payload.get("channels", []):
        lines.append(
            f"{item['channel']}: clip[{item['clip_min']:.4f}, {item['clip_max']:.4f}] "
            f"mean={item['mean']:.4f} std={item['std']:.4f}"
            + (f" robust_mean={item['robust_mean']:.4f} robust_std={item['robust_std']:.4f}" if 'robust_mean' in item else "")
        )
    return " | ".join(lines)
