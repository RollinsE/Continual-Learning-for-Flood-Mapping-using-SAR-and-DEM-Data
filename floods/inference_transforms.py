"""Dependency-light transforms used by deployment inference.

Training augmentation is implemented with Albumentations in :mod:`floods.prepare`.
Deployment only needs deterministic clipping, normalisation, and conversion from
HWC NumPy arrays to CHW PyTorch tensors, so importing the training augmentation
stack here would be unnecessary and fragile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch


def _channel_values(value: Union[float, Sequence[float]], channels: int) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 0:
        values = np.repeat(values.reshape(1), channels)
    elif values.size < channels:
        values = np.pad(values.reshape(-1), (0, channels - values.size), mode="edge")
    else:
        values = values.reshape(-1)[:channels]
    return values.reshape(1, 1, channels)


@dataclass(frozen=True)
class InferenceTransform:
    """Apply saved training normalisation and return a tensor-ready sample."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    clip_min: tuple[float, ...]
    clip_max: tuple[float, ...]
    normalization_mode: str = "stats"

    def __call__(self, *, image: np.ndarray, mask: np.ndarray | None = None, **kwargs) -> dict:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image = image[..., None]
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image array, got shape {image.shape}")

        channels = int(image.shape[-1])
        mean = _channel_values(self.mean, channels)
        std = np.maximum(_channel_values(self.std, channels), 1e-6)
        clip_min = _channel_values(self.clip_min, channels)
        clip_max = _channel_values(self.clip_max, channels)

        missing = np.isnan(image)
        pos_fill = float(np.max(clip_max))
        neg_fill = float(np.min(clip_min))
        image = np.nan_to_num(image, nan=0.0, posinf=pos_fill, neginf=neg_fill)
        image = np.clip(image, clip_min, clip_max).astype(np.float32, copy=False)

        mode = str(self.normalization_mode or "stats").strip().lower().replace("-", "_")
        if mode in {"robust_percentile", "notebook_robust", "robust_minmax"}:
            scale = np.maximum(clip_max - clip_min, 1e-6)
            image = np.clip((image - clip_min) / scale, 0.0, 1.0)

        image = (image - mean) / std
        image = np.nan_to_num(image, nan=0.0, posinf=30.0, neginf=-30.0)
        image = np.clip(image, -30.0, 30.0).astype(np.float32, copy=False)
        # Missing inputs are set to 0 (the training mean) after standardisation,
        # matching floods.transforms; their outputs are masked as no-data downstream.
        image[missing] = 0.0

        # np.moveaxis can return negative strides; materialise a contiguous array
        # before creating the tensor.
        tensor = torch.from_numpy(np.ascontiguousarray(np.moveaxis(image, -1, 0)))
        result = {"image": tensor}
        if mask is not None:
            result["mask"] = torch.from_numpy(np.ascontiguousarray(mask))
        return result


def eval_transforms(
    mean: tuple,
    std: tuple,
    clip_min: tuple,
    clip_max: tuple,
    normalization_mode: str = "stats",
) -> InferenceTransform:
    """Build deterministic deployment transforms without Albumentations/SciPy."""
    return InferenceTransform(
        mean=tuple(float(v) for v in mean),
        std=tuple(float(v) for v in std),
        clip_min=tuple(float(v) for v in clip_min),
        clip_max=tuple(float(v) for v in clip_max),
        normalization_mode=str(normalization_mode or "stats"),
    )


# --------------------------------------------------------------------------- SAR units

SAR_UNIT_ALIASES = {"db": "db10", "decibel": "db10", "db10": "db10", "linear": "linear", "power": "linear",
                    "log1p": "log1p", "auto": "auto"}


def canonical_sar_units(value: str | None, default: str = "auto") -> str:
    key = str(value or default).strip().lower().replace("-", "_")
    if key not in SAR_UNIT_ALIASES:
        raise ValueError(f"Unknown SAR units '{value}'. Use one of: auto, linear, db, log1p")
    return SAR_UNIT_ALIASES[key]


def sar_to_db(values: np.ndarray, source: str) -> np.ndarray:
    """Convert SAR values in ``source`` units to dB, preserving NaN."""
    values = np.asarray(values, dtype=np.float32)
    source = canonical_sar_units(source, "db10")
    if source == "db10":
        return values
    with np.errstate(invalid="ignore", divide="ignore"):
        if source == "log1p":
            linear = np.power(10.0, values) - 1.0
        elif source == "linear":
            linear = values
        else:
            raise ValueError(f"Unsupported SAR units: {source}")
        return (10.0 * np.log10(np.maximum(linear, 1e-7))).astype(np.float32)


def db_to_sar(values_db: np.ndarray, source: str) -> np.ndarray:
    """Convert dB to ``source`` units, preserving NaN."""
    values_db = np.asarray(values_db, dtype=np.float32)
    source = canonical_sar_units(source, "db10")
    if source == "db10":
        return values_db
    linear = np.power(10.0, values_db / 10.0)
    if source == "log1p":
        return np.log10(1.0 + linear).astype(np.float32)
    if source == "linear":
        return linear.astype(np.float32)
    raise ValueError(f"Unsupported SAR units: {source}")


def detect_sar_units(*bands: np.ndarray) -> str:
    """Guess whether uploaded backscatter is linear power or dB.

    Linear sigma0/gamma0 is non-negative and mostly below ~1; dB values are mostly
    negative (typically -30 to +5). Only finite, non-zero pixels are considered.
    """
    values = np.concatenate([np.asarray(b, dtype=np.float32).reshape(-1) for b in bands])
    values = values[np.isfinite(values) & (values != 0.0)]
    if values.size == 0:
        raise ValueError("SAR input contains no finite, non-zero pixels")
    negative_fraction = float(np.mean(values < 0.0))
    if negative_fraction > 0.5:
        return "db10"
    if negative_fraction < 0.01 and float(np.percentile(values, 99)) < 10.0:
        return "linear"
    raise ValueError(
        "Could not tell whether the SAR input is linear or dB "
        f"(negative fraction {negative_fraction:.2f}). Pass the units explicitly (--input-sar-units linear|db)."
    )


def prepare_sar_input(vv: np.ndarray, vh: np.ndarray, input_units: str, model_units: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Mark no-data as NaN and convert uploaded VV/VH to the units the model was trained on.

    No-data: non-finite values, and non-positive values for linear inputs (Sentinel-1 uses 0
    outside the swath). Returns (vv, vh, detected_input_units).
    """
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    units = canonical_sar_units(input_units, "auto")
    if units == "auto":
        units = detect_sar_units(vv, vh)
    model_units = canonical_sar_units(model_units, "db10")
    converted = []
    for band in (vv, vh):
        band = np.where(np.isfinite(band), band, np.nan).astype(np.float32)
        if units in {"linear", "log1p"}:
            band = np.where(band > 0.0, band, np.nan).astype(np.float32)
        converted.append(db_to_sar(sar_to_db(band, units), model_units))
    return converted[0], converted[1], units
