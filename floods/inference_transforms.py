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
