"""Sparse-flood crop supervision for segmentation training.

The transform operates on the fully stacked input (SAR plus optional DEM) and
its mask so every modality remains spatially aligned. It is intentionally
separate from generic augmentation profiles: controlled experiments can enable
this supervision while leaving the model, loss, optimiser, sampler, and other
augmentations unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np


MODE_NORMAL = 0
MODE_FLOOD_CENTERED = 1
MODE_HARD_BACKGROUND = 2

MODE_NAMES = {
    MODE_NORMAL: "normal",
    MODE_FLOOD_CENTERED: "flood-centred",
    MODE_HARD_BACKGROUND: "hard-background",
}


@dataclass(frozen=True)
class SparseCropMetadata:
    """Small integer metadata returned with a crop-supervised sample."""

    requested_mode: int
    applied_mode: int
    crop_size: int


class SparseFloodCropSupervision:
    """Apply a controlled mixture of full tiles and spatial zoom crops.

    The configured mixture is sampled only for tiles containing flood pixels.
    Empty tiles remain full-size normal samples, so hard-background examples
    come from flood scenes rather than unrelated empty scenes.
    """

    def __init__(
        self,
        target_size: int,
        crop_sizes: Sequence[int] = (256, 320, 384, 448),
        normal_fraction: float = 0.50,
        flood_centered_fraction: float = 0.25,
        hard_background_fraction: float = 0.25,
        attempts: int = 24,
        hard_background_max_fg_ratio: float = 0.001,
        min_valid_ratio: float = 0.50,
        ignore_index: int = 255,
    ) -> None:
        self.target_size = int(target_size)
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")

        cleaned_sizes = sorted({int(size) for size in crop_sizes if int(size) > 0})
        if not cleaned_sizes:
            raise ValueError("crop_sizes must contain at least one positive integer")
        self.crop_sizes = tuple(cleaned_sizes)

        fractions = np.asarray(
            [normal_fraction, flood_centered_fraction, hard_background_fraction],
            dtype=np.float64,
        )
        if np.any(fractions < 0.0):
            raise ValueError("sparse-crop fractions cannot be negative")
        total = float(fractions.sum())
        if total <= 0.0:
            raise ValueError("at least one sparse-crop fraction must be positive")
        self.mode_probabilities = tuple((fractions / total).tolist())

        self.attempts = max(int(attempts), 1)
        self.hard_background_max_fg_ratio = float(hard_background_max_fg_ratio)
        if not 0.0 <= self.hard_background_max_fg_ratio <= 1.0:
            raise ValueError("hard_background_max_fg_ratio must be between 0 and 1")
        self.min_valid_ratio = float(min_valid_ratio)
        if not 0.0 <= self.min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio must be between 0 and 1")
        self.ignore_index = int(ignore_index)

    @property
    def normal_fraction(self) -> float:
        return float(self.mode_probabilities[MODE_NORMAL])

    @property
    def flood_centered_fraction(self) -> float:
        return float(self.mode_probabilities[MODE_FLOOD_CENTERED])

    @property
    def hard_background_fraction(self) -> float:
        return float(self.mode_probabilities[MODE_HARD_BACKGROUND])

    def __repr__(self) -> str:
        return (
            "SparseFloodCropSupervision("
            f"target_size={self.target_size}, crop_sizes={self.crop_sizes}, "
            f"mix=({self.normal_fraction:.3f}, {self.flood_centered_fraction:.3f}, "
            f"{self.hard_background_fraction:.3f}), attempts={self.attempts}, "
            f"hard_background_max_fg_ratio={self.hard_background_max_fg_ratio:.6f})"
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image = np.asarray(image)
        mask = np.asarray(mask)
        if image.ndim != 3:
            raise ValueError(f"sparse crop expects an HWC image, got shape {image.shape}")
        if mask.ndim != 2:
            raise ValueError(f"sparse crop expects a 2D mask, got shape {mask.shape}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"image/mask shape mismatch: {image.shape[:2]} != {mask.shape}")

        flood_pixels = np.argwhere(mask == 1)
        # Empty tiles stay full-size. This prevents the hard-background branch
        # from turning into ordinary empty-scene oversampling.
        if flood_pixels.size == 0:
            image, mask = self._ensure_target_size(image, mask)
            return image, mask, SparseCropMetadata(MODE_NORMAL, MODE_NORMAL, 0)

        requested_mode = int(np.random.choice(
            [MODE_NORMAL, MODE_FLOOD_CENTERED, MODE_HARD_BACKGROUND],
            p=self.mode_probabilities,
        ))
        if requested_mode == MODE_NORMAL:
            image, mask = self._ensure_target_size(image, mask)
            return image, mask, SparseCropMetadata(requested_mode, MODE_NORMAL, 0)

        crop_size = self._choose_crop_size(mask.shape)
        if crop_size is None:
            image, mask = self._ensure_target_size(image, mask)
            return image, mask, SparseCropMetadata(requested_mode, MODE_NORMAL, 0)

        if requested_mode == MODE_FLOOD_CENTERED:
            crop = self._flood_centered_crop(image, mask, flood_pixels, crop_size)
        else:
            crop = self._hard_background_crop(image, mask, crop_size)

        if crop is None:
            image, mask = self._ensure_target_size(image, mask)
            return image, mask, SparseCropMetadata(requested_mode, MODE_NORMAL, 0)

        cropped_image, cropped_mask = crop
        cropped_image, cropped_mask = self._resize(cropped_image, cropped_mask)
        return cropped_image, cropped_mask, SparseCropMetadata(requested_mode, requested_mode, crop_size)

    def _choose_crop_size(self, mask_shape: Tuple[int, int]):
        h, w = int(mask_shape[0]), int(mask_shape[1])
        max_size = min(h, w)
        eligible = [size for size in self.crop_sizes if size < max_size]
        if not eligible:
            return None
        return int(np.random.choice(eligible))

    def _flood_centered_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        flood_pixels: np.ndarray,
        crop_size: int,
    ):
        h, w = mask.shape
        # Randomly place a flood anchor within the central half of the crop.
        # This keeps the positive target visible without always centring it.
        anchor_y, anchor_x = flood_pixels[np.random.randint(0, len(flood_pixels))]
        anchor_in_crop_y = int(np.random.randint(crop_size // 4, max(crop_size * 3 // 4, crop_size // 4 + 1)))
        anchor_in_crop_x = int(np.random.randint(crop_size // 4, max(crop_size * 3 // 4, crop_size // 4 + 1)))
        y0 = int(np.clip(int(anchor_y) - anchor_in_crop_y, 0, h - crop_size))
        x0 = int(np.clip(int(anchor_x) - anchor_in_crop_x, 0, w - crop_size))
        cropped_image, cropped_mask = self._crop(image, mask, y0, x0, crop_size)
        if not np.any(cropped_mask == 1):
            return None
        if self._valid_ratio(cropped_mask) < self.min_valid_ratio:
            return None
        return cropped_image, cropped_mask

    def _hard_background_crop(self, image: np.ndarray, mask: np.ndarray, crop_size: int):
        h, w = mask.shape
        max_y = h - crop_size
        max_x = w - crop_size
        background_pixels = np.argwhere(mask == 0)
        if background_pixels.size == 0:
            return None

        best = None
        best_fg_ratio = float("inf")
        best_valid_ratio = -1.0
        for _ in range(self.attempts):
            # Anchor candidates on valid background pixels, with a random
            # location inside the crop, so the search covers the full tile.
            anchor_y, anchor_x = background_pixels[np.random.randint(0, len(background_pixels))]
            anchor_in_crop_y = int(np.random.randint(0, crop_size))
            anchor_in_crop_x = int(np.random.randint(0, crop_size))
            y0 = int(np.clip(int(anchor_y) - anchor_in_crop_y, 0, max_y))
            x0 = int(np.clip(int(anchor_x) - anchor_in_crop_x, 0, max_x))
            cropped_image, cropped_mask = self._crop(image, mask, y0, x0, crop_size)
            valid_ratio = self._valid_ratio(cropped_mask)
            if valid_ratio < self.min_valid_ratio:
                continue
            fg_ratio = self._foreground_ratio(cropped_mask)
            if fg_ratio < best_fg_ratio or (fg_ratio == best_fg_ratio and valid_ratio > best_valid_ratio):
                best = (cropped_image, cropped_mask)
                best_fg_ratio = fg_ratio
                best_valid_ratio = valid_ratio
            if fg_ratio <= self.hard_background_max_fg_ratio:
                break

        if best is None or best_fg_ratio > self.hard_background_max_fg_ratio:
            return None
        return best

    @staticmethod
    def _crop(image: np.ndarray, mask: np.ndarray, y0: int, x0: int, size: int):
        return image[y0:y0 + size, x0:x0 + size, ...], mask[y0:y0 + size, x0:x0 + size]

    def _valid_ratio(self, mask: np.ndarray) -> float:
        return float(np.count_nonzero(mask != self.ignore_index) / max(mask.size, 1))

    def _foreground_ratio(self, mask: np.ndarray) -> float:
        valid = mask != self.ignore_index
        valid_pixels = int(np.count_nonzero(valid))
        if valid_pixels == 0:
            return 1.0
        return float(np.count_nonzero((mask == 1) & valid) / valid_pixels)

    def _ensure_target_size(self, image: np.ndarray, mask: np.ndarray):
        if image.shape[0] == self.target_size and image.shape[1] == self.target_size:
            return image, mask
        return self._resize(image, mask)

    def _resize(self, image: np.ndarray, mask: np.ndarray):
        size = (self.target_size, self.target_size)
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        if resized_image.ndim == 2:
            resized_image = resized_image[..., None]
        resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return resized_image.astype(image.dtype, copy=False), resized_mask.astype(mask.dtype, copy=False)
