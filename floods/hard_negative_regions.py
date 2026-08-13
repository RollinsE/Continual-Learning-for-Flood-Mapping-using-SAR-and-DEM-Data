"""Audit-guided hard-negative region mining and crop supervision.

This module targets the specific precision failures found by an existing model.
It mines high-confidence false-positive regions from a labelled training split,
then reuses the mined crop coordinates during a controlled fine-tune.  Unlike
random background cropping, every crop is anchored on an actual model error.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from floods.evaluation import BinaryThresholdSweep, load_checkpoint_state
from floods.hard_examples import _tile_keys
from floods.utils.common import get_logger
from floods.utils.console import progress_iter

LOG = get_logger(__name__)

MODE_NORMAL = 0
MODE_AUDIT_HARD_NEGATIVE = 3


@dataclass(frozen=True)
class CropMetadata:
    requested_mode: int
    applied_mode: int
    crop_size: int


def _event_id(name: str) -> str:
    match = re.search(r"(EMSR\d+)", name)
    return match.group(1) if match else "unknown"


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clamp_crop(center_y: float, center_x: float, size: int, height: int, width: int) -> Tuple[int, int]:
    size = int(min(size, height, width))
    y0 = int(round(center_y - size / 2.0))
    x0 = int(round(center_x - size / 2.0))
    y0 = int(np.clip(y0, 0, max(height - size, 0)))
    x0 = int(np.clip(x0, 0, max(width - size, 0)))
    return y0, x0


def _box_iou(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    ay, ax, asz = a
    by, bx, bsz = b
    ay2, ax2 = ay + asz, ax + asz
    by2, bx2 = by + bsz, bx + bsz
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter = inter_h * inter_w
    union = asz * asz + bsz * bsz - inter
    return float(inter / union) if union > 0 else 0.0


def _connected_components(mask: np.ndarray) -> List[Dict[str, Any]]:
    mask_u8 = mask.astype(np.uint8, copy=False)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components: List[Dict[str, Any]] = []
    for label_id in range(1, count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[label_id]
        components.append({"label": label_id, "x": x, "y": y, "w": w, "h": h, "area": area, "cx": float(cx), "cy": float(cy), "labels": labels})
    return components


def _candidate_for_component(
    component: Dict[str, Any],
    prob: np.ndarray,
    target: np.ndarray,
    threshold: float,
    crop_sizes: Sequence[int],
    min_fp_pixels: int,
    max_label_fg_ratio: float,
    min_valid_ratio: float,
) -> Optional[Dict[str, Any]]:
    height, width = target.shape
    valid = target != 255
    background = (target == 0) & valid
    hard = (prob >= threshold) & background
    labels = component["labels"]
    component_mask = labels == int(component["label"])

    preferred = max(int(component["w"]), int(component["h"]), 1)
    eligible = [int(s) for s in crop_sizes if int(s) <= min(height, width)]
    if not eligible:
        return None
    size = next((s for s in sorted(eligible) if s >= preferred), max(eligible))

    # Anchor on the highest-probability pixel in the component rather than only
    # the centroid. This keeps the most confident model error inside the crop.
    component_prob = np.where(component_mask, prob, -1.0)
    max_index = int(np.argmax(component_prob))
    peak_y, peak_x = np.unravel_index(max_index, prob.shape)
    center_y = (float(component["cy"]) + float(peak_y)) / 2.0
    center_x = (float(component["cx"]) + float(peak_x)) / 2.0
    y0, x0 = _clamp_crop(center_y, center_x, size, height, width)

    crop_valid = valid[y0:y0 + size, x0:x0 + size]
    valid_pixels = int(np.count_nonzero(crop_valid))
    valid_ratio = _safe_div(valid_pixels, size * size)
    if valid_ratio < min_valid_ratio or valid_pixels <= 0:
        return None

    crop_target = target[y0:y0 + size, x0:x0 + size]
    crop_hard = hard[y0:y0 + size, x0:x0 + size]
    crop_prob = prob[y0:y0 + size, x0:x0 + size]
    fp_pixels = int(np.count_nonzero(crop_hard))
    if fp_pixels < int(min_fp_pixels):
        return None
    fg_pixels = int(np.count_nonzero((crop_target == 1) & crop_valid))
    fg_ratio = _safe_div(fg_pixels, valid_pixels)
    if fg_ratio > float(max_label_fg_ratio):
        return None

    hard_probs = crop_prob[crop_hard]
    mean_fp_probability = float(hard_probs.mean()) if hard_probs.size else 0.0
    max_probability = float(hard_probs.max()) if hard_probs.size else 0.0
    fp_ratio = _safe_div(fp_pixels, valid_pixels)
    score = float(fp_pixels * max(mean_fp_probability - threshold, 1e-4))
    return {
        "x0": int(x0),
        "y0": int(y0),
        "crop_size": int(size),
        "component_area": int(component["area"]),
        "fp_pixels": fp_pixels,
        "fp_ratio": fp_ratio,
        "fg_pixels": fg_pixels,
        "fg_ratio": fg_ratio,
        "valid_ratio": valid_ratio,
        "mean_fp_probability": mean_fp_probability,
        "max_probability": max_probability,
        "score": score,
    }


def _select_non_overlapping(candidates: List[Dict[str, Any]], max_regions: int, nms_iou: float) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: (row["score"], row["fp_pixels"]), reverse=True):
        box = (int(candidate["y0"]), int(candidate["x0"]), int(candidate["crop_size"]))
        if any(_box_iou(box, (int(row["y0"]), int(row["x0"]), int(row["crop_size"]))) > nms_iou for row in selected):
            continue
        selected.append(candidate)
        if len(selected) >= int(max_regions):
            break
    return selected


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, y, index


def mine_hard_negative_regions(
    config: Any,
    checkpoint_path: Path,
    output_dir: Path,
    split: str = "train",
    threshold: float = 0.60,
    crop_sizes: Sequence[int] = (256, 320, 384),
    min_component_area: int = 64,
    min_fp_pixels: int = 128,
    max_label_fg_ratio: float = 0.001,
    min_valid_ratio: float = 0.50,
    max_regions_per_tile: int = 3,
    nms_iou: float = 0.30,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Mine high-confidence false-positive crop coordinates from a labelled split."""
    from floods.datasets.flood import FloodDataset, RGBFloodDataset
    from floods.eval_collate import pad_segmentation_batch
    from floods.normalization import describe_stats, load_normalization_stats
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.utils.ml import seed_everything, seed_worker

    if split != "train":
        LOG.warning("Hard-negative mining is normally run on split=train. Using split=%s as requested.", split)
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    crop_sizes = sorted({int(v) for v in crop_sizes if int(v) > 0})
    if not crop_sizes:
        raise ValueError("crop_sizes must contain at least one positive integer")

    seed_everything(config.seed, deterministic=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not bool(config.trainer.cpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = bool(config.trainer.amp and use_cuda)

    dataset, modalities, use_rgb = prepare_evaluation_dataset(config, split=split)
    indexed: Dataset = _IndexedDataset(dataset)
    if max_samples is not None and int(max_samples) > 0:
        indexed = torch.utils.data.Subset(indexed, list(range(min(int(max_samples), len(dataset)))))
    loader = DataLoader(indexed, batch_size=config.trainer.batch_size, shuffle=False,
                        num_workers=config.trainer.num_workers, worker_init_fn=seed_worker,
                        collate_fn=pad_segmentation_batch)

    model = prepare_model(config=config, num_classes=1, stage="eval")
    model.load_state_dict(load_checkpoint_state(Path(checkpoint_path)), strict=not config.model.multibranch)
    model = model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    tiles_with_fp = 0
    LOG.info("Mining hard-negative regions from: %s", checkpoint_path)
    LOG.info("Dataset: %s split, %d samples | threshold=%.2f | crop sizes=%s", split, len(indexed), threshold, crop_sizes)
    with torch.no_grad():
        for x, y, index in progress_iter(loader, desc=f"Mine hard negatives {split}", unit="batch", colour="red"):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0).to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x)
            logits = BinaryThresholdSweep._main_prediction(out).detach().float()
            prob_batch = torch.sigmoid(BinaryThresholdSweep._squeeze_logits(logits)).cpu().numpy()
            target_batch = y.detach().cpu().numpy()
            if target_batch.ndim == 4 and target_batch.shape[1] == 1:
                target_batch = target_batch[:, 0]
            indices = index.detach().cpu().numpy().tolist() if isinstance(index, torch.Tensor) else list(index)

            for b, dataset_index in enumerate(indices):
                dataset_index = int(dataset_index)
                target = target_batch[b].astype(np.uint8)
                prob = prob_batch[b].astype(np.float32)
                valid = target != 255
                hard_mask = (prob >= threshold) & (target == 0) & valid
                components = [c for c in _connected_components(hard_mask) if int(c["area"]) >= int(min_component_area)]
                candidates: List[Dict[str, Any]] = []
                for component in components:
                    candidate = _candidate_for_component(component, prob, target, threshold, crop_sizes,
                                                         min_fp_pixels, max_label_fg_ratio, min_valid_ratio)
                    if candidate is not None:
                        candidates.append(candidate)
                selected = _select_non_overlapping(candidates, max_regions_per_tile, nms_iou)
                if not selected:
                    continue
                tiles_with_fp += 1
                image_path = Path(dataset.image_files[dataset_index])
                mask_path = Path(dataset.label_files[dataset_index])
                dem_path = Path(dataset.dem_files[dataset_index]) if getattr(dataset, "_include_dem", False) else None
                for rank, candidate in enumerate(selected, start=1):
                    rows.append({
                        "split": split,
                        "index": dataset_index,
                        "tile_id": image_path.stem,
                        "file": image_path.name,
                        "event_id": _event_id(image_path.name),
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "dem_path": str(dem_path) if dem_path else "",
                        "threshold": threshold,
                        "region_rank": rank,
                        **candidate,
                    })

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "hard_negative_regions.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples": int(len(indexed)),
        "threshold": threshold,
        "crop_sizes": crop_sizes,
        "min_component_area": int(min_component_area),
        "min_fp_pixels": int(min_fp_pixels),
        "max_label_fg_ratio": float(max_label_fg_ratio),
        "min_valid_ratio": float(min_valid_ratio),
        "max_regions_per_tile": int(max_regions_per_tile),
        "nms_iou": float(nms_iou),
        "tiles_with_regions": int(tiles_with_fp),
        "regions": int(len(manifest)),
        "manifest": str(manifest_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOG.info("Hard-negative mining written to: %s", output_dir)
    LOG.info("Mined %d regions from %d/%d tiles", len(manifest), tiles_with_fp, len(indexed))
    if manifest.empty:
        LOG.warning("No hard-negative regions met the criteria. Lower --threshold or --min-fp-pixels, or increase --max-label-fg-ratio slightly.")
    return summary


class AuditGuidedHardNegativeCropSupervision:
    """Apply crops from a mined hard-negative region manifest."""

    def __init__(self, manifest_path: str | Path, target_size: int, probability: float = 1.0,
                 ignore_index: int = 255) -> None:
        self.manifest_path = Path(manifest_path).expanduser()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Hard-negative region manifest does not exist: {self.manifest_path}")
        frame = pd.read_csv(self.manifest_path)
        required = {"x0", "y0", "crop_size"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Hard-negative manifest is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"Hard-negative manifest is empty: {self.manifest_path}")
        self.target_size = int(target_size)
        self.probability = float(probability)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("hard-negative crop probability must be between 0 and 1")
        self.ignore_index = int(ignore_index)
        self.regions: Dict[str, List[Dict[str, Any]]] = {}
        for _, row in frame.iterrows():
            record = row.to_dict()
            keys = set()
            for column in ("image_path", "mask_path", "file", "tile_id"):
                if column in record:
                    keys.update(_tile_keys(record[column]))
            for key in keys:
                self.regions.setdefault(key, []).append(record)
        if not self.regions:
            raise ValueError(f"No usable tile identifiers found in manifest: {self.manifest_path}")

    def matching_regions(self, sample_path: str | Path) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        seen = set()
        for key in _tile_keys(sample_path):
            for record in self.regions.get(key, []):
                marker = (int(record["x0"]), int(record["y0"]), int(record["crop_size"]))
                if marker not in seen:
                    matches.append(record)
                    seen.add(marker)
        return matches

    def has_regions(self, sample_path: str | Path) -> bool:
        return bool(self.matching_regions(sample_path))

    def __call__(self, image: np.ndarray, mask: np.ndarray, sample_path: str | Path):
        matches = self.matching_regions(sample_path)
        if not matches or np.random.random() > self.probability:
            return image, mask, CropMetadata(MODE_NORMAL, MODE_NORMAL, 0)
        weights = np.asarray([max(float(row.get("score", row.get("fp_pixels", 1.0))), 1e-8) for row in matches], dtype=np.float64)
        weights = weights / weights.sum()
        row = matches[int(np.random.choice(len(matches), p=weights))]
        size = int(row["crop_size"])
        y0, x0 = int(row["y0"]), int(row["x0"])
        height, width = mask.shape
        if size <= 0 or y0 < 0 or x0 < 0 or y0 + size > height or x0 + size > width:
            return image, mask, CropMetadata(MODE_AUDIT_HARD_NEGATIVE, MODE_NORMAL, 0)
        cropped_image = image[y0:y0 + size, x0:x0 + size, ...]
        cropped_mask = mask[y0:y0 + size, x0:x0 + size]
        if cropped_image.size == 0 or cropped_mask.size == 0:
            return image, mask, CropMetadata(MODE_AUDIT_HARD_NEGATIVE, MODE_NORMAL, 0)
        target = (self.target_size, self.target_size)
        if size != self.target_size:
            cropped_image = cv2.resize(cropped_image, target, interpolation=cv2.INTER_LINEAR)
            if cropped_image.ndim == 2:
                cropped_image = cropped_image[..., None]
            cropped_mask = cv2.resize(cropped_mask, target, interpolation=cv2.INTER_NEAREST)
        return (cropped_image.astype(image.dtype, copy=False),
                cropped_mask.astype(mask.dtype, copy=False),
                CropMetadata(MODE_AUDIT_HARD_NEGATIVE, MODE_AUDIT_HARD_NEGATIVE, size))


def manifest_matching_indices(label_files: Sequence[str], manifest_path: str | Path) -> set[int]:
    supervisor = AuditGuidedHardNegativeCropSupervision(manifest_path=manifest_path, target_size=1, probability=0.0)
    return {idx for idx, path in enumerate(label_files) if supervisor.has_regions(path)}


def prepare_hard_negative_region_sampler(dataset: Any, manifest_path: str | Path,
                                         weight: float = 4.0, max_fraction: float = 0.35,
                                         samples_multiplier: float = 1.0) -> WeightedRandomSampler:
    """Oversample tiles that contain mined hard-negative regions."""
    weight = float(weight)
    max_fraction = float(max_fraction)
    if weight <= 1.0:
        raise ValueError("hard_negative_region_weight must be greater than 1.0")
    if not 0.0 < max_fraction < 1.0:
        raise ValueError("hard_negative_region_max_fraction must be between 0 and 1")
    indices = manifest_matching_indices(dataset.label_files, manifest_path)
    if not indices:
        raise ValueError("Hard-negative manifest does not match any current training tiles")
    natural_fraction = _safe_div(len(indices), len(dataset))
    if max_fraction <= natural_fraction:
        LOG.warning(
            "hard_negative_region_max_fraction=%.2f is at or below the natural region-tile prevalence %.2f; "
            "the cap will not oversample hard-negative tiles and may downweight them.",
            max_fraction, natural_fraction,
        )
    weights = np.ones(len(dataset), dtype=np.float64)
    hard_mask = np.zeros(len(dataset), dtype=bool)
    hard_mask[list(indices)] = True
    weights[hard_mask] *= weight
    hard_mass = float(weights[hard_mask].sum())
    normal_mass = float(weights[~hard_mask].sum())
    current_fraction = _safe_div(hard_mass, hard_mass + normal_mass)
    if current_fraction > max_fraction and normal_mass > 0:
        target_hard_mass = (max_fraction * normal_mass) / (1.0 - max_fraction)
        weights[hard_mask] *= target_hard_mass / hard_mass
        hard_mass = float(weights[hard_mask].sum())
        current_fraction = _safe_div(hard_mass, hard_mass + normal_mass)
    multiplier = float(samples_multiplier or 1.0)
    if multiplier <= 0:
        raise ValueError("weighted_samples_multiplier must be greater than 0")
    num_samples = max(1, int(round(len(dataset) * multiplier)))
    LOG.info("Audit-guided hard-negative region sampling: %d samples per epoch | region tiles=%d/%d | effective region mass=%.2f | weight=%.2f",
             num_samples, int(np.count_nonzero(hard_mask)), len(dataset), current_fraction, weight)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
