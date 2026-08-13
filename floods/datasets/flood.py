from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from floods.datasets.base import DatasetBase
from floods.modalities import DERIVED_MODALITIES, canonicalize_modalities
from floods.sparse_crops import SparseFloodCropSupervision
from floods.hard_negative_regions import AuditGuidedHardNegativeCropSupervision
from floods.hard_positive_regions import AuditGuidedHardPositiveCropSupervision
from floods.utils.gis import imread


class FloodDataset(DatasetBase):

    _name = "flood"
    _categories = {0: "background", 1: "flood"}
    _palette = {0: (0, 0, 0), 1: (255, 255, 255), 255: (255, 0, 255)}
    _ignore_index = 255
    # Legacy channel statistics. Derived-feature runs must use train-fitted stats.
    _mean = (4.9329374e-02, 1.1776519e-02, 1.4241237e+02)
    _std = (3.91287043e-02, 1.03687926e-02, 8.11010422e+01)

    def __init__(self,
                 path: Path,
                 subset: str = "train",
                 include_dem: bool = False,
                 input_modalities: Optional[Sequence[str]] = None,
                 transform_base: Callable = None,
                 transform_sar: Callable = None,
                 transform_dem: Callable = None,
                 normalization: Callable = None,
                 modality_dropout_indices: Optional[Sequence[int]] = None,
                 modality_dropout_prob: float = 0.0,
                 sparse_crop_supervision: Optional[SparseFloodCropSupervision] = None,
                 hard_negative_crop_supervision: Optional[AuditGuidedHardNegativeCropSupervision] = None,
                 hard_positive_crop_supervision: Optional[AuditGuidedHardPositiveCropSupervision] = None) -> None:
        super().__init__()
        self.input_modalities = canonicalize_modalities(
            input_modalities or (["vv", "vh"] + (["dem"] if include_dem else []))
        )
        self._include_dem = "dem" in self.input_modalities
        self._requires_derived = any(name in DERIVED_MODALITIES for name in self.input_modalities)
        self._name = "flood"
        self._subset = subset
        self.transform_base = transform_base
        self.transform_sar = transform_sar
        self.transform_dem = transform_dem
        self.normalization = normalization
        self.modality_dropout_indices = tuple(int(i) for i in (modality_dropout_indices or []))
        self.modality_dropout_prob = float(modality_dropout_prob or 0.0)
        self.sparse_crop_supervision = sparse_crop_supervision
        self.hard_negative_crop_supervision = hard_negative_crop_supervision
        self.hard_positive_crop_supervision = hard_positive_crop_supervision

        root = Path(path) / subset
        self.image_files = sorted(glob(str(root / "sar" / "*.tif")))
        self.label_files = sorted(glob(str(root / "mask" / "*.tif")))
        if not self.image_files:
            raise FileNotFoundError(f"No SAR tiles found under: {root / 'sar'}")
        if len(self.image_files) != len(self.label_files):
            raise ValueError(
                f"Length mismatch between tiles and masks: {len(self.image_files)} != {len(self.label_files)}"
            )
        for image, mask in zip(self.image_files, self.label_files):
            image_tile = Path(image).stem
            label_tile = Path(mask).stem
            if image_tile != label_tile:
                raise ValueError(f"Tile stem mismatch: image={image_tile}, mask={label_tile}")

        self.dem_files: List[str] = []
        if self._include_dem:
            self.dem_files = sorted(glob(str(root / "dem" / "*.tif")))
            if len(self.image_files) != len(self.dem_files):
                raise ValueError("Length mismatch between SAR tiles and DEM tiles")
            for image, dem in zip(self.image_files, self.dem_files):
                if Path(image).stem != Path(dem).stem:
                    raise ValueError(f"Tile stem mismatch: image={Path(image).stem}, dem={Path(dem).stem}")

        self.derived_files: List[str] = []
        if self._requires_derived:
            self.derived_files = sorted(glob(str(root / "derived" / "*.tif")))
            if len(self.image_files) != len(self.derived_files):
                raise ValueError(
                    "Length mismatch between tiles and derived features. "
                    "Run `floodmap derive-features` for this processed dataset."
                )
            for image, derived in zip(self.image_files, self.derived_files):
                if Path(image).stem != Path(derived).stem:
                    raise ValueError(
                        f"Tile stem mismatch: image={Path(image).stem}, derived={Path(derived).stem}"
                    )

    @classmethod
    def name(cls) -> str:
        return cls._name

    @classmethod
    def categories(cls) -> Dict[int, str]:
        return cls._categories

    @classmethod
    def palette(cls) -> Dict[int, tuple]:
        return cls._palette

    @classmethod
    def ignore_index(cls) -> int:
        return cls._ignore_index

    @classmethod
    def mean(cls) -> Tuple[float, ...]:
        return cls._mean

    @classmethod
    def std(cls) -> Tuple[float, ...]:
        return cls._std

    def stage(self) -> str:
        return self._subset

    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        if len(mask) != len(self.image_files):
            raise ValueError(f"Selection mask length mismatch: expected {len(self.image_files)}, got {len(mask)}")
        self.image_files = [x for include, x in zip(mask, self.image_files) if include]
        self.label_files = [x for include, x in zip(mask, self.label_files) if include]
        if self._include_dem:
            self.dem_files = [x for include, x in zip(mask, self.dem_files) if include]
        if self._requires_derived:
            self.derived_files = [x for include, x in zip(mask, self.derived_files) if include]
        if stage:
            self._subset = stage

    @staticmethod
    def _channels_last(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr[..., None]
        if arr.ndim == 3:
            return arr
        raise ValueError(f"Unsupported raster shape: {arr.shape}")

    def _stack_modalities(self, index: int, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sar = imread(self.image_files[index], channels_first=False).astype(np.float32)
        sar = self._channels_last(sar)
        sar = np.nan_to_num(sar, nan=0.0, posinf=30.0, neginf=-30.0)
        if self.transform_sar is not None:
            pair = self.transform_sar(image=sar, mask=label)
            sar = self._channels_last(pair.get("image"))
            label = pair.get("mask")

        dem = None
        if self._include_dem:
            dem = imread(self.dem_files[index], channels_first=False).astype(np.float32)
            dem = self._channels_last(dem)
            dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
            if self.transform_dem is not None:
                pair = self.transform_dem(image=dem, mask=label)
                dem = self._channels_last(pair.get("image"))
                label = pair.get("mask")

        derived = None
        if self._requires_derived:
            derived = imread(self.derived_files[index], channels_first=False).astype(np.float32)
            derived = self._channels_last(derived)
            if derived.shape[-1] < 3:
                raise ValueError(
                    f"Derived feature tile {self.derived_files[index]} has {derived.shape[-1]} band(s); expected 3"
                )
            derived = np.nan_to_num(derived, nan=0.0, posinf=0.0, neginf=0.0)

        source = {
            "vv": (sar, 0),
            "vh": (sar, 1),
            "r": (sar, 0),
            "g": (sar, 1),
            "b": (sar, 2),
            "dem": (dem, 0),
            "vv_vh_log_ratio": (derived, 0),
            "dem_slope": (derived, 1),
            "dem_tpi": (derived, 2),
        }
        channels: list[np.ndarray] = []
        expected_shape = sar.shape[:2]
        for modality in self.input_modalities:
            arr, band = source[modality]
            if arr is None or band >= arr.shape[-1]:
                raise ValueError(
                    f"Modality '{modality}' is unavailable in tile {Path(self.image_files[index]).stem}"
                )
            channel = arr[..., band]
            if channel.shape != expected_shape:
                raise ValueError(
                    f"Spatial mismatch for modality '{modality}': {channel.shape} != {expected_shape}"
                )
            channels.append(channel)
        return np.stack(channels, axis=-1).astype(np.float32, copy=False), label

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        label = imread(self.label_files[index]).squeeze(0).astype(np.uint8)
        image, label = self._stack_modalities(index, label)

        crop_metadata = None
        if self.hard_positive_crop_supervision is not None:
            image, label, crop_metadata = self.hard_positive_crop_supervision(
                image=image, mask=label, sample_path=self.label_files[index]
            )
        elif self.hard_negative_crop_supervision is not None:
            image, label, crop_metadata = self.hard_negative_crop_supervision(
                image=image, mask=label, sample_path=self.label_files[index]
            )
        elif self.sparse_crop_supervision is not None:
            image, label, crop_metadata = self.sparse_crop_supervision(image=image, mask=label)

        if self.transform_base is not None:
            pair = self.transform_base(image=image, mask=label)
            image = pair.get("image")
            label = pair.get("mask")
        if self.normalization:
            pair = self.normalization(image=image, mask=label)
            image = pair.get("image")
            label = pair.get("mask")

        if self.modality_dropout_indices and self.modality_dropout_prob > 0.0 and np.random.random() < self.modality_dropout_prob:
            if torch.is_tensor(image):
                image = image.clone()
                for channel_idx in self.modality_dropout_indices:
                    if 0 <= channel_idx < image.shape[0]:
                        image[channel_idx, ...] = 0.0
            elif isinstance(image, np.ndarray):
                image = image.copy()
                if image.ndim == 3:
                    for channel_idx in self.modality_dropout_indices:
                        if 0 <= channel_idx < image.shape[-1]:
                            image[..., channel_idx] = 0.0
                elif image.ndim == 2 and 0 in self.modality_dropout_indices:
                    image[...] = 0.0
        if isinstance(image, np.ndarray):
            image = np.nan_to_num(image, nan=0.0, posinf=30.0, neginf=-30.0).astype(np.float32, copy=False)
        if crop_metadata is not None:
            return (
                image,
                label,
                int(crop_metadata.requested_mode),
                int(crop_metadata.applied_mode),
                int(crop_metadata.crop_size),
            )
        return image, label

    def __len__(self) -> int:
        return len(self.image_files)


class RGBFloodDataset(FloodDataset):
    _mean = (0.485, 0.456, 0.406, 1.4241237e+02)
    _std = (0.229, 0.224, 0.225, 8.11010422e+01)


class WeightedFloodDataset(FloodDataset):
    def __init__(self,
                 path: Path,
                 subset: str = "train",
                 include_dem: bool = False,
                 input_modalities: Optional[Sequence[str]] = None,
                 transform_base: Callable = None,
                 transform_sar: Callable = None,
                 transform_dem: Callable = None,
                 normalization: Callable = None,
                 class_weights: Tuple[float, float, float] = (1.0, 0.5, 5.0),
                 modality_dropout_indices: Optional[Sequence[int]] = None,
                 modality_dropout_prob: float = 0.0,
                 sparse_crop_supervision: Optional[SparseFloodCropSupervision] = None,
                 hard_negative_crop_supervision: Optional[AuditGuidedHardNegativeCropSupervision] = None,
                 hard_positive_crop_supervision: Optional[AuditGuidedHardPositiveCropSupervision] = None) -> None:
        super().__init__(path,
                         subset=subset,
                         include_dem=include_dem,
                         input_modalities=input_modalities,
                         transform_base=transform_base,
                         transform_sar=transform_sar,
                         transform_dem=transform_dem,
                         normalization=normalization,
                         modality_dropout_indices=modality_dropout_indices,
                         modality_dropout_prob=modality_dropout_prob,
                         sparse_crop_supervision=sparse_crop_supervision,
                         hard_negative_crop_supervision=hard_negative_crop_supervision,
                         hard_positive_crop_supervision=hard_positive_crop_supervision)
        weights_array = np.zeros(256, dtype=np.float32)
        weights_array[:len(class_weights)] = np.array(class_weights)
        self.class_weights = weights_array
        self.weight_files = sorted(glob(str(Path(path) / subset / "weight" / "*.tif")))
        if len(self.image_files) != len(self.weight_files):
            raise ValueError(
                f"Length mismatch between tiles and weights: {len(self.image_files)} != {len(self.weight_files)}"
            )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = super().__getitem__(index)
        image, label, *metadata = sample
        weight_indices = imread(self.weight_files[index]).squeeze(0).astype(np.uint8)
        weight = self.class_weights[weight_indices]
        if metadata:
            return image, label, weight, *metadata
        return image, label, weight
