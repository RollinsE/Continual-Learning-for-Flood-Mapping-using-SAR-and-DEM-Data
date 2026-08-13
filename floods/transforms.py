from typing import Sequence, Union

import numpy as np
import torch
from albumentations import Normalize
from torch import Tensor


class Denormalize:

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        single_image = tensor.ndim == 3
        tensor = tensor.unsqueeze(0) if single_image else tensor
        channels = tensor.size(1)
        # slice to support a lower number of channels
        means = self.mean[:channels].view(1, -1, 1, 1).to(tensor.device)
        stds = self.std[:channels].view(1, -1, 1, 1).to(tensor.device)
        tensor = tensor * stds + means
        # swap from [B, C, H, W] to [B, H, W, C]
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor[0] if single_image else tensor
        return tensor.detach().cpu().numpy()


class ClipNormalize(Normalize):

    def __init__(self,
                 mean: tuple,
                 std: tuple,
                 clip_min: Union[float, tuple],
                 clip_max: Union[float, tuple],
                 max_pixel_value: float = 1.0,
                 always_apply: bool = False,
                 p: float = 1.0):
        # The wrapper accepts always_apply for config compatibility while using
        # the current Albumentations Normalize signature.
        super().__init__(mean=mean, std=std, max_pixel_value=max_pixel_value, p=p)
        self.clip_min = clip_min
        self.clip_max = clip_max

    @staticmethod
    def _broadcast_clip(value, channels: int):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return arr
        if arr.size < channels:
            arr = np.pad(arr, (0, channels - arr.size), mode="edge")
        return arr[:channels].reshape((1, 1, channels))

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Clip raw channels first, then apply mean/std normalization.

        The robust-normalization order is:
        finite guard -> raw clipping -> standardization -> finite guard.
        """
        img = img.astype(np.float32, copy=False)
        channels = img.shape[-1] if img.ndim == 3 else 1
        clip_min = self._broadcast_clip(self.clip_min, channels)
        clip_max = self._broadcast_clip(self.clip_max, channels)
        pos_fill = float(np.max(clip_max))
        neg_fill = float(np.min(clip_min))
        img = np.nan_to_num(img, nan=0.0, posinf=pos_fill, neginf=neg_fill)
        img = np.clip(img, clip_min, clip_max).astype(np.float32, copy=False)
        result = super().apply(img, **params)
        result = np.nan_to_num(result, nan=0.0, posinf=30.0, neginf=-30.0)
        return np.clip(result, -30.0, 30.0).astype(np.float32, copy=False)

    def get_transform_init_args_names(self):
        parent = list(super().get_transform_init_args_names())
        return tuple(parent + ["clip_min", "clip_max"])


class RobustMinMaxNormalize(Normalize):

    def __init__(self,
                 mean: tuple,
                 std: tuple,
                 clip_min: Union[float, tuple],
                 clip_max: Union[float, tuple],
                 max_pixel_value: float = 1.0,
                 p: float = 1.0):
        """Apply percentile clipping, robust min-max scaling, then normalization."""
        super().__init__(mean=mean, std=std, max_pixel_value=max_pixel_value, p=p)
        self.clip_min = clip_min
        self.clip_max = clip_max

    @staticmethod
    def _broadcast_clip(value, channels: int):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return arr
        if arr.size < channels:
            arr = np.pad(arr, (0, channels - arr.size), mode="edge")
        return arr[:channels].reshape((1, 1, channels))

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.astype(np.float32, copy=False)
        channels = img.shape[-1] if img.ndim == 3 else 1
        clip_min = self._broadcast_clip(self.clip_min, channels)
        clip_max = self._broadcast_clip(self.clip_max, channels)
        pos_fill = float(np.max(clip_max))
        neg_fill = float(np.min(clip_min))
        img = np.nan_to_num(img, nan=0.0, posinf=pos_fill, neginf=neg_fill)
        img = np.clip(img, clip_min, clip_max).astype(np.float32, copy=False)
        scale = np.maximum(clip_max - clip_min, 1e-6)
        img = ((img - clip_min) / scale).astype(np.float32, copy=False)
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
        result = super().apply(img, **params)
        result = np.nan_to_num(result, nan=0.0, posinf=30.0, neginf=-30.0)
        return np.clip(result, -30.0, 30.0).astype(np.float32, copy=False)

    def get_transform_init_args_names(self):
        parent = list(super().get_transform_init_args_names())
        return tuple(parent + ["clip_min", "clip_max"])


class ProviderSARNormalize:
    """Convert processed Sentinel-1 channels to dB and apply provider statistics.

    The transform follows the Albumentations call contract so it can be composed with
    ``ToTensorV2`` without coupling the dataset to a particular pretrained provider.
    DEM channels, when present, remain in metres before provider standardisation.
    """

    def __init__(
        self,
        modalities: Sequence[str],
        *,
        source_sar_transform: str,
        sar_mean: Sequence[float],
        sar_std: Sequence[float],
        dem_mean: float | None = None,
        dem_std: float | None = None,
    ):
        self.modalities = [str(value).lower() for value in modalities]
        self.source_sar_transform = str(source_sar_transform).lower().replace("-", "_")
        self.sar_mean = np.asarray(sar_mean, dtype=np.float32)
        self.sar_std = np.asarray(sar_std, dtype=np.float32)
        self.dem_mean = None if dem_mean is None else float(dem_mean)
        self.dem_std = None if dem_std is None else float(dem_std)

    def __repr__(self) -> str:
        return (
            f"ProviderSARNormalize(modalities={self.modalities}, "
            f"source_sar_transform={self.source_sar_transform!r})"
        )

    @staticmethod
    def _to_db(values: np.ndarray, source: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if source == "db10":
            return values
        if source == "log1p":
            linear = np.expm1(np.maximum(values, 0.0))
        elif source == "linear":
            linear = np.maximum(values, 0.0)
        else:
            raise ValueError(f"Unsupported source SAR transform: {source}")
        return 10.0 * np.log10(np.maximum(linear, 1e-7))

    def __call__(self, *, image: np.ndarray, mask: np.ndarray, **kwargs):
        image = np.asarray(image, dtype=np.float32).copy()
        if image.ndim == 2:
            image = image[..., None]
        sar_index = {"vv": 0, "vh": 1}
        for channel_index, modality in enumerate(self.modalities):
            if modality in sar_index:
                values = self._to_db(image[..., channel_index], self.source_sar_transform)
                idx = sar_index[modality]
                image[..., channel_index] = (values - self.sar_mean[idx]) / max(float(self.sar_std[idx]), 1e-6)
            elif modality == "dem":
                if self.dem_mean is None or self.dem_std is None:
                    raise ValueError("DEM provider statistics are required when DEM is active")
                image[..., channel_index] = (image[..., channel_index] - self.dem_mean) / max(self.dem_std, 1e-6)
        image = np.nan_to_num(image, nan=0.0, posinf=30.0, neginf=-30.0)
        image = np.clip(image, -30.0, 30.0).astype(np.float32, copy=False)
        return {"image": image, "mask": mask}
