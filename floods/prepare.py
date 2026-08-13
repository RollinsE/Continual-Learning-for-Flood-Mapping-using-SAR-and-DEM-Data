from pathlib import Path
from typing import Dict, Tuple, Sequence
import re

import inspect

import albumentations as alb
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data.sampler import WeightedRandomSampler

from floods.config import TestConfig, TrainConfig
from floods.datasets.base import DatasetBase
from floods.datasets.flood import FloodDataset, RGBFloodDataset
from floods.metrics import ConfusionMatrix, F1Score, IoU, MCC, Metric, Precision, Recall
from floods.model_factory import prepare_model
from floods.sparse_crops import SparseFloodCropSupervision
from floods.hard_negative_regions import AuditGuidedHardNegativeCropSupervision
from floods.hard_positive_regions import AuditGuidedHardPositiveCropSupervision
from floods.normalization import describe_stats, load_normalization_stats
from floods.modalities import has_derived_modalities, resolve_input_modalities
from floods.transforms import ClipNormalize, Denormalize, RobustMinMaxNormalize, ProviderSARNormalize
from floods.hard_examples import selected_hard_example_indices
from floods.utils.common import get_logger
from floods.utils.tiling.functional import entropy_weights, foreground_ratios_from_labels, mask_body_ratio_from_threshold

LOG = get_logger(__name__)


def _random_sized_crop(image_size: int, min_crop: int, max_crop: int):
    """Create a RandomSizedCrop transform across Albumentations 1.x and 2.x.

    Albumentations 1.x used ``height`` and ``width``. Albumentations 2.x
    replaced those arguments with ``size=(height, width)``.
    """
    try:
        return alb.RandomSizedCrop(
            min_max_height=(min_crop, max_crop),
            size=(image_size, image_size),
            p=0.8,
        )
    except (TypeError, ValueError):
        return alb.RandomSizedCrop(
            min_max_height=(min_crop, max_crop),
            height=image_size,
            width=image_size,
            p=0.8,
        )


def _flip_transform(p: float = 0.5):
    """Create a random flip transform across Albumentations versions."""
    if hasattr(alb, "Flip"):
        return alb.Flip(p=p)
    return alb.OneOf([
        alb.HorizontalFlip(p=1.0),
        alb.VerticalFlip(p=1.0),
        alb.Transpose(p=1.0),
    ], p=p)


def _elastic_transform(p: float = 0.5):
    """Create ElasticTransform with arguments valid for the installed version."""
    params = inspect.signature(alb.ElasticTransform).parameters
    kwargs = {"alpha": 1, "sigma": 50, "p": p}
    if "alpha_affine" in params:
        kwargs["alpha_affine"] = 50
    if "approximate" in params:
        kwargs["approximate"] = False
    if "interpolation" in params:
        kwargs["interpolation"] = 1
    return alb.ElasticTransform(**kwargs)


def _grid_distortion_composite(p: float = 0.2):
    params = inspect.signature(alb.GridDistortion).parameters
    kwargs = {"num_steps": 5, "distort_limit": (-0.3, 0.3), "p": p}
    if "interpolation" in params:
        kwargs["interpolation"] = 1
    return alb.GridDistortion(**kwargs)


def _rotate_composite(p: float = 0.5):
    params = inspect.signature(alb.Rotate).parameters
    kwargs = {"limit": 30, "p": p}
    if "border_mode" in params:
        kwargs["border_mode"] = 0
    if "interpolation" in params:
        kwargs["interpolation"] = 1
    if "mask_interpolation" in params:
        kwargs["mask_interpolation"] = 0
    return alb.Rotate(**kwargs)


def _composite_pixel_transforms():
    """Composite pixel-level augmentations applied to the stacked image."""
    options = [
        alb.GaussianBlur(blur_limit=(3, 7), p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
    ]
    try:
        options.append(alb.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25), hole_width_range=(0.1, 0.25), p=1.0))
    except (TypeError, ValueError):
        options.append(alb.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=1.0))
    return alb.OneOf(options, p=0.5)


def _normalise_augmentation_profile(profile: str) -> str:
    value = str(profile or "standard").strip().lower().replace("-", "_")
    aliases = {
        "light": "geometric",
        "heavy": "deformation",
        "full_deformation": "deformation",
        "note" + "book": "composite",
        "safe": "geometric",
        "sar_safe": "sar_radiometric",
        "risky": "deformation",
    }
    value = aliases.get(value, value)
    allowed = {"none", "geometric", "sar_radiometric", "standard", "crop_aware", "deformation", "composite"}
    if value not in allowed:
        raise ValueError(f"augmentation_profile must be one of {sorted(allowed)}; got {profile!r}")
    return value



class CallableCompose:
    """Small compose wrapper that accepts Albumentations transforms and callables."""

    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __repr__(self) -> str:
        return "CallableCompose(" + ", ".join(repr(t) for t in self.transforms) + ")"

    def __call__(self, image: np.ndarray, mask: np.ndarray, **kwargs):
        data = {"image": image, "mask": mask}
        for transform in self.transforms:
            data = transform(image=data["image"], mask=data["mask"])
        return data


def train_transforms_base(image_size: int,
                          augmentation_profile: str = "standard",
                          disable_random_crop: bool = False,
                          disable_elastic: bool = False,
                          disable_grid_distortion: bool = False):
    """Build shared training augmentations using professional profile names.

    Profiles:
      none: no train-time spatial augmentation.
      geometric: flips and 90-degree rotations.
      sar_radiometric: no shared spatial augmentation; SAR jitter is applied separately.
      standard: geometric + SAR radiometric augmentation (handled in SAR transform).
      crop_aware: standard + mask-aware random crop.
      deformation: geometric + elastic/grid deformation; experimental.
      composite: geometric + rotation + deformation + pixel-level regularisation.
    """
    profile = _normalise_augmentation_profile(augmentation_profile)
    if profile == "none" or profile == "sar_radiometric":
        return CallableCompose([])

    min_crop = image_size // 2
    max_crop = image_size
    transforms = []

    if profile == "composite":
        transforms.extend([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            _elastic_transform(p=0.2),
            _grid_distortion_composite(p=0.2),
            _rotate_composite(p=0.5),
            _composite_pixel_transforms(),
        ])
        if disable_elastic:
            transforms = [t for t in transforms if not t.__class__.__name__.lower().startswith("elastic")]
        if disable_grid_distortion:
            transforms = [t for t in transforms if not t.__class__.__name__.lower().startswith("grid")]
        return CallableCompose(transforms)

    if profile == "crop_aware" and not disable_random_crop:
        # Real crop-aware supervision is applied after SAR/DEM stacking in the
        # dataset. Keeping it out of this generic transform avoids the old
        # no-op 512x512 crop on already-512x512 tiles.
        pass
    elif profile == "deformation" and not disable_random_crop:
        transforms.append(_random_sized_crop(image_size=image_size, min_crop=min_crop, max_crop=max_crop))

    transforms.extend([
        _flip_transform(p=0.5),
        alb.RandomRotate90(p=0.5),
    ])

    if profile in {"deformation"}:
        if not disable_elastic:
            transforms.append(_elastic_transform())
        if not disable_grid_distortion:
            transforms.append(alb.GridDistortion(p=0.5))

    return CallableCompose(transforms)


def train_transforms_sar(augmentation_profile: str = "standard", disable_sar_noise: bool = False):
    profile = _normalise_augmentation_profile(augmentation_profile)
    if disable_sar_noise or profile in {"none", "geometric", "deformation", "composite"}:
        return alb.Compose([])
    transforms = [
        alb.OneOf([
            alb.GaussianBlur(blur_limit=(3, 7), p=0.5),
            alb.MultiplicativeNoise(multiplier=(0.85, 1.15), elementwise=True, per_channel=True, p=0.5),
        ], p=0.5)
    ]
    return alb.Compose(transforms)


def train_transforms_dem(channel_dropout: float = 0.0):
    transforms = []
    if channel_dropout > 0:
        transforms.append(alb.ChannelDropout(p=channel_dropout))
    return alb.Compose(transforms)


def _resolve_source_sar_transform(data_root: Path, configured: str = "auto") -> str:
    value = str(configured or "auto").strip().lower().replace("-", "_")
    allowed = {"auto", "linear", "log1p", "db10"}
    if value not in allowed:
        raise ValueError(f"source_sar_transform must be one of {sorted(allowed)}")
    if value != "auto":
        return value
    manifest_path = Path(data_root) / "preprocessing_manifest.json"
    if manifest_path.exists():
        import json
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_value = str(payload.get("sar_transform", "")).strip().lower().replace("-", "_")
        if manifest_value in {"linear", "log1p", "db10"}:
            LOG.info("Provider normalization detected processed SAR transform from manifest: %s", manifest_value)
            return manifest_value
    # Older processed datasets predate the manifest. Their positive, compressed SAR
    # tiles came from the project's log1p preprocessing path. Make the fallback
    # explicit in the log rather than silently treating them as dB values.
    LOG.warning(
        "preprocessing_manifest.json is missing or has no recognised sar_transform; "
        "assuming source_sar_transform=log1p. Pass --source-sar-transform explicitly to override."
    )
    return "log1p"


def provider_normalization_transform(
    *, modalities: Sequence[str], mode: str, source_sar_transform: str, data_root: Path
) -> CallableCompose:
    mode = str(mode).strip().lower()
    source = _resolve_source_sar_transform(data_root, source_sar_transform)
    if mode == "terramind_v1":
        normalize = ProviderSARNormalize(
            modalities, source_sar_transform=source,
            sar_mean=(-12.599, -20.293), sar_std=(5.195, 5.890),
            dem_mean=670.665, dem_std=951.272,
        )
    elif mode == "ssl4eo_s1":
        if "dem" in modalities:
            raise ValueError("SSL4EO/FG-MAE Sentinel-1 weights support VV+VH only, not DEM")
        normalize = ProviderSARNormalize(
            modalities, source_sar_transform=source,
            sar_mean=(-12.59, -20.26), sar_std=(5.26, 5.91),
        )
    else:
        raise ValueError(f"Unknown provider normalization mode: {mode}")
    return CallableCompose([normalize, ToTensorV2()])


def eval_transforms(mean: tuple,
                    std: tuple,
                    clip_min: tuple,
                    clip_max: tuple,
                    normalization_mode: str = "stats") -> alb.Compose:
    mode = str(normalization_mode or "stats").lower()
    if mode in {"robust_percentile", "note" + "book_robust", "robust_minmax"}:
        norm = RobustMinMaxNormalize(mean=mean, std=std, clip_min=clip_min, clip_max=clip_max)
    else:
        norm = ClipNormalize(mean=mean, std=std, clip_min=clip_min, clip_max=clip_max)
    return alb.Compose([norm, ToTensorV2()])


def inverse_transform(mean: tuple, std: tuple):
    return Denormalize(mean=mean, std=std)


def prepare_evaluation_dataset(config: TrainConfig, split: str = "val") -> Tuple[DatasetBase, list[str], bool]:
    configured = getattr(config.data, "input_modalities", None)
    implicit_rgb = (config.data.in_channels - int(config.data.include_dem)) == 3 if not configured else False
    modalities = resolve_input_modalities(
        configured,
        in_channels=config.data.in_channels,
        include_dem=config.data.include_dem,
        use_rgb=implicit_rgb,
    )
    use_rgb = modalities[:3] == ["r", "g", "b"]
    dataset_cls = RGBFloodDataset if use_rgb else FloodDataset
    config.data.input_modalities = list(modalities)
    config.data.in_channels = len(modalities)
    config.data.include_dem = "dem" in modalities
    derived_active = has_derived_modalities(modalities)
    norm_mode = str(getattr(config.data, "normalization_mode", "fixed") or "fixed").lower()
    if derived_active and norm_mode == "fixed":
        raise ValueError("Derived modalities require train-fitted normalization statistics.")
    if norm_mode in {"terramind_v1", "ssl4eo_s1"}:
        transform = provider_normalization_transform(
            modalities=modalities, mode=norm_mode,
            source_sar_transform=getattr(config.data, "source_sar_transform", "auto"),
            data_root=Path(config.data.path),
        )
        LOG.info("Using provider normalization: mode=%s | modalities=%s", norm_mode, modalities)
    else:
        stats_candidate = Path(config.data.path) / "normalization_stats.json"
        if not config.data.normalization_stats_path and stats_candidate.exists() and norm_mode in {"stats", "robust_percentile", "note" + "book_robust", "robust_minmax"}:
            config.data.normalization_stats_path = str(stats_candidate)
        if norm_mode in {"stats", "robust_percentile", "note" + "book_robust", "robust_minmax"} or getattr(config.data, "normalization_stats_path", None):
            if not config.data.normalization_stats_path:
                raise ValueError(f"normalization_mode='{norm_mode}' requires --normalization-stats-path")
            mean, std, clip_min, clip_max = load_normalization_stats(
                Path(config.data.normalization_stats_path), modalities, mode=norm_mode
            )
            LOG.info("Using train-fitted normalization stats (%s): %s", norm_mode, describe_stats(Path(config.data.normalization_stats_path)))
        else:
            mean = dataset_cls.mean()[:config.data.in_channels]
            std = dataset_cls.std()[:config.data.in_channels]
            if len(mean) != config.data.in_channels or len(std) != config.data.in_channels:
                raise ValueError(f"Fixed normalization has no statistics for modalities={modalities}")
            clip_min = tuple([-30.0] * config.data.in_channels)
            clip_max = tuple([30.0] * config.data.in_channels)
        transform = eval_transforms(
            mean=mean, std=std, clip_min=clip_min, clip_max=clip_max, normalization_mode=norm_mode
        )
    dataset = dataset_cls(
        path=Path(config.data.path),
        subset=split,
        include_dem=config.data.include_dem,
        input_modalities=modalities,
        normalization=transform,
    )
    return dataset, modalities, use_rgb


def _apply_event_filter(dataset: DatasetBase, *, include_events=None, exclude_events=None, label: str) -> None:
    include = {str(v).upper() for v in (include_events or [])}
    exclude = {str(v).upper() for v in (exclude_events or [])}
    if not include and not exclude:
        return
    keep = []
    for path in dataset.label_files:
        event = _event_id_from_path(path).upper()
        selected = (not include or event in include) and event not in exclude
        keep.append(selected)
    before = len(dataset)
    dataset.add_mask(keep)
    LOG.info("%s event filter: kept %d/%d tiles | include=%s | exclude=%s", label, len(dataset), before, sorted(include), sorted(exclude))
    if len(dataset) == 0:
        raise ValueError(f"{label} event filter removed every tile")


def prepare_datasets(config: TrainConfig, use_rgb: bool = False) -> Tuple[DatasetBase, DatasetBase]:
    modalities = resolve_input_modalities(
        getattr(config.data, "input_modalities", None),
        in_channels=config.data.in_channels,
        include_dem=config.data.include_dem,
        use_rgb=use_rgb,
    )
    if not 2 <= len(modalities) <= 9:
        raise ValueError(f"Declared input modalities are not supported: {modalities}")
    config.data.input_modalities = list(modalities)
    config.data.in_channels = len(modalities)
    config.data.include_dem = "dem" in modalities
    if hasattr(config.data, "refresh_cache_hash"):
        config.data.refresh_cache_hash()

    use_rgb = all(name in modalities for name in ("r", "g", "b"))
    dataset_cls = RGBFloodDataset if use_rgb else FloodDataset
    derived_active = has_derived_modalities(modalities)
    profile = _normalise_augmentation_profile(config.data.augmentation_profile)
    if derived_active and profile not in {"none", "geometric"}:
        raise ValueError(
            "Precomputed derived channels require augmentation_profile='geometric' or 'none'. "
            "SAR radiometric/composite augmentation would make VV/VH inconsistent with the precomputed log-ratio channel."
        )

    data_root = Path(config.data.path)
    norm_mode = str(config.data.normalization_mode or "fixed").lower()
    if derived_active and norm_mode == "fixed":
        raise ValueError(
            "Derived modalities require train-fitted normalization. Run `floodmap fit-normalization` "
            "with the full modality list and set normalization_mode to robust_minmax, robust_percentile, or stats."
        )
    provider_transform = None
    if norm_mode in {"terramind_v1", "ssl4eo_s1"}:
        provider_transform = provider_normalization_transform(
            modalities=modalities, mode=norm_mode,
            source_sar_transform=getattr(config.data, "source_sar_transform", "auto"),
            data_root=data_root,
        )
        mean = std = clip_min = clip_max = None
        LOG.info("Using provider normalization: mode=%s | modalities=%s", norm_mode, modalities)
    else:
        stats_candidate = data_root / "normalization_stats.json"
        if not config.data.normalization_stats_path and stats_candidate.exists() and norm_mode in {"stats", "robust_percentile", "note" + "book_robust", "robust_minmax"}:
            config.data.normalization_stats_path = str(stats_candidate)
        if norm_mode in {"stats", "robust_percentile", "note" + "book_robust", "robust_minmax"} or config.data.normalization_stats_path:
            if not config.data.normalization_stats_path:
                raise ValueError(f"normalization_mode='{norm_mode}' requires --normalization-stats-path")
            mean, std, clip_min, clip_max = load_normalization_stats(
                Path(config.data.normalization_stats_path), modalities, mode=norm_mode
            )
            LOG.info("Using train-fitted normalization stats (%s): %s", norm_mode, describe_stats(Path(config.data.normalization_stats_path)))
        else:
            mean = dataset_cls.mean()[:config.data.in_channels]
            std = dataset_cls.std()[:config.data.in_channels]
            if len(mean) != config.data.in_channels or len(std) != config.data.in_channels:
                raise ValueError(
                    f"Fixed normalization has no statistics for modalities={modalities}. Use train-fitted normalization."
                )
            clip_min = tuple([-30.0] * config.data.in_channels)
            clip_max = tuple([30.0] * config.data.in_channels)
            LOG.info("Using fixed dataset normalization statistics")

    base_trf = train_transforms_base(image_size=config.image_size,
                                     augmentation_profile=config.data.augmentation_profile,
                                     disable_random_crop=config.data.disable_random_crop,
                                     disable_elastic=config.data.disable_elastic,
                                     disable_grid_distortion=config.data.disable_grid_distortion)
    sar_trf = train_transforms_sar(augmentation_profile=config.data.augmentation_profile,
                                   disable_sar_noise=config.data.disable_sar_noise or derived_active)
    dem_trf = train_transforms_dem(channel_dropout=0)
    drop_names = [str(m).lower() for m in getattr(config.data, "drop_modalities", [])]
    modality_dropout_indices = [modalities.index(m) for m in drop_names if m in modalities]
    modality_dropout_enabled = bool(getattr(config.data, "modality_dropout", False)) and float(getattr(config.data, "modality_dropout_prob", 0.0) or 0.0) > 0.0 and bool(modality_dropout_indices)
    sparse_crop = None
    hard_negative_crop = None
    hard_positive_crop = None
    sparse_crop_enabled = bool(getattr(config.data, "sparse_crop_supervision", False))
    hard_negative_crop_enabled = bool(getattr(config.data, "hard_negative_region_sampling", False))
    hard_positive_crop_enabled = bool(getattr(config.data, "hard_positive_region_sampling", False))
    if sum(bool(v) for v in (sparse_crop_enabled, hard_negative_crop_enabled, hard_positive_crop_enabled)) > 1:
        raise ValueError("Sparse, hard-negative-region, and hard-positive-region crop supervision are mutually exclusive.")
    if profile == "crop_aware" and not bool(config.data.disable_random_crop):
        sparse_crop_enabled = True
        config.data.sparse_crop_supervision = True
        if hasattr(config.data, "refresh_cache_hash"):
            config.data.refresh_cache_hash()
    if sparse_crop_enabled:
        eligible_crop_sizes = sorted({int(size) for size in config.data.sparse_crop_sizes if 0 < int(size) < int(config.image_size)})
        if not eligible_crop_sizes:
            raise ValueError(
                "Sparse crop supervision requires at least one --sparse-crop-size smaller than --image-size. "
                f"Got crop sizes={config.data.sparse_crop_sizes} and image_size={config.image_size}."
            )
        sparse_crop = SparseFloodCropSupervision(
            target_size=config.image_size,
            crop_sizes=eligible_crop_sizes,
            normal_fraction=config.data.sparse_crop_normal_fraction,
            flood_centered_fraction=config.data.sparse_crop_flood_fraction,
            hard_background_fraction=config.data.sparse_crop_hard_background_fraction,
            attempts=config.data.sparse_crop_attempts,
            hard_background_max_fg_ratio=config.data.sparse_crop_hard_background_max_fg_ratio,
            min_valid_ratio=config.data.sparse_crop_min_valid_ratio,
            ignore_index=dataset_cls.ignore_index(),
        )
    if hard_negative_crop_enabled:
        if not config.data.hard_negative_manifest:
            raise ValueError("--hard-negative-manifest is required when audit-guided hard-negative region sampling is enabled")
        hard_negative_crop = AuditGuidedHardNegativeCropSupervision(
            manifest_path=config.data.hard_negative_manifest,
            target_size=config.image_size,
            probability=config.data.hard_negative_crop_probability,
            ignore_index=dataset_cls.ignore_index(),
        )
    if hard_positive_crop_enabled:
        if not config.data.hard_positive_manifest:
            raise ValueError("--hard-positive-manifest is required when audit-guided hard-positive region sampling is enabled")
        hard_positive_crop = AuditGuidedHardPositiveCropSupervision(
            manifest_path=config.data.hard_positive_manifest,
            target_size=config.image_size,
            probability=config.data.hard_positive_crop_probability,
            ignore_index=dataset_cls.ignore_index(),
        )
    if getattr(config.data, "modality_dropout", False) and not modality_dropout_indices:
        LOG.warning("Modality dropout requested but none of drop_modalities=%s are present in active modalities=%s; disabling it.", drop_names, modalities)
    LOG.info("Training augmentation: profile=%s | crop_aware=%s | deformation=%s | sar_radiometric=%s",
             profile,
             (profile == "crop_aware") and not bool(config.data.disable_random_crop),
             (profile in {"deformation"}) and not (bool(config.data.disable_elastic) and bool(config.data.disable_grid_distortion)),
             (profile in {"sar_radiometric", "standard", "crop_aware", "deformation"}) and not bool(config.data.disable_sar_noise) and not derived_active)
    if sparse_crop is not None:
        LOG.info(
            "Sparse-flood crop supervision: enabled on flood-containing tiles | "
            "normal=%.2f flood-centred=%.2f hard-background=%.2f | crop sizes=%s -> %dx%d | "
            "hard-background max flood ratio=%.6f",
            sparse_crop.normal_fraction,
            sparse_crop.flood_centered_fraction,
            sparse_crop.hard_background_fraction,
            list(sparse_crop.crop_sizes),
            config.image_size,
            config.image_size,
            sparse_crop.hard_background_max_fg_ratio,
        )
    else:
        LOG.info("Sparse-flood crop supervision: disabled")
    if hard_positive_crop is not None:
        LOG.info(
            "Audit-guided hard-positive region crops: enabled | manifest=%s | crop probability=%.2f",
            config.data.hard_positive_manifest, config.data.hard_positive_crop_probability)
    else:
        LOG.info("Audit-guided hard-positive region crops: disabled")
    if hard_negative_crop is not None:
        LOG.info(
            "Audit-guided hard-negative region crops: enabled | manifest=%s | crop probability=%.2f",
            config.data.hard_negative_manifest,
            config.data.hard_negative_crop_probability,
        )
    else:
        LOG.info("Audit-guided hard-negative region crops: disabled")
    if modality_dropout_enabled:
        LOG.info("Training modality dropout: p=%.3f | drop_modalities=%s | channel_indices=%s | validation/test unchanged",
                 float(config.data.modality_dropout_prob), drop_names, modality_dropout_indices)
    else:
        LOG.info("Training modality dropout: disabled")
    # Store the transform summary for reproducible run logs.
    config.model.transforms = (str(base_trf) + str(sar_trf) + str(dem_trf)
                               + (str(sparse_crop) if sparse_crop is not None else "")
                               + (str(hard_negative_crop) if hard_negative_crop is not None else "")
                               + (str(hard_positive_crop) if hard_positive_crop is not None else ""))
    normalize = provider_transform or eval_transforms(mean=mean,
                                std=std,
                                clip_min=clip_min,
                                clip_max=clip_max,
                                normalization_mode=norm_mode,)
    # Store detailed transform definitions in the saved config; keep console output compact.
    LOG.debug("Train transforms: %s", config.model.transforms)
    LOG.debug("Eval. transforms: %s", str(normalize))
    # Create train and validation datasets.
    train_dataset = dataset_cls(path=data_root,
                                subset=str(getattr(config.data, "train_source_split", "train") or "train"),
                                include_dem=config.data.include_dem,
                                input_modalities=modalities,
                                transform_base=base_trf,
                                transform_sar=sar_trf,
                                transform_dem=dem_trf,
                                normalization=normalize,
                                modality_dropout_indices=modality_dropout_indices if modality_dropout_enabled else None,
                                modality_dropout_prob=float(config.data.modality_dropout_prob) if modality_dropout_enabled else 0.0,
                                sparse_crop_supervision=sparse_crop,
                                hard_negative_crop_supervision=hard_negative_crop,
                                hard_positive_crop_supervision=hard_positive_crop)
    valid_dataset = dataset_cls(path=data_root,
                                subset=str(getattr(config.data, "val_source_split", "val") or "val"),
                                include_dem=config.data.include_dem,
                                input_modalities=modalities,
                                normalization=normalize)
    _apply_event_filter(
        train_dataset,
        include_events=getattr(config.data, "train_include_events", None),
        exclude_events=getattr(config.data, "train_exclude_events", None),
        label="Training",
    )
    _apply_event_filter(
        valid_dataset,
        include_events=getattr(config.data, "val_include_events", None),
        exclude_events=getattr(config.data, "val_exclude_events", None),
        label="Validation",
    )
    enforce_event_disjoint = (
        str(getattr(config.data, "train_source_split", "train")) == str(getattr(config.data, "val_source_split", "val"))
        or bool(getattr(config.data, "train_include_events", None))
        or bool(getattr(config.data, "train_exclude_events", None))
        or bool(getattr(config.data, "val_include_events", None))
        or bool(getattr(config.data, "val_exclude_events", None))
    )
    if enforce_event_disjoint:
        overlap = set(_event_id_from_path(p).upper() for p in train_dataset.label_files) & set(_event_id_from_path(p).upper() for p in valid_dataset.label_files)
        if overlap:
            raise ValueError(f"Training/validation event leakage detected: {sorted(overlap)}")
    # Training-set filtering is optional. Validation is kept unfiltered by default
    # so reported metrics reflect the full validation split unless explicitly changed.
    train_mask_ratio = config.data.train_mask_body_ratio
    if train_mask_ratio is None:
        train_mask_ratio = config.data.mask_body_ratio
    val_mask_ratio = config.data.val_mask_body_ratio

    train_mask_ratio = None if train_mask_ratio is None else float(train_mask_ratio)
    val_mask_ratio = None if val_mask_ratio is None else float(val_mask_ratio)

    if train_mask_ratio is not None and train_mask_ratio > 0.0:
        train_imgs_mask, train_counts = mask_body_ratio_from_threshold(labels=train_dataset.label_files,
                                                                       ratio_threshold=train_mask_ratio,
                                                                       label="train",
                                                                       cache_hash=config.data.cache_hash,
                                                                       cache_dir=config.data.cache_dir,
                                                                       force_recompute=config.data.clear_cache)
        if int(train_counts[1]) == 0:
            raise ValueError("Training mask-ratio filtering removed every tile. Reduce --train-mask-body-ratio or use 0.0 to disable it.")
        train_dataset.add_mask(train_imgs_mask)
        LOG.info("Training mask filter: kept %d/%d tiles (%.2f%%) with foreground ratio >= %.8g",
                 int(train_counts[1]), len(train_imgs_mask), 100 * train_counts[1] / len(train_imgs_mask), train_mask_ratio)
    else:
        LOG.info("Training mask filter: disabled; using all %d training tiles", len(train_dataset))

    if val_mask_ratio is not None and val_mask_ratio > 0.0:
        val_imgs_mask, val_counts = mask_body_ratio_from_threshold(labels=valid_dataset.label_files,
                                                                   ratio_threshold=val_mask_ratio,
                                                                   label="val",
                                                                   cache_hash=config.data.cache_hash,
                                                                   cache_dir=config.data.cache_dir,
                                                                   force_recompute=config.data.clear_cache)
        valid_dataset.add_mask(val_imgs_mask)
        LOG.info("Validation mask filter: kept %d/%d tiles (%.2f%%) with foreground ratio >= %.8g",
                 int(val_counts[1]), len(val_imgs_mask), 100 * val_counts[1] / len(val_imgs_mask), val_mask_ratio)
    else:
        LOG.info("Validation mask filter: disabled; using all %d validation tiles", len(valid_dataset))

    return train_dataset, valid_dataset


def _event_id_from_path(path: str | Path) -> str:
    match = re.search(r"(EMSR\d+)", Path(path).name)
    return match.group(1) if match else "unknown"


def compute_binary_pos_weight_from_labels(labels: Sequence[str | Path],
                                          max_value: float = 20.0,
                                          cache_hash: str = "",
                                          cache_dir: str = "data/cache",
                                          force_recompute: bool = False) -> float:
    """Compute a clipped positive-class weight from binary mask tiles.

    ``pos_weight = negative_pixels / positive_pixels`` over valid pixels. The
    value is clipped because raw flood masks are extremely sparse and an
    unconstrained value can destabilise BCE/focal terms.
    """
    from floods.utils.gis import imread
    if len(labels) == 0:
        return 1.0
    target_file = Path(cache_dir) / f"pos-weight_{cache_hash}.npy"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists() and not force_recompute:
        try:
            return float(np.load(str(target_file))[0])
        except Exception:
            pass
    positive = 0
    negative = 0
    for label_path in labels:
        mask = imread(label_path).reshape(-1)
        valid = mask != 255
        positive += int(np.count_nonzero((mask == 1) & valid))
        negative += int(np.count_nonzero((mask == 0) & valid))
    if positive <= 0 or negative <= 0:
        value = 1.0
    else:
        value = float(negative / positive)
    value = float(min(max(value, 1.0), float(max_value or value)))
    np.save(str(target_file), np.asarray([value], dtype=np.float32))
    return value


def _sampler_num_samples(dataset: FloodDataset, samples_multiplier: float = 1.0) -> int:
    multiplier = float(samples_multiplier or 1.0)
    if multiplier <= 0:
        raise ValueError("weighted_samples_multiplier must be greater than 0")
    return max(1, int(round(len(dataset) * multiplier)))


def prepare_sampler(dataset: FloodDataset, cache_hash: str, smoothing: float = 0.8,
                    cache_dir: str = "data/cache", force_recompute: bool = False,
                    samples_multiplier: float = 1.0) -> WeightedRandomSampler:
    target_file = Path(cache_dir) / f"sample-weights_{cache_hash}.npy"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists() and target_file.is_file() and not force_recompute:
        LOG.info("Found an existing array of sample weights")
        weights = np.load(str(target_file))
    else:
        LOG.info("Computing weights for weighted random sampling")
        weights = entropy_weights(dataset.label_files, smoothing=smoothing)
        np.save(str(target_file), weights)
    if len(weights) != len(dataset):
        LOG.info("Cached sampler weights do not match the current dataset; recomputing them.")
        weights = entropy_weights(dataset.label_files, smoothing=smoothing)
        np.save(str(target_file), weights)
    num_samples = _sampler_num_samples(dataset, samples_multiplier=samples_multiplier)
    LOG.info("Weighted sampling: %d samples per epoch (%.2fx dataset length)", num_samples, num_samples / max(len(dataset), 1))
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


def prepare_foreground_balanced_sampler(dataset: FloodDataset, cache_hash: str,
                                        foreground_sample_ratio: float = 0.65,
                                        foreground_min_ratio: float = 0.0,
                                        cache_dir: str = "data/cache",
                                        force_recompute: bool = False,
                                        samples_multiplier: float = 1.0) -> WeightedRandomSampler:
    if not 0 < foreground_sample_ratio < 1:
        raise ValueError("foreground_sample_ratio must be between 0 and 1")
    ratios = foreground_ratios_from_labels(dataset.label_files,
                                           cache_hash=cache_hash,
                                           cache_dir=cache_dir,
                                           force_recompute=force_recompute)
    foreground = ratios > float(foreground_min_ratio)
    n_fg = int(np.count_nonzero(foreground))
    n_bg = int(len(ratios) - n_fg)
    if n_fg == 0 or n_bg == 0:
        LOG.warning("Foreground-balanced sampling requires both empty and non-empty tiles. Falling back to entropy-weighted sampling.")
        return prepare_sampler(dataset, cache_hash=cache_hash, cache_dir=cache_dir, force_recompute=force_recompute, samples_multiplier=samples_multiplier)
    weights = np.zeros(len(ratios), dtype=np.float64)
    weights[foreground] = foreground_sample_ratio / n_fg
    weights[~foreground] = (1.0 - foreground_sample_ratio) / n_bg
    num_samples = _sampler_num_samples(dataset, samples_multiplier=samples_multiplier)
    LOG.info("Foreground-balanced sampling: %d samples per epoch (target foreground-tile ratio %.2f; actual foreground tiles %d/%d)",
             num_samples, foreground_sample_ratio, n_fg, len(ratios))
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)



def prepare_stratified_sampler(dataset: FloodDataset, cache_hash: str,
                               fg_bin_edges: Sequence[float] = (0.0, 0.005, 0.02, 0.10),
                               fg_bin_sample_weights: Sequence[float] = (0.20, 0.20, 0.25, 0.25, 0.10),
                               cache_dir: str = "data/cache",
                               force_recompute: bool = False,
                               samples_multiplier: float = 1.0) -> WeightedRandomSampler:
    """Build a replacement sampler over foreground-ratio bins.

    Default bins are: empty, tiny, small, medium, large. This is more robust
    than binary empty/non-empty sampling for sparse flood masks because many
    positive tiles contain only a few foreground pixels.
    """
    ratios = foreground_ratios_from_labels(dataset.label_files,
                                           cache_hash=cache_hash,
                                           cache_dir=cache_dir,
                                           force_recompute=force_recompute)
    edges = [float(v) for v in fg_bin_edges]
    if len(edges) != 4:
        raise ValueError("fg_bin_edges must contain exactly four values: 0.0, tiny, small, medium")
    target = np.asarray([float(v) for v in fg_bin_sample_weights], dtype=np.float64)
    if target.size != 5:
        raise ValueError("fg_bin_sample_weights must contain five fractions for empty, tiny, small, medium, large")
    if np.any(target < 0) or target.sum() <= 0:
        raise ValueError("fg_bin_sample_weights must be non-negative and sum to a positive value")
    target = target / target.sum()

    masks = [
        ratios <= edges[0],
        (ratios > edges[0]) & (ratios < edges[1]),
        (ratios >= edges[1]) & (ratios < edges[2]),
        (ratios >= edges[2]) & (ratios < edges[3]),
        ratios >= edges[3],
    ]
    names = ["empty", "tiny", "small", "medium", "large"]
    counts = np.asarray([int(np.count_nonzero(m)) for m in masks], dtype=np.int64)
    available = counts > 0
    if np.count_nonzero(available) < 2:
        LOG.warning("Stratified sampling requires at least two non-empty foreground-ratio bins. Falling back to foreground-balanced sampling.")
        return prepare_foreground_balanced_sampler(dataset=dataset,
                                                   cache_hash=cache_hash,
                                                   cache_dir=cache_dir,
                                                   force_recompute=force_recompute,
                                                   samples_multiplier=samples_multiplier)
    effective = target.copy()
    effective[~available] = 0.0
    effective = effective / effective.sum()
    weights = np.zeros(len(ratios), dtype=np.float64)
    for idx, mask in enumerate(masks):
        if counts[idx] > 0 and effective[idx] > 0:
            weights[mask] = effective[idx] / counts[idx]
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("Invalid stratified sampler weights")
    num_samples = _sampler_num_samples(dataset, samples_multiplier=samples_multiplier)
    summary = ", ".join(f"{name}={count} target={frac:.2f}" for name, count, frac in zip(names, counts, effective))
    LOG.info("Foreground-ratio stratified sampling: %d samples per epoch | %s", num_samples, summary)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


def prepare_event_balanced_stratified_sampler(dataset: FloodDataset, cache_hash: str,
                                              fg_bin_edges: Sequence[float] = (0.0, 0.005, 0.02, 0.10),
                                              fg_bin_sample_weights: Sequence[float] = (0.20, 0.20, 0.25, 0.25, 0.10),
                                              cache_dir: str = "data/cache",
                                              force_recompute: bool = False,
                                              samples_multiplier: float = 1.0,
                                              event_balance_power: float = 0.0,
                                              tile_weight_cap: float = 0.0) -> WeightedRandomSampler:
    """Balance replacement sampling by event and foreground-ratio bin.

    Event probability mass is proportional to ``event_size ** event_balance_power``.
    A power of 0 reproduces equal event mass, 0.5 gives square-root tempered
    balancing, and 1 approaches equal per-tile event mass. Within each event,
    probability mass is distributed across empty/tiny/small/medium/large bins
    according to ``fg_bin_sample_weights`` after removing unavailable bins. This
    prevents a large event or a dominant foreground-size bin from controlling
    most batches.
    """
    ratios = foreground_ratios_from_labels(dataset.label_files,
                                           cache_hash=cache_hash,
                                           cache_dir=cache_dir,
                                           force_recompute=force_recompute)
    edges = [float(v) for v in fg_bin_edges]
    if len(edges) != 4:
        raise ValueError("fg_bin_edges must contain exactly four values: 0.0, tiny, small, medium")
    bin_target = np.asarray([float(v) for v in fg_bin_sample_weights], dtype=np.float64)
    if bin_target.size != 5 or np.any(bin_target < 0) or bin_target.sum() <= 0:
        raise ValueError("fg_bin_sample_weights must contain five non-negative fractions")
    bin_target = bin_target / bin_target.sum()
    bin_masks = [
        ratios <= edges[0],
        (ratios > edges[0]) & (ratios < edges[1]),
        (ratios >= edges[1]) & (ratios < edges[2]),
        (ratios >= edges[2]) & (ratios < edges[3]),
        ratios >= edges[3],
    ]
    bin_names = ["empty", "tiny", "small", "medium", "large"]
    events = np.asarray([_event_id_from_path(p) for p in dataset.label_files])
    unique_events = sorted(set(events.tolist()))
    if len(unique_events) < 2:
        LOG.warning("Event-balanced sampling requires at least two events. Falling back to foreground-ratio stratified sampling.")
        return prepare_stratified_sampler(dataset=dataset,
                                          fg_bin_edges=fg_bin_edges,
                                          fg_bin_sample_weights=fg_bin_sample_weights,
                                          cache_hash=cache_hash,
                                          cache_dir=cache_dir,
                                          force_recompute=force_recompute,
                                          samples_multiplier=samples_multiplier)
    power = float(event_balance_power)
    if not 0.0 <= power <= 1.0:
        raise ValueError("event_balance_power must be between 0.0 and 1.0")
    event_sizes = {event: int(np.count_nonzero(events == event)) for event in unique_events}
    raw_event_mass = {event: float(max(event_sizes[event], 1) ** power) for event in unique_events}
    mass_total = sum(raw_event_mass.values())
    weights = np.zeros(len(dataset), dtype=np.float64)
    event_summaries = []
    for event in unique_events:
        event_mask = events == event
        event_count = event_sizes[event]
        event_mass = raw_event_mass[event] / mass_total
        available = np.asarray([np.count_nonzero(event_mask & bm) > 0 for bm in bin_masks], dtype=bool)
        local_target = bin_target.copy()
        local_target[~available] = 0.0
        if local_target.sum() <= 0:
            weights[event_mask] = event_mass / max(event_count, 1)
            event_summaries.append(f"{event}:{event_count}")
            continue
        local_target = local_target / local_target.sum()
        for bi, bm in enumerate(bin_masks):
            selected = event_mask & bm
            count = int(np.count_nonzero(selected))
            if count > 0 and local_target[bi] > 0:
                weights[selected] = event_mass * local_target[bi] / count
        event_summaries.append(f"{event}:{event_count}")
    cap_multiple = float(tile_weight_cap)
    capped = 0
    if cap_multiple > 0.0:
        positive = weights[weights > 0]
        median = float(np.median(positive)) if positive.size else 0.0
        if median > 0.0:
            cap = median * cap_multiple
            capped = int(np.count_nonzero(weights > cap))
            weights = np.minimum(weights, cap)
            weights = weights / weights.sum()
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("Invalid event-balanced sampler weights")
    num_samples = _sampler_num_samples(dataset, samples_multiplier=samples_multiplier)
    global_counts = [int(np.count_nonzero(m)) for m in bin_masks]
    LOG.info("Event-balanced foreground-ratio sampling: %d samples per epoch | events=%d | power=%.2f | cap=%.2fx median | capped=%d (%s)",
             num_samples, len(unique_events), power, cap_multiple, capped,
             ", ".join(event_summaries[:12]) + ("..." if len(event_summaries) > 12 else ""))
    LOG.info("Foreground bins available globally: %s", ", ".join(f"{n}={c}" for n, c in zip(bin_names, global_counts)))
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)



def prepare_hard_example_sampler(dataset: FloodDataset,
                                 hard_example_csv: str,
                                 hard_example_categories: Sequence[str] = ("false_negative_low_recall", "poor_overlap", "false_positive_empty", "false_negative_missed"),
                                 hard_example_fg_bins: Sequence[str] = ("tiny", "small"),
                                 hard_example_max_f1: float = 0.30,
                                 hard_example_weight: float = 4.0,
                                 hard_example_max_fraction: float = 0.60,
                                 samples_multiplier: float = 1.0) -> WeightedRandomSampler:
    """Build a sampler that oversamples audited hard examples.

    This sampler is intended for a second-stage fine-tune. It keeps every tile
    available, but assigns extra probability mass to training tiles that a
    previous model handled badly according to a train-split error audit.
    """
    weight = float(hard_example_weight)
    max_fraction = float(hard_example_max_fraction)
    if weight <= 1.0:
        raise ValueError("hard_example_weight must be greater than 1.0")
    if not 0.0 < max_fraction < 1.0:
        raise ValueError("hard_example_max_fraction must be between 0 and 1")

    hard_indices = selected_hard_example_indices(label_files=dataset.label_files,
                                                 hard_example_csv=hard_example_csv,
                                                 hard_example_categories=hard_example_categories,
                                                 hard_example_fg_bins=hard_example_fg_bins,
                                                 hard_example_max_f1=hard_example_max_f1)
    weights = np.ones(len(dataset), dtype=np.float64)
    hard_mask = np.zeros(len(dataset), dtype=bool)
    hard_mask[list(hard_indices)] = True
    weights[hard_mask] *= weight

    hard_mass = float(weights[hard_mask].sum())
    normal_mass = float(weights[~hard_mask].sum())
    if hard_mass > 0 and normal_mass > 0:
        current_fraction = hard_mass / (hard_mass + normal_mass)
        if current_fraction > max_fraction:
            target_hard_mass = (max_fraction * normal_mass) / (1.0 - max_fraction)
            weights[hard_mask] *= target_hard_mass / hard_mass
            hard_mass = float(weights[hard_mask].sum())
            current_fraction = hard_mass / (hard_mass + normal_mass)
    else:
        current_fraction = 1.0 if hard_mass > 0 else 0.0

    num_samples = _sampler_num_samples(dataset, samples_multiplier=samples_multiplier)
    LOG.info("Hard-example sampling: %d samples per epoch | hard tiles=%d/%d | target mass<=%.2f | effective hard mass=%.2f | weight=%.2f",
             num_samples, int(np.count_nonzero(hard_mask)), len(dataset), max_fraction, current_fraction, weight)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

# prepare_model is imported from floods.model_factory above.
def prepare_metrics(config: TrainConfig, device: torch.device) -> Tuple[dict, dict]:
    # Build train and validation metric collections.
    t_metrics = config.trainer.train_metrics
    v_metrics = config.trainer.val_metrics
    train_metrics = {e.name: e.value(device=device) for e in t_metrics}
    valid_metrics = {e.name: e.value(device=device) for e in v_metrics}
    # Ensure epoch summaries always include the key segmentation metrics.
    valid_metrics.setdefault("f1", F1Score(ignore_index=255, device=device))
    valid_metrics.setdefault("iou", IoU(ignore_index=255, device=device))
    valid_metrics.setdefault("precision", Precision(ignore_index=255, reduction="macro", device=device))
    valid_metrics.setdefault("recall", Recall(ignore_index=255, reduction="macro", device=device))
    valid_metrics.setdefault("mcc", MCC(ignore_index=255, device=device))
    valid_metrics.update(dict(class_iou=IoU(reduction=None, device=device),
                              class_f1=F1Score(reduction=None, device=device)))
    LOG.debug("Train metrics: %s", str(list(train_metrics.keys())))
    LOG.debug("Eval. metrics: %s", str(list(valid_metrics.keys())))
    return train_metrics, valid_metrics


def prepare_test_metrics(config: TestConfig, device: torch.device) -> Dict[str, Metric]:
    test_metrics = {e.name: e.value(device=device) for e in config.test_metrics}
    # include class-wise metrics
    test_metrics.update(dict(precision=Precision(reduction=None, device=device),
                             recall=Recall(reduction=None, device=device),
                             fg_iou=IoU(reduction=None, device=device),
                             fg_f1=F1Score(reduction=None, device=device),
                             mcc=MCC(device=device),
                             bg_iou=IoU(reduction=None, device=device, background=True),
                             bg_f1=F1Score(reduction=None, device=device, background=True)))
    # include a confusion matrix
    test_metrics.update(dict(conf_mat=ConfusionMatrix(device=device)))
    return test_metrics
