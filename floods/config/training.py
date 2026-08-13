import hashlib
from enum import member
from pathlib import Path
from typing import Any, List, Optional

try:
    from pydantic.v1 import BaseModel as BaseSettings, Field, validator
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings, Field, validator
from torch.nn import BatchNorm2d, Identity, LeakyReLU, ReLU

try:
    from inplace_abn import InPlaceABN, InPlaceABNSync
except Exception:  # pragma: no cover - optional normalization dependency
    InPlaceABN = BatchNorm2d
    InPlaceABNSync = BatchNorm2d
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau

from floods.config.base import CallableEnum, EnvConfig, Initializer, InstantiableSettings
from floods.losses import BCEWithLogitsLoss, BCETverskyLoss, CombinedLoss, FocalLoss, FocalTverskyComboLoss, FocalTverskyLoss, LovaszSoftmax
from floods.metrics import F1Score, IoU, MCC, Precision, Recall
from floods.utils.schedulers import PolynomialLRDecay


class Optimizers(CallableEnum):
    adam = member(Initializer(Adam))
    adamw = member(Initializer(AdamW))
    sgd = member(Initializer(SGD, momentum=0.9))


class Schedulers(CallableEnum):
    plateau = member(Initializer(ReduceLROnPlateau))
    exp = member(Initializer(ExponentialLR, gamma=0.96))
    cosine = member(Initializer(CosineAnnealingLR, T_max=10))
    poly = member(Initializer(PolynomialLRDecay, max_decay_steps=99, end_learning_rate=0.0001, power=3.0))


class Losses(CallableEnum):
    bce = member(BCEWithLogitsLoss)
    focal = member(FocalLoss)
    tversky = member(FocalTverskyLoss)
    bce_tversky = member(BCETverskyLoss)
    focal_tversky_combo = member(FocalTverskyComboLoss)
    lovasz = member(LovaszSoftmax)
    combo = member(Initializer(CombinedLoss,
                        criterion_a=Initializer(BCEWithLogitsLoss),
                        criterion_b=Initializer(FocalTverskyLoss)))


class Metrics(CallableEnum):
    f1 = member(Initializer(F1Score, ignore_index=255))
    iou = member(Initializer(IoU, ignore_index=255))
    precision = member(Initializer(Precision, ignore_index=255, reduction="macro"))
    recall = member(Initializer(Recall, ignore_index=255, reduction="macro"))
    mcc = member(Initializer(MCC, ignore_index=255))


class NormLayers(CallableEnum):
    std = member(Initializer(BatchNorm2d))
    iabn = member(Initializer(InPlaceABN, activation="leaky_relu", activation_param=0.01))
    iabn_sync = member(Initializer(InPlaceABNSync, activation="leaky_relu", activation_param=0.01))


class ActivationLayers(CallableEnum):
    ident = member(Initializer(Identity))
    relu = member(Initializer(ReLU, inplace=True))
    lrelu = member(Initializer(LeakyReLU, inplace=True))


def _enum_from_name(enum_cls, value, aliases: Optional[dict[str, str]] = None):
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return value
    key = str(value).strip().lower().replace("-", "_")
    key = (aliases or {}).get(key, key)
    try:
        return enum_cls[key]
    except KeyError as exc:
        allowed = ", ".join(item.name for item in enum_cls)
        raise ValueError(f"{enum_cls.__name__} value must be one of: {allowed}") from exc


def _enum_list_from_names(enum_cls, values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [_enum_from_name(enum_cls, item) for item in values]


class TrainerConfig(BaseSettings):
    cpu: bool = Field(False, description="Whether to use CPU or not")
    amp: bool = Field(True, description="Whether to use mixed precision (native)")
    progress_bar: bool = Field(True, description="Show one compact tqdm progress bar during training and validation")
    progress_log_interval: int = Field(0, description="Optional heartbeat interval, in batches, when progress bars are hidden. Use 0 to disable heartbeat logs")
    progress_label: str = Field("Training", description="Short label shown at the start of each tqdm progress bar")
    batch_size: int = Field(8, description="Batch size for training")
    num_workers: int = Field(4, description="Number of workers per dataloader")
    max_epochs: int = Field(100, description="How many epochs")
    train_metrics: List[Metrics] = Field([Metrics.iou], description="Which training metrics to use")
    val_metrics: List[Metrics] = Field([Metrics.f1, Metrics.iou, Metrics.mcc, Metrics.precision, Metrics.recall],
                                       description="Which validation metrics to use")
    monitor: Metrics = Field(Metrics.iou, description="Metric to be monitored")
    patience: int = Field(25, description="Amount of epochs without improvement in the monitored metric")
    validate_every: int = Field(1, description="How many epochs between validation rounds")
    save_last: bool = Field(True, description="Save a resumable last.ckpt checkpoint after every epoch")
    save_epoch_checkpoints: bool = Field(False, description="Save an additional epoch_NNN.ckpt checkpoint after every epoch")
    extend_epochs: Optional[int] = Field(None, description="When resuming, run this many additional epochs beyond the checkpoint epoch")
    reset_early_stopping_on_resume: bool = Field(False, description="Reset the early-stopping patience counter after loading a resume checkpoint")
    temperature: float = Field(2.0, description="Temperature for simulated annealing, >= 1")
    temp_epochs: int = Field(20, description="How many epochs before T goes back to 1")
    grad_clip_norm: Optional[float] = Field(1.0, description="Maximum gradient norm. Use 0 or null to disable clipping")
    detect_anomaly: bool = Field(False, description="Enable PyTorch anomaly detection for debugging")
    skip_nonfinite_batches: bool = Field(True, description="Skip isolated batches with non-finite losses or gradients instead of updating weights")
    amp_full_precision_retry: bool = Field(True, description="When AMP produces non-finite gradients, retry that batch once with autocast disabled before skipping it.")
    max_skipped_batch_fraction: float = Field(0.0, description="Abort an epoch when skipped non-finite batches exceed this fraction of all batches. Use 0 to disable the budget.")
    threshold_sweep: bool = Field(False, description="Evaluate validation metrics across multiple probability thresholds")
    event_macro_validation: bool = Field(False, description="Compute event-macro and worst-event validation metrics and select checkpoints by event-macro F1.")
    thresholds: List[float] = Field([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70], description="Probability thresholds used when threshold_sweep is enabled")
    threshold_metric: str = Field("f1", description="Metric used to select the best threshold: f1, iou, mcc, precision, or recall")
    monitor_threshold_sweep: bool = Field(True, description="When threshold_sweep is enabled, use best_<threshold_metric> for early stopping and best checkpoint selection")
    metric_mode: str = Field("global", description="Validation metric accounting mode: global, batch_average, or both. Training still logs global metrics; evaluation can report both.")
    group_dro: bool = Field(False, description="Use event-level GroupDRO instead of ordinary empirical-risk minimisation.")
    group_dro_eta: float = Field(0.01, description="Exponentiated-gradient step size for GroupDRO event weights.")
    group_dro_min_weight: float = Field(0.001, description="Minimum probability retained for every training event after each GroupDRO update.")
    group_dro_warmup_epochs: int = Field(1, description="Initial ERM epochs before GroupDRO event reweighting becomes active.")

    @validator("group_dro_eta", "group_dro_min_weight")
    def validate_group_dro_nonnegative(cls, v, values, **kwargs):
        value = float(v)
        if value < 0.0:
            raise ValueError("GroupDRO eta and minimum weight must be non-negative")
        return value

    @validator("group_dro_warmup_epochs")
    def validate_group_dro_warmup_epochs(cls, v, values, **kwargs):
        value = int(v)
        if value < 0:
            raise ValueError("group_dro_warmup_epochs must be non-negative")
        return value

    @validator("max_skipped_batch_fraction")
    def validate_max_skipped_batch_fraction(cls, v, values, **kwargs):
        value = float(v or 0.0)
        if value < 0.0 or value >= 1.0:
            raise ValueError("max_skipped_batch_fraction must be 0 or a value below 1")
        return value

    @validator("train_metrics", "val_metrics", pre=True)
    def parse_metric_lists(cls, v, values, **kwargs):
        return _enum_list_from_names(Metrics, v)

    @validator("monitor", pre=True)
    def parse_monitor_metric(cls, v, values, **kwargs):
        return _enum_from_name(Metrics, v)


class OptimizerConfig(InstantiableSettings):
    target: Optimizers = Field(Optimizers.adamw, description="Which optimizer to apply")
    lr: float = Field(1e-3, description="Global LR, still required to build optimizers")
    encoder_lr: float = Field(1e-3, description="Learning rate for the encoder branch")
    decoder_lr: float = Field(1e-3, description="Learning rate for the decoder branch")
    momentum: float = Field(0.9, description="Momentum for SGD")
    weight_decay: float = Field(1e-2, description="Weight decay for the optimizer")

    @validator("target", pre=True)
    def parse_optimizer_target(cls, v, values, **kwargs):
        return _enum_from_name(Optimizers, v)

    def instantiate(self, *args, **kwargs) -> Any:
        kwargs = dict(lr=self.lr, weight_decay=self.weight_decay, **kwargs)
        if self.target == Optimizers.sgd:
            kwargs.update(dict(momentum=self.momentum))
        return self.target(*args, **kwargs)


class SchedulerConfig(InstantiableSettings):
    target: Schedulers = Field(Schedulers.exp, description="Which scheduler to apply")

    @validator("target", pre=True)
    def parse_scheduler_target(cls, v, values, **kwargs):
        return _enum_from_name(Schedulers, v)

    def instantiate(self, *args, **kwargs) -> Any:
        return self.target(*args, **kwargs)


class LossConfig(InstantiableSettings):
    target: Losses = Field(Losses.bce, description="Which loss to apply")
    alpha: float = Field(0.6, description="Alpha/Tversky false-positive penalty")
    beta: float = Field(0.4, description="Beta/Tversky false-negative penalty")
    gamma: float = Field(2.0, description="Gamma for focal/Tversky-style losses")
    focal_gamma: float = Field(2.0, description="Gamma for the focal BCE component in focal_tversky_combo")
    focal_alpha: float = Field(1.0, description="Alpha multiplier for the focal BCE component")
    bce_weight: float = Field(0.5, description="Weight of BCE term in bce_tversky")
    focal_weight: float = Field(0.5, description="Weight of focal BCE term in focal_tversky_combo")
    tversky_weight: float = Field(0.5, description="Weight of Tversky term in combined losses")
    reduction: str = Field("mean", description="How to reduce the loss")

    @validator("target", pre=True)
    def parse_loss_target(cls, v, values, **kwargs):
        return _enum_from_name(Losses, v)

    def instantiate(self, *args, **kwargs) -> Any:
        if "ignore_index" not in kwargs:
            raise ValueError("ignore_index is required when instantiating the configured loss")
        # Add loss-specific parameters while keeping the shared interface stable.
        if self.target == Losses.focal:
            kwargs.update(gamma=self.gamma, alpha=self.focal_alpha)
        elif self.target == Losses.tversky:
            kwargs.update(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        elif self.target == Losses.bce_tversky:
            kwargs.update(alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                          bce_weight=self.bce_weight, tversky_weight=self.tversky_weight)
        elif self.target == Losses.focal_tversky_combo:
            kwargs.update(alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                          focal_gamma=self.focal_gamma, focal_alpha=self.focal_alpha,
                          focal_weight=self.focal_weight, tversky_weight=self.tversky_weight)
        return self.target(*args, **kwargs)


class DatasetConfig(EnvConfig):
    path: str = Field("", description="Path to the dataset")
    train_source_split: str = Field("train", description="Processed split used to construct the training dataset.")
    val_source_split: str = Field("val", description="Processed split used to construct the validation dataset.")
    train_include_events: List[str] = Field(default_factory=list, description="Optional EMSR events retained for training.")
    train_exclude_events: List[str] = Field(default_factory=list, description="Optional EMSR events excluded from training.")
    val_include_events: List[str] = Field(default_factory=list, description="Optional EMSR events retained for validation.")
    val_exclude_events: List[str] = Field(default_factory=list, description="Optional EMSR events excluded from validation.")
    in_channels: int = Field(3, description="How many input channels, including extras")
    include_dem: bool = Field(False, description="Whether to include the DEM as an additional input channel")
    input_modalities: List[str] = Field(default_factory=list, description="Ordered input channel names. Empty uses implicit vv/vh[/dem] inference; derived examples: vv vh dem vv_vh_log_ratio dem_slope dem_tpi.")
    class_weights: str = Field(None, description="Optional path to a class weight array (npy format)")
    mask_body_ratio: float = Field(0.0, description="Alias for train_mask_body_ratio. Use 0.0 to disable filtering.")
    train_mask_body_ratio: Optional[float] = Field(None, description="Minimum foreground-mask ratio required to keep a training tile. Use 0.0 to disable filtering.")
    val_mask_body_ratio: float = Field(0.0, description="Optional foreground-mask ratio for validation tiles. Defaults to 0.0 to keep validation unfiltered.")
    test_mask_body_ratio: float = Field(0.0, description="Optional foreground-mask ratio for test tiles. Defaults to 0.0 to keep test evaluation unfiltered.")
    weighted_sampling: bool = Field(False, description="Whether to sample images based on mask entropy")
    foreground_balanced_sampling: bool = Field(False, description="Sample empty and non-empty tiles to approximate a target foreground-tile ratio")
    foreground_sample_ratio: float = Field(0.65, description="Target fraction of sampled tiles that should contain at least foreground_min_ratio flood pixels")
    foreground_min_ratio: float = Field(0.0, description="Minimum foreground ratio used to classify a tile as non-empty for foreground-balanced sampling")
    weighted_samples_multiplier: float = Field(1.0, description="Number of sampled items per epoch as a multiplier of the filtered training set length")
    sample_smoothing: float = Field(0.8, description="Value between 0 and 1 to smooth entropy weights")
    augmentation_profile: str = Field("standard", description="Training augmentation profile: none, geometric, sar_radiometric, standard, crop_aware, deformation, or composite.")
    disable_random_crop: bool = Field(False, description="Disable random/crop-aware crop augmentation regardless of augmentation_profile")
    disable_elastic: bool = Field(False, description="Disable ElasticTransform in training augmentation regardless of augmentation_profile")
    disable_grid_distortion: bool = Field(False, description="Disable GridDistortion in training augmentation regardless of augmentation_profile")
    disable_sar_noise: bool = Field(False, description="Disable SAR GaussianBlur/MultiplicativeNoise augmentation regardless of augmentation_profile")
    sparse_crop_supervision: bool = Field(False, description="Enable real sparse-flood crop supervision after stacking all input modalities.")
    sparse_crop_normal_fraction: float = Field(0.50, description="Full-tile fraction used on flood-containing samples when sparse crop supervision is enabled.")
    sparse_crop_flood_fraction: float = Field(0.25, description="Flood-centred crop fraction used on flood-containing samples.")
    sparse_crop_hard_background_fraction: float = Field(0.25, description="Hard-background crop fraction used on flood-containing samples.")
    sparse_crop_sizes: List[int] = Field([256, 320, 384, 448], description="Candidate square crop sizes before resizing back to image_size.")
    sparse_crop_attempts: int = Field(24, description="Maximum candidate searches for a hard-background crop.")
    sparse_crop_hard_background_max_fg_ratio: float = Field(0.001, description="Maximum flood-pixel ratio allowed in a hard-background crop.")
    sparse_crop_min_valid_ratio: float = Field(0.50, description="Minimum non-ignore-pixel ratio required in a crop.")
    normalization_stats_path: Optional[str] = Field(None, description="Optional train-fitted normalization_stats.json to use for clipping and normalization")
    normalization_mode: str = Field("fixed", description="Normalization source: fixed, stats, robust_percentile, robust_minmax, ssl4eo_s1, or terramind_v1")
    source_sar_transform: str = Field("auto", description="Processed SAR representation before provider normalization: auto, linear, log1p, or db10")
    modality_dropout: bool = Field(False, description="Apply train-only input-modality dropout after normalization. Useful for regularising DEM dependence.")
    modality_dropout_prob: float = Field(0.0, description="Probability of applying modality dropout to each training sample.")
    drop_modalities: List[str] = Field(["dem"], description="Modalities to zero when modality dropout triggers, e.g. ['dem'] or ['vv', 'vh'].")
    stratified_sampling: bool = Field(False, description="Use foreground-ratio stratified replacement sampling")
    event_balanced_sampling: bool = Field(False, description="Balance training samples by EMSR event first, then by foreground-ratio bin within each event")
    event_balance_power: float = Field(0.0, description="Event-mass exponent: 0.0 gives equal event mass; 0.5 gives square-root tempered balancing; 1.0 approaches per-tile sampling.")
    event_tile_weight_cap: float = Field(0.0, description="Optional cap on event-balanced tile weights as a multiple of the median positive weight. Use 0 to disable.")
    hard_example_sampling: bool = Field(False, description="Oversample audited hard examples listed in an error-audit tile_error_metrics.csv file")
    hard_example_csv: Optional[str] = Field(None, description="Path to a train-split error-audit CSV used by hard-example sampling")
    hard_example_categories: List[str] = Field(["false_negative_low_recall", "poor_overlap", "false_positive_empty", "false_negative_missed"], description="Error categories to oversample from the hard-example CSV")
    hard_example_fg_bins: List[str] = Field(["tiny", "small"], description="Foreground-size bins to treat as hard when their tile F1 is at or below hard_example_max_f1")
    hard_example_max_f1: float = Field(0.30, description="Maximum tile F1 for foreground-bin examples to be selected as hard")
    hard_example_weight: float = Field(4.0, description="Relative sampling weight multiplier for selected hard examples")
    hard_example_max_fraction: float = Field(0.60, description="Maximum probability mass assigned to hard examples after weighting")
    hard_positive_region_sampling: bool = Field(False, description="Oversample tiles with mined hard-positive regions and crop directly around false-negative flood areas")
    hard_positive_manifest: Optional[str] = Field(None, description="Path to hard_positive_regions.csv produced by floodmap mine-hard-positives")
    hard_positive_region_weight: float = Field(3.0, description="Relative sampling weight multiplier for tiles with mined hard-positive regions")
    hard_positive_region_max_fraction: float = Field(0.20, description="Maximum probability mass assigned to tiles with mined hard-positive regions")
    hard_positive_crop_probability: float = Field(1.0, description="Probability of applying a mined hard-positive crop when a manifest tile is sampled")
    hard_negative_region_sampling: bool = Field(False, description="Oversample tiles with mined hard-negative regions and crop directly around actual false-positive areas")
    hard_negative_manifest: Optional[str] = Field(None, description="Path to hard_negative_regions.csv produced by floodmap mine-hard-negatives")
    hard_negative_region_weight: float = Field(4.0, description="Relative sampling weight multiplier for tiles with mined hard-negative regions")
    hard_negative_region_max_fraction: float = Field(0.35, description="Maximum probability mass assigned to tiles with mined hard-negative regions")
    hard_negative_crop_probability: float = Field(1.0, description="Probability of applying a mined hard-negative crop when a manifest tile is sampled")
    pos_weight_from_train: bool = Field(False, description="Compute BCE/focal positive-class weight from the filtered training masks")
    pos_weight_max: float = Field(20.0, description="Maximum value for train-derived BCE/focal positive-class weight")
    fg_bin_edges: List[float] = Field([0.0, 0.005, 0.02, 0.10], description="Foreground-ratio bin edges for stratified sampling: empty, tiny, small, medium, large")
    fg_bin_sample_weights: List[float] = Field([0.20, 0.20, 0.25, 0.25, 0.10], description="Target sample fractions for empty, tiny, small, medium, large foreground-ratio bins")
    cache_hash: str = Field(None, description="Cache key generated from the effective dataset configuration")
    cache_dir: str = Field("data/cache", description="Directory used for mask-filter and sampler caches")
    clear_cache: bool = Field(False, description="Recompute cached masks and sampler weights before training")

    @validator("sparse_crop_normal_fraction", "sparse_crop_flood_fraction", "sparse_crop_hard_background_fraction", "sparse_crop_hard_background_max_fg_ratio", "sparse_crop_min_valid_ratio")
    def validate_sparse_crop_fraction(cls, v, values, **kwargs):
        value = float(v)
        if value < 0.0 or value > 1.0:
            raise ValueError("sparse-crop fractions and ratios must be between 0 and 1")
        return value

    @validator("sparse_crop_sizes", pre=True)
    def validate_sparse_crop_sizes(cls, v, values, **kwargs):
        if v is None:
            return [256, 320, 384, 448]
        if isinstance(v, (int, float, str)):
            v = [v]
        values_out = sorted({int(item) for item in v})
        if not values_out or any(item <= 0 for item in values_out):
            raise ValueError("sparse_crop_sizes must contain positive integers")
        return values_out

    @validator("hard_positive_crop_probability")
    def validate_hard_positive_crop_probability(cls, v, values, **kwargs):
        value = float(v)
        if value < 0.0 or value > 1.0:
            raise ValueError("hard_positive_crop_probability must be between 0 and 1")
        return value

    @validator("hard_positive_region_weight")
    def validate_hard_positive_region_weight(cls, v, values, **kwargs):
        value = float(v)
        if value <= 1.0:
            raise ValueError("hard_positive_region_weight must be greater than 1")
        return value

    @validator("hard_positive_region_max_fraction")
    def validate_hard_positive_region_max_fraction(cls, v, values, **kwargs):
        value = float(v)
        if not 0.0 < value < 1.0:
            raise ValueError("hard_positive_region_max_fraction must be between 0 and 1")
        return value

    @validator("hard_negative_crop_probability")
    def validate_hard_negative_crop_probability(cls, v, values, **kwargs):
        value = float(v)
        if value < 0.0 or value > 1.0:
            raise ValueError("hard_negative_crop_probability must be between 0 and 1")
        return value

    @validator("hard_negative_region_weight")
    def validate_hard_negative_region_weight(cls, v, values, **kwargs):
        value = float(v)
        if value <= 1.0:
            raise ValueError("hard_negative_region_weight must be greater than 1")
        return value

    @validator("hard_negative_region_max_fraction")
    def validate_hard_negative_region_max_fraction(cls, v, values, **kwargs):
        value = float(v)
        if not 0.0 < value < 1.0:
            raise ValueError("hard_negative_region_max_fraction must be between 0 and 1")
        return value

    @validator("sparse_crop_attempts")
    def validate_sparse_crop_attempts(cls, v, values, **kwargs):
        value = int(v)
        if value <= 0:
            raise ValueError("sparse_crop_attempts must be positive")
        return value

    @validator("augmentation_profile")
    def validate_augmentation_profile(cls, v, values, **kwargs):
        value = str(v or "standard").strip().lower().replace("-", "_")
        aliases = {
            "light": "geometric",
            "heavy": "deformation",
            "sar_safe": "sar_radiometric",
            "safe": "geometric",
            "risky": "deformation",
            "full_deformation": "deformation",
            "composite_profile": "composite",
            "notebook": "composite",
        }
        value = aliases.get(value, value)
        allowed = {"none", "geometric", "sar_radiometric", "standard", "crop_aware", "deformation", "composite"}
        if value not in allowed:
            raise ValueError(f"augmentation_profile must be one of {sorted(allowed)}")
        return value

    @validator("normalization_mode")
    def validate_normalization_mode(cls, v, values, **kwargs):
        value = str(v or "fixed").strip().lower().replace("-", "_")
        aliases = {"notebook_robust": "robust_percentile"}
        value = aliases.get(value, value)
        allowed = {
            "fixed",
            "stats",
            "robust_percentile",
            "robust_minmax",
            "ssl4eo_s1",
            "terramind_v1",
        }
        if value not in allowed:
            raise ValueError(f"normalization_mode must be one of {sorted(allowed)}")
        return value

    @validator("modality_dropout_prob")
    def validate_modality_dropout_prob(cls, v, values, **kwargs):
        value = float(v or 0.0)
        if value < 0.0 or value > 1.0:
            raise ValueError("modality_dropout_prob must be between 0 and 1")
        return value

    @validator("input_modalities", pre=True)
    def validate_input_modalities(cls, v, values, **kwargs):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        from floods.modalities import canonicalize_modalities
        return canonicalize_modalities(v)

    @validator("drop_modalities", pre=True)
    def validate_drop_modalities(cls, v, values, **kwargs):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        allowed = {"vv", "vh", "dem", "r", "g", "b", "vv_vh_log_ratio", "dem_slope", "dem_tpi"}
        result = []
        for item in v:
            key = str(item).strip().lower().replace("-", "_")
            if not key:
                continue
            if key not in allowed:
                raise ValueError(f"drop_modalities entries must be one of {sorted(allowed)}")
            if key not in result:
                result.append(key)
        return result

    @validator("cache_hash", always=True)
    def post_load(cls, v, values, **kwargs):
        if v:
            return v
        return cls.build_cache_hash(values)

    @staticmethod
    def build_cache_hash(values) -> str:
        relevant = {
            "path": str(values.get("path", "")),
            "train_source_split": values.get("train_source_split"),
            "val_source_split": values.get("val_source_split"),
            "train_include_events": values.get("train_include_events"),
            "train_exclude_events": values.get("train_exclude_events"),
            "val_include_events": values.get("val_include_events"),
            "val_exclude_events": values.get("val_exclude_events"),
            "in_channels": values.get("in_channels"),
            "include_dem": values.get("include_dem"),
            "input_modalities": values.get("input_modalities"),
            "mask_body_ratio": values.get("mask_body_ratio"),
            "train_mask_body_ratio": values.get("train_mask_body_ratio"),
            "val_mask_body_ratio": values.get("val_mask_body_ratio"),
            "weighted_sampling": values.get("weighted_sampling"),
            "foreground_balanced_sampling": values.get("foreground_balanced_sampling"),
            "foreground_sample_ratio": values.get("foreground_sample_ratio"),
            "foreground_min_ratio": values.get("foreground_min_ratio"),
            "stratified_sampling": values.get("stratified_sampling"),
            "event_balanced_sampling": values.get("event_balanced_sampling"),
            "hard_example_sampling": values.get("hard_example_sampling"),
            "hard_example_csv": values.get("hard_example_csv"),
            "hard_example_categories": values.get("hard_example_categories"),
            "hard_example_fg_bins": values.get("hard_example_fg_bins"),
            "hard_example_max_f1": values.get("hard_example_max_f1"),
            "hard_example_weight": values.get("hard_example_weight"),
            "hard_example_max_fraction": values.get("hard_example_max_fraction"),
            "hard_positive_region_sampling": values.get("hard_positive_region_sampling"),
            "hard_positive_manifest": values.get("hard_positive_manifest"),
            "hard_positive_region_weight": values.get("hard_positive_region_weight"),
            "hard_positive_region_max_fraction": values.get("hard_positive_region_max_fraction"),
            "hard_positive_crop_probability": values.get("hard_positive_crop_probability"),
            "hard_negative_region_sampling": values.get("hard_negative_region_sampling"),
            "hard_negative_manifest": values.get("hard_negative_manifest"),
            "hard_negative_region_weight": values.get("hard_negative_region_weight"),
            "hard_negative_region_max_fraction": values.get("hard_negative_region_max_fraction"),
            "hard_negative_crop_probability": values.get("hard_negative_crop_probability"),
            "pos_weight_from_train": values.get("pos_weight_from_train"),
            "pos_weight_max": values.get("pos_weight_max"),
            "fg_bin_edges": values.get("fg_bin_edges"),
            "fg_bin_sample_weights": values.get("fg_bin_sample_weights"),
            "weighted_samples_multiplier": values.get("weighted_samples_multiplier"),
            "sample_smoothing": values.get("sample_smoothing"),
            "normalization_stats_path": values.get("normalization_stats_path"),
            "normalization_mode": values.get("normalization_mode"),
            "source_sar_transform": values.get("source_sar_transform"),
            "modality_dropout": values.get("modality_dropout"),
            "modality_dropout_prob": values.get("modality_dropout_prob"),
            "drop_modalities": values.get("drop_modalities"),
            "sparse_crop_supervision": values.get("sparse_crop_supervision"),
            "sparse_crop_normal_fraction": values.get("sparse_crop_normal_fraction"),
            "sparse_crop_flood_fraction": values.get("sparse_crop_flood_fraction"),
            "sparse_crop_hard_background_fraction": values.get("sparse_crop_hard_background_fraction"),
            "sparse_crop_sizes": values.get("sparse_crop_sizes"),
            "sparse_crop_attempts": values.get("sparse_crop_attempts"),
            "sparse_crop_hard_background_max_fg_ratio": values.get("sparse_crop_hard_background_max_fg_ratio"),
            "sparse_crop_min_valid_ratio": values.get("sparse_crop_min_valid_ratio"),
        }
        return hashlib.sha1(repr(sorted(relevant.items())).encode("utf-8")).hexdigest()

    def refresh_cache_hash(self) -> None:
        self.cache_hash = self.build_cache_hash(self.dict())


class ModelConfig(EnvConfig):
    weights_source: str = Field("random", description="Registered initialisation source: random, imagenet, SSL4EO, FG-MAE SAR, CROMA, or TerraMind")
    encoder: str = Field("resnet34", description="Which backbone to use (see timm library or the registered provider adapter)")
    decoder: str = Field("pspnet", description="Which decoder to apply: unet, unetpp, pspnet, deeplabv3, deeplabv3p, or segformer")
    pretrained: bool = Field(False, description="Whether to use a pretrained encoder or not")
    freeze: bool = Field(False, description="Freeze the feature extractor in incremental steps")
    multibranch: bool = Field(False, description="Includes an additional low-res output, right after the encoder")
    output_stride: int = Field(16, description="Output stride for ResNet-like models")
    act: ActivationLayers = Field(ActivationLayers.relu, description="Which activation layer to use")
    norm: NormLayers = Field(NormLayers.std, description="Which normalization layer to use")
    dropout2d: bool = Field(False, description="Whether to apply standard drop. or channel drop. to the last f.map")
    transforms: str = Field("", description="Automatically populated by the script for tracking purposes")
    foundation_input_size: int = Field(224, description="Internal input size used by foundation-model adapters")
    foundation_pyramid_channels: int = Field(256, description="Common decoder width for foundation-model feature pyramids")

    @validator("act", pre=True)
    def parse_activation_layer(cls, v, values, **kwargs):
        return _enum_from_name(ActivationLayers, v)

    @validator("norm", pre=True)
    def parse_normalization_layer(cls, v, values, **kwargs):
        return _enum_from_name(NormLayers, v)

    @validator("weights_source")
    def validate_weights_source(cls, v, values, **kwargs):
        from floods.pretrained import normalize_source_name
        return normalize_source_name(v)

    @validator("foundation_input_size", "foundation_pyramid_channels")
    def validate_foundation_sizes(cls, v, values, **kwargs):
        value = int(v)
        if value <= 0:
            raise ValueError("foundation model sizes must be positive")
        return value

    @validator("decoder")
    def validate_decoder(cls, v, values, **kwargs):
        value = str(v or "").strip().lower().replace("-", "_")
        aliases = {
            "unet++": "unetpp",
            "unet_plus_plus": "unetpp",
            "deeplabv3+": "deeplabv3p",
            "deeplabv3_plus": "deeplabv3p",
            "deeplabv3plus": "deeplabv3p",
        }
        value = aliases.get(value, value)
        allowed = {"unet", "unetpp", "pspnet", "deeplabv3", "deeplabv3p", "segformer"}
        if value not in allowed:
            raise ValueError(f"decoder must be one of {sorted(allowed)}")
        return value

    @validator("norm")
    def post_load(cls, v, values, **kwargs):
        # Activated normalization layers already include the non-linearity.
        if v in (NormLayers.iabn, NormLayers.iabn_sync):
            values["act"] = ActivationLayers.ident
        return v


class TrainConfig(BaseSettings):
    seed: int = Field(1337, description="Random seed for deterministic runs")
    image_size: int = Field(512, description="Size of the input images")
    trainer: TrainerConfig = TrainerConfig()
    # ML options
    data: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    # logging options
    debug: bool = Field(False, description="Enables debug prints and logs")
    name: Optional[str] = Field(None, description="Run identifier, autogenerated when omitted")
    output_folder: str = Field("outputs", description="Root folder for run artifacts")
    num_samples: int = Field(8, description="How many sample batches to visualize, requires visualize=true")
    visualize: bool = Field(False, description="Turn on validation sample visualization in TensorBoard")
    comment: str = Field("", description="Optional run description stored in TensorBoard")
    version: str = Field("", description="Code revision recorded as a Git commit or installed release version")
    init_checkpoint: Optional[str] = Field(None, description="Load model weights from a checkpoint and start a fresh optimiser/scheduler/epoch counter")
    init_channel_adaptation: str = Field("strict", description="How to handle a wider input layer when warm-starting: strict or zero_extra.")
    resume: bool = Field(False, description="Resume from output_folder/name/models/last.ckpt when available")
    resume_from: Optional[str] = Field(None, description="Explicit checkpoint path to resume from")

    @validator("init_channel_adaptation")
    def validate_init_channel_adaptation(cls, v, values, **kwargs):
        value = str(v or "strict").strip().lower().replace("-", "_")
        allowed = {"strict", "zero_extra"}
        if value not in allowed:
            raise ValueError(f"init_channel_adaptation must be one of {sorted(allowed)}")
        return value
