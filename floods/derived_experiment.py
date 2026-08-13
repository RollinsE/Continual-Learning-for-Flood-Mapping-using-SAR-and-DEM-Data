from __future__ import annotations

import json
from pathlib import Path

from floods.config import TrainConfig
from floods.modalities import canonicalize_modalities
from floods.utils.common import get_logger, store_config

LOG = get_logger(__name__)

SIX_CHANNEL_MODALITIES = [
    "vv",
    "vh",
    "dem",
    "vv_vh_log_ratio",
    "dem_slope",
    "dem_tpi",
]


def _validate_normalization_file(path: Path) -> None:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    available = {
        str(item.get("channel", "")).strip().lower()
        for item in payload.get("channels", [])
        if item.get("channel")
    }
    missing = [name for name in SIX_CHANNEL_MODALITIES if name not in available]
    if missing:
        raise ValueError(
            f"Normalization file {path} is missing channels {missing}. "
            "Fit six-channel statistics before preparing the experiment."
        )


def prepare_derived_experiment_config(
    base_config: TrainConfig,
    *,
    output_config: Path,
    run_id: str,
    artifacts_dir: Path,
    baseline_checkpoint: Path,
    normalization_stats_path: Path,
    batch_size: int = 4,
    epochs: int = 20,
    patience: int = 6,
    encoder_lr: float = 1e-5,
    decoder_lr: float = 1e-5,
    max_skipped_batch_fraction: float = 0.02,
) -> TrainConfig:
    """Create a controlled six-channel fine-tune from a three-channel baseline."""
    baseline_checkpoint = Path(baseline_checkpoint)
    normalization_stats_path = Path(normalization_stats_path)
    output_config = Path(output_config)
    artifacts_dir = Path(artifacts_dir)
    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")
    if not normalization_stats_path.is_file():
        raise FileNotFoundError(f"Normalization stats not found: {normalization_stats_path}")
    _validate_normalization_file(normalization_stats_path)

    configured = list(getattr(base_config.data, "input_modalities", []) or [])
    if configured:
        baseline_modalities = canonicalize_modalities(configured)
    else:
        baseline_modalities = ["vv", "vh"] + (["dem"] if base_config.data.include_dem else [])
    if baseline_modalities != ["vv", "vh", "dem"] or int(base_config.data.in_channels) != 3:
        raise ValueError(
            "The controlled derived-feature experiment requires a three-channel VV/VH/DEM baseline. "
            f"Resolved baseline modalities: {baseline_modalities}; in_channels={base_config.data.in_channels}."
        )

    config = base_config.copy(deep=True)
    config.name = str(run_id)
    config.output_folder = str(artifacts_dir)
    config.data.input_modalities = list(SIX_CHANNEL_MODALITIES)
    config.data.in_channels = len(SIX_CHANNEL_MODALITIES)
    config.data.include_dem = True
    config.data.normalization_stats_path = str(normalization_stats_path)

    # Derived channels are precomputed from the unaugmented tile. Shared geometric
    # transforms remain valid, while SAR-only radiometric transforms would break
    # consistency between VV/VH and the stored ratio.
    config.data.augmentation_profile = "geometric"
    config.data.disable_random_crop = True
    config.data.disable_elastic = True
    config.data.disable_grid_distortion = True
    config.data.disable_sar_noise = True

    # Isolate the input-channel change from every previous specialist intervention.
    config.data.weighted_sampling = False
    config.data.foreground_balanced_sampling = False
    config.data.stratified_sampling = False
    config.data.event_balanced_sampling = False
    config.data.hard_example_sampling = False
    config.data.hard_positive_region_sampling = False
    config.data.hard_negative_region_sampling = False
    config.data.sparse_crop_supervision = False
    config.data.modality_dropout = False
    config.data.weighted_samples_multiplier = 1.0
    config.data.hard_example_csv = None
    config.data.hard_positive_manifest = None
    config.data.hard_negative_manifest = None

    config.model.pretrained = False
    config.init_checkpoint = str(baseline_checkpoint)
    config.init_channel_adaptation = "zero_extra"
    config.resume = False
    config.resume_from = None

    config.trainer.batch_size = int(batch_size)
    config.trainer.max_epochs = int(epochs)
    config.trainer.patience = int(patience)
    config.trainer.amp = False
    config.trainer.save_last = True
    config.trainer.save_epoch_checkpoints = False
    config.trainer.grad_clip_norm = 1.0
    config.trainer.skip_nonfinite_batches = True
    config.trainer.max_skipped_batch_fraction = float(max_skipped_batch_fraction)
    config.trainer.threshold_sweep = True
    config.trainer.monitor_threshold_sweep = True

    config.optimizer.encoder_lr = float(encoder_lr)
    config.optimizer.decoder_lr = float(decoder_lr)
    config.comment = (
        "Controlled six-channel warm start from the retained VV/VH/DEM baseline. "
        "Added channels: VV-VH log-ratio, DEM slope, and local DEM topographic position."
    )
    if hasattr(config.data, "refresh_cache_hash"):
        config.data.refresh_cache_hash()

    output_config.parent.mkdir(parents=True, exist_ok=True)
    store_config(config, output_config)
    LOG.info("Derived-feature experiment configuration written to: %s", output_config)
    LOG.info(
        "Experiment plan: run_id=%s | modalities=%s | batch_size=%d | epochs=%d | patience=%d | "
        "encoder_lr=%.3g | decoder_lr=%.3g | amp=%s | nonfinite_budget=%.3f",
        run_id,
        SIX_CHANNEL_MODALITIES,
        config.trainer.batch_size,
        config.trainer.max_epochs,
        config.trainer.patience,
        config.optimizer.encoder_lr,
        config.optimizer.decoder_lr,
        config.trainer.amp,
        config.trainer.max_skipped_batch_fraction,
    )
    return config
