from pathlib import Path
from typing import Any, Dict, Optional

from floods.config import TrainConfig
from floods.utils.common import config_to_plain_dict, get_logger

LOG = get_logger(__name__)


def resolve_resume_checkpoint(config: TrainConfig, model_folder: Path) -> Optional[Path]:
    """Resolve a resumable checkpoint before model construction.

    ``--resume`` is intentionally strict: it must never silently start a new run.
    An interrupted run must never be mistaken for a fresh run that restarts at epoch 1.
    """
    explicit = Path(config.resume_from).expanduser() if config.resume_from else None
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {explicit}. "
                "Use a valid --resume-from path, or remove --resume to start a new run."
            )
        LOG.info("Resume requested: using explicit checkpoint %s", explicit)
        return explicit

    if not config.resume:
        return None

    primary = Path(model_folder) / "last.ckpt"
    if primary.exists():
        LOG.info("Resume requested: using %s", primary)
        return primary

    epoch_checkpoints = sorted(Path(model_folder).glob("epoch_*.ckpt"))
    if epoch_checkpoints:
        fallback = epoch_checkpoints[-1]
        LOG.warning("Resume checkpoint %s is missing; falling back to %s", primary, fallback)
        return fallback

    raise FileNotFoundError(
        "Resume was requested, but no resumable checkpoint exists. "
        f"Expected: {primary}. "
        "The run may have been started with save_last=false, the run-id may be wrong, "
        "or the checkpoint may not have finished syncing to its storage location. "
        "Do not delete the existing run folder. Verify the run-id and models directory, "
        "then retry with --resume-from PATH if a checkpoint exists elsewhere."
    )



def active_sampler_name(config: TrainConfig) -> str:
    """Return the single effective training sampler name for logging and resume checks."""
    ordered = (
        "hard_positive_region_sampling",
        "hard_negative_region_sampling",
        "hard_example_sampling",
        "event_balanced_sampling",
        "stratified_sampling",
        "foreground_balanced_sampling",
        "weighted_sampling",
    )
    for name in ordered:
        if bool(getattr(config.data, name, False)):
            return name
    return "standard_shuffle"


def build_resume_signature(config: TrainConfig) -> Dict[str, Any]:
    """Build a stable signature for settings that must not change during resume.

    Runtime-only values such as data location, output folder, progress display, device,
    and worker count are intentionally excluded. Training semantics are included so reconnecting or resuming cannot silently change the epoch target, sampler, or other optimisation behaviour.
    """
    plain = config_to_plain_dict(config)
    trainer = plain.get("trainer", {})
    data = plain.get("data", {})
    model = plain.get("model", {})
    optimizer = plain.get("optimizer", {})
    scheduler = plain.get("scheduler", {})
    loss = plain.get("loss", {})

    model_keys = (
        "encoder", "decoder", "multibranch", "output_stride", "act", "norm",
        "dropout2d", "freeze",
    )
    data_keys = (
        "train_source_split", "val_source_split", "train_include_events", "train_exclude_events",
        "val_include_events", "val_exclude_events",
        "in_channels", "include_dem", "input_modalities", "train_mask_body_ratio", "val_mask_body_ratio",
        "weighted_sampling", "foreground_balanced_sampling", "stratified_sampling",
        "event_balanced_sampling", "hard_example_sampling", "hard_negative_region_sampling", "hard_positive_region_sampling",
        "foreground_sample_ratio", "foreground_min_ratio", "weighted_samples_multiplier",
        "event_balance_power", "event_tile_weight_cap", "hard_example_csv",
        "hard_example_categories", "hard_example_fg_bins", "hard_example_max_f1",
        "hard_example_weight", "hard_example_max_fraction", "hard_positive_manifest",
        "hard_positive_region_weight", "hard_positive_region_max_fraction", "hard_positive_crop_probability", "hard_negative_manifest",
        "hard_negative_region_weight", "hard_negative_region_max_fraction",
        "hard_negative_crop_probability", "fg_bin_edges", "fg_bin_sample_weights",
        "augmentation_profile", "disable_random_crop", "disable_elastic",
        "disable_grid_distortion", "disable_sar_noise", "sparse_crop_supervision",
        "sparse_crop_normal_fraction", "sparse_crop_flood_fraction",
        "sparse_crop_hard_background_fraction", "sparse_crop_sizes", "sparse_crop_attempts",
        "sparse_crop_hard_background_max_fg_ratio", "sparse_crop_min_valid_ratio",
        "modality_dropout", "modality_dropout_prob", "drop_modalities",
        "normalization_mode", "normalization_stats_path", "pos_weight_from_train",
        "pos_weight_max",
    )
    trainer_keys = (
        "batch_size", "max_epochs", "patience", "threshold_sweep", "thresholds",
        "threshold_metric", "monitor_threshold_sweep", "metric_mode", "monitor",
        "grad_clip_norm", "skip_nonfinite_batches", "amp_full_precision_retry", "max_skipped_batch_fraction",
        "event_macro_validation",
        "group_dro", "group_dro_eta", "group_dro_min_weight", "group_dro_warmup_epochs",
    )

    return {
        "image_size": plain.get("image_size"),
        "sampler": active_sampler_name(config),
        "init_channel_adaptation": plain.get("init_channel_adaptation"),
        "model": {key: model.get(key) for key in model_keys if key in model},
        "data": {key: data.get(key) for key in data_keys if key in data},
        "trainer": {key: trainer.get(key) for key in trainer_keys if key in trainer},
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss": loss,
    }


def diff_resume_signatures(expected: Dict[str, Any], actual: Dict[str, Any], prefix: str = "") -> list[str]:
    """Return human-readable differences between two nested resume signatures."""
    differences: list[str] = []
    keys = sorted(set(expected) | set(actual))
    for key in keys:
        path = f"{prefix}.{key}" if prefix else str(key)
        left = expected.get(key, "<missing>")
        right = actual.get(key, "<missing>")
        if isinstance(left, dict) and isinstance(right, dict):
            differences.extend(diff_resume_signatures(left, right, prefix=path))
        elif left != right:
            differences.append(f"{path}: saved={left!r}, requested={right!r}")
    return differences
