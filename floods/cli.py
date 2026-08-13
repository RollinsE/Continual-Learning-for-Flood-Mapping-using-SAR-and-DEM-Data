from __future__ import annotations

import argparse
import logging
from copy import deepcopy
import os
import shlex
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Type

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "2")
warnings.filterwarnings("ignore", message="Importing from timm.models.features is deprecated.*", category=FutureWarning)

import yaml

from floods.config import PreparationConfig, StatsConfig, TrainConfig
from floods.config.testing import TestConfig
from floods.config.training import Losses, Metrics, Optimizers, Schedulers
from floods.utils.common import command_logging, prepare_logging
from floods.utils.console import progress_logging_context

LOG = logging.getLogger(__name__)


def _load_trusted_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        # Saved run configs may contain Python object tags. Only project-owned or
        # user-trusted configs should be passed to the CLI.
        data = yaml.load(text, Loader=yaml.FullLoader)
    return data or {}


def _load_config_from_yaml(path: Path, config_cls: Type[Any]) -> Any:
    data = _load_trusted_yaml(path)
    return config_cls(**data)


def _load_training_config_for_args(args: argparse.Namespace) -> TrainConfig:
    """Load a fresh-run config or the preserved run config for strict resume.

    A resume command may point at the original baseline YAML for convenience, but once
    a run directory exists its own config.yaml is authoritative. Explicit CLI changes
    are applied to a copy and rejected when they alter training semantics. Runtime-only
    changes such as data location, device, worker count, and progress output remain safe.
    """
    base = _load_config_from_yaml(args.config, TrainConfig)
    candidate = _apply_training_overrides(base, args)
    resume_requested = bool(candidate.resume or candidate.resume_from or candidate.trainer.extend_epochs)
    if not resume_requested:
        return candidate

    if not candidate.name:
        raise ValueError("Resume requires a stable --run-id (or name in the config) so the saved run configuration can be located.")
    run_config_path = Path(candidate.output_folder) / str(candidate.name) / "config.yaml"
    if not run_config_path.exists():
        raise FileNotFoundError(
            f"Resume run configuration not found: {run_config_path}. "
            "Do not resume from a baseline config into an unidentified run. Verify --artifacts-dir and --run-id."
        )

    saved = _load_config_from_yaml(run_config_path, TrainConfig)
    requested = _apply_training_overrides(deepcopy(saved), args)

    from floods.resume import build_resume_signature, diff_resume_signatures
    differences = diff_resume_signatures(build_resume_signature(saved), build_resume_signature(requested))
    if differences:
        detail = "\n  - ".join(differences[:20])
        suffix = "" if len(differences) <= 20 else f"\n  - ... and {len(differences) - 20} more"
        raise ValueError(
            "Resume command attempted to change the saved training plan. Resume must continue the same experiment. "
            "Use --extend-epochs to add epochs, or choose a new --run-id for a changed experiment.\n"
            f"  - {detail}{suffix}"
        )

    # init_checkpoint records how the run was originally initialised. During a
    # strict resume, model and optimiser state must come only from last.ckpt (or
    # an explicit resume checkpoint), so do not reapply the provenance checkpoint.
    requested.init_checkpoint = None

    LOG.info("Resume configuration loaded from preserved run config: %s", run_config_path)
    return requested


def _normalise_enum_option(enum_cls: Type[Any], value: Optional[str], *, aliases: Optional[dict[str, str]] = None) -> Any:
    if value is None:
        return None
    key = value.lower().strip().replace("-", "_")
    aliases = aliases or {}
    key = aliases.get(key, key)
    if key.startswith("val_"):
        key = key[4:]
    if key.startswith("validation_"):
        key = key[11:]
    try:
        return enum_cls[key]
    except KeyError as exc:
        allowed = ", ".join(e.name for e in enum_cls)
        raise argparse.ArgumentTypeError(f"Invalid value '{value}'. Choose one of: {allowed}") from exc


def _set_when_provided(obj: Any, attr: str, value: Any) -> None:
    if value is not None:
        setattr(obj, attr, value)


def _set_input_modalities(config: TrainConfig, values: Optional[Iterable[str]]) -> None:
    if not values:
        return
    from floods.modalities import canonicalize_modalities
    modalities = canonicalize_modalities(list(values))
    config.data.input_modalities = modalities
    config.data.in_channels = len(modalities)
    config.data.include_dem = "dem" in modalities
    if hasattr(config.data, "refresh_cache_hash"):
        config.data.refresh_cache_hash()


def _activate_training_sampler(config: TrainConfig, sampler_flag: str) -> None:
    """Enable one replacement sampler and disable the other sampler modes."""
    sampler_flags = [
        "weighted_sampling",
        "foreground_balanced_sampling",
        "stratified_sampling",
        "event_balanced_sampling",
        "hard_example_sampling",
        "hard_negative_region_sampling",
        "hard_positive_region_sampling",
    ]
    for flag in sampler_flags:
        setattr(config.data, flag, flag == sampler_flag)


def _apply_sampler_overrides(config: TrainConfig, args: argparse.Namespace) -> None:
    """Apply CLI sampler choices without leaving incompatible modes enabled."""
    positive_choices = []
    mapping = {
        "weighted_sampling": getattr(args, "weighted_sampling", None),
        "foreground_balanced_sampling": getattr(args, "foreground_balanced_sampling", None),
        "stratified_sampling": getattr(args, "stratified_sampling", None),
        "event_balanced_sampling": getattr(args, "event_balanced_sampling", None),
        "hard_example_sampling": getattr(args, "hard_example_sampling", None),
        "hard_negative_region_sampling": getattr(args, "hard_negative_region_sampling", None),
        "hard_positive_region_sampling": getattr(args, "hard_positive_region_sampling", None),
    }
    for flag, value in mapping.items():
        if value is True:
            positive_choices.append(flag)
    if len(positive_choices) > 1:
        choices = ", ".join(f"--{flag.replace('_', '-')}" for flag in positive_choices)
        raise argparse.ArgumentTypeError(f"Only one training sampler can be enabled in one command. Requested: {choices}")
    if positive_choices:
        _activate_training_sampler(config, positive_choices[0])
        return
    for flag, value in mapping.items():
        if value is False:
            setattr(config.data, flag, False)


def _add_pretrained_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Build the model with pretrained encoder weights before loading the checkpoint.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Build the model without downloading pretrained encoder weights; recommended when evaluating/resuming a full checkpoint.",
    )


def _apply_eval_model_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    _set_when_provided(config.model, "pretrained", getattr(args, "pretrained", None))
    return config


def _apply_training_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    _set_when_provided(config.data, "path", args.processed_data_dir)
    _set_when_provided(config, "output_folder", args.output_folder)
    _set_when_provided(config, "name", args.run_id)
    _set_when_provided(config, "seed", args.seed)
    _set_when_provided(config, "image_size", args.image_size)
    _set_when_provided(config, "init_checkpoint", args.init_checkpoint)
    _set_when_provided(config, "init_channel_adaptation", args.init_channel_adaptation)
    _set_when_provided(config.trainer, "max_epochs", args.epochs)
    _set_when_provided(config.trainer, "batch_size", args.batch_size)
    _set_when_provided(config.trainer, "num_workers", args.num_workers)
    _set_when_provided(config.trainer, "patience", args.patience)
    _set_when_provided(config.trainer, "amp", args.amp)
    _set_when_provided(config.trainer, "cpu", args.cpu)
    _set_when_provided(config.trainer, "save_last", args.save_last)
    _set_when_provided(config.trainer, "save_epoch_checkpoints", args.save_epoch_checkpoints)
    _set_when_provided(config.trainer, "extend_epochs", args.extend_epochs)
    _set_when_provided(config.trainer, "reset_early_stopping_on_resume", args.reset_early_stopping_on_resume)
    _set_when_provided(config.trainer, "grad_clip_norm", args.grad_clip_norm)
    _set_when_provided(config.trainer, "detect_anomaly", args.detect_anomaly)
    _set_when_provided(config.trainer, "skip_nonfinite_batches", args.skip_nonfinite_batches)
    _set_when_provided(config.trainer, "amp_full_precision_retry", args.amp_full_precision_retry)
    _set_when_provided(config.trainer, "max_skipped_batch_fraction", args.max_skipped_batch_fraction)
    _set_when_provided(config.trainer, "progress_bar", args.progress_bar)
    _set_when_provided(config.trainer, "progress_log_interval", args.progress_log_interval)
    _set_when_provided(config.trainer, "progress_label", args.progress_label)
    _set_when_provided(config.trainer, "threshold_sweep", args.threshold_sweep)
    _set_when_provided(config.trainer, "threshold_metric", args.threshold_metric)
    _set_when_provided(config.trainer, "metric_mode", args.metric_mode)
    _set_when_provided(config.trainer, "group_dro", args.group_dro)
    _set_when_provided(config.trainer, "group_dro_eta", args.group_dro_eta)
    _set_when_provided(config.trainer, "group_dro_min_weight", args.group_dro_min_weight)
    _set_when_provided(config.trainer, "group_dro_warmup_epochs", args.group_dro_warmup_epochs)
    if args.thresholds:
        config.trainer.thresholds = [float(v) for v in args.thresholds]
    _set_when_provided(config.optimizer, "lr", args.lr)
    _set_when_provided(config.optimizer, "encoder_lr", args.encoder_lr)
    _set_when_provided(config.optimizer, "decoder_lr", args.decoder_lr)
    _set_when_provided(config.optimizer, "weight_decay", args.weight_decay)
    _set_when_provided(config.data, "in_channels", args.in_channels)
    _set_when_provided(config.data, "include_dem", args.include_dem)
    if args.mask_body_ratio is not None:
        config.data.mask_body_ratio = args.mask_body_ratio
        config.data.train_mask_body_ratio = args.mask_body_ratio
    _set_when_provided(config.data, "train_mask_body_ratio", args.train_mask_body_ratio)
    _set_when_provided(config.data, "val_mask_body_ratio", args.val_mask_body_ratio)
    _set_when_provided(config.data, "test_mask_body_ratio", args.test_mask_body_ratio)
    _apply_sampler_overrides(config, args)
    _set_when_provided(config.data, "foreground_sample_ratio", args.foreground_sample_ratio)
    _set_when_provided(config.data, "foreground_min_ratio", args.foreground_min_ratio)
    _set_when_provided(config.data, "weighted_samples_multiplier", args.weighted_samples_multiplier)
    _set_when_provided(config.data, "event_balance_power", args.event_balance_power)
    _set_when_provided(config.data, "event_tile_weight_cap", args.event_tile_weight_cap)
    _set_when_provided(config.data, "hard_example_csv", args.hard_example_csv)
    _set_when_provided(config.data, "hard_example_weight", args.hard_example_weight)
    _set_when_provided(config.data, "hard_example_max_fraction", args.hard_example_max_fraction)
    _set_when_provided(config.data, "hard_example_max_f1", args.hard_example_max_f1)
    _set_when_provided(config.data, "hard_positive_manifest", args.hard_positive_manifest)
    _set_when_provided(config.data, "hard_positive_region_weight", args.hard_positive_region_weight)
    _set_when_provided(config.data, "hard_positive_region_max_fraction", args.hard_positive_region_max_fraction)
    _set_when_provided(config.data, "hard_positive_crop_probability", args.hard_positive_crop_probability)
    _set_when_provided(config.data, "hard_negative_manifest", args.hard_negative_manifest)
    _set_when_provided(config.data, "hard_negative_region_weight", args.hard_negative_region_weight)
    _set_when_provided(config.data, "hard_negative_region_max_fraction", args.hard_negative_region_max_fraction)
    _set_when_provided(config.data, "hard_negative_crop_probability", args.hard_negative_crop_probability)
    if args.hard_example_categories:
        config.data.hard_example_categories = [str(v) for v in args.hard_example_categories]
    if args.hard_example_fg_bins:
        config.data.hard_example_fg_bins = [str(v).lower() for v in args.hard_example_fg_bins]
    _set_when_provided(config.data, "pos_weight_from_train", args.pos_weight_from_train)
    _set_when_provided(config.data, "pos_weight_max", args.pos_weight_max)
    if args.fg_bin_edges:
        config.data.fg_bin_edges = [float(v) for v in args.fg_bin_edges]
    if args.fg_bin_sample_weights:
        config.data.fg_bin_sample_weights = [float(v) for v in args.fg_bin_sample_weights]
    _set_when_provided(config.data, "augmentation_profile", args.augmentation_profile)
    _set_when_provided(config.data, "disable_random_crop", args.disable_random_crop)
    _set_when_provided(config.data, "disable_elastic", args.disable_elastic)
    _set_when_provided(config.data, "disable_grid_distortion", args.disable_grid_distortion)
    _set_when_provided(config.data, "disable_sar_noise", args.disable_sar_noise)
    _set_when_provided(config.data, "sparse_crop_supervision", args.sparse_crop_supervision)
    _set_when_provided(config.data, "sparse_crop_normal_fraction", args.sparse_crop_normal_fraction)
    _set_when_provided(config.data, "sparse_crop_flood_fraction", args.sparse_crop_flood_fraction)
    _set_when_provided(config.data, "sparse_crop_hard_background_fraction", args.sparse_crop_hard_background_fraction)
    if args.sparse_crop_sizes:
        config.data.sparse_crop_sizes = [int(v) for v in args.sparse_crop_sizes]
    _set_when_provided(config.data, "sparse_crop_attempts", args.sparse_crop_attempts)
    _set_when_provided(config.data, "sparse_crop_hard_background_max_fg_ratio", args.sparse_crop_hard_background_max_fg_ratio)
    _set_when_provided(config.data, "sparse_crop_min_valid_ratio", args.sparse_crop_min_valid_ratio)
    _set_when_provided(config.data, "normalization_stats_path", args.normalization_stats_path)
    _set_when_provided(config.data, "normalization_mode", args.normalization_mode)
    _set_when_provided(config.data, "modality_dropout", args.modality_dropout)
    _set_when_provided(config.data, "modality_dropout_prob", args.modality_dropout_prob)
    if args.drop_modalities:
        config.data.drop_modalities = [str(m).lower() for m in args.drop_modalities]
    _set_when_provided(config.trainer, "monitor_threshold_sweep", args.monitor_threshold_sweep)
    _set_when_provided(config.data, "cache_dir", args.cache_dir)
    _set_when_provided(config.data, "clear_cache", args.clear_cache)
    _set_when_provided(config.model, "encoder", args.encoder)
    _set_when_provided(config.model, "decoder", args.decoder)
    _set_when_provided(config.model, "pretrained", args.pretrained)
    _set_when_provided(config.model, "freeze", args.freeze_encoder)
    _set_when_provided(config.model, "output_stride", args.output_stride)
    _set_when_provided(config.model, "act", args.activation)
    _set_when_provided(config.model, "norm", args.norm_layer)
    _set_when_provided(config.model, "dropout2d", args.dropout2d)
    _set_when_provided(config, "visualize", args.visualize)

    _set_input_modalities(config, args.input_modalities)

    optimizer = _normalise_enum_option(Optimizers, args.optimizer) if args.optimizer else None
    scheduler = _normalise_enum_option(Schedulers, args.scheduler) if args.scheduler else None
    loss = _normalise_enum_option(Losses, args.loss) if args.loss else None
    monitor = _normalise_enum_option(Metrics, args.monitor or args.selection_metric) if (args.monitor or args.selection_metric) else None
    _set_when_provided(config.optimizer, "target", optimizer)
    _set_when_provided(config.scheduler, "target", scheduler)
    _set_when_provided(config.loss, "target", loss)
    _set_when_provided(config.loss, "alpha", args.loss_alpha)
    _set_when_provided(config.loss, "beta", args.loss_beta)
    _set_when_provided(config.loss, "gamma", args.loss_gamma)
    _set_when_provided(config.loss, "focal_gamma", args.focal_gamma)
    _set_when_provided(config.loss, "focal_alpha", args.focal_alpha)
    _set_when_provided(config.loss, "bce_weight", args.bce_weight)
    _set_when_provided(config.loss, "focal_weight", args.focal_weight)
    _set_when_provided(config.loss, "tversky_weight", args.tversky_weight)
    _set_when_provided(config.trainer, "monitor", monitor)

    if args.resume_from:
        config.resume_from = str(Path(args.resume_from))
        config.resume = True
    elif args.resume is not None:
        config.resume = args.resume

    if args.extend_epochs is not None:
        config.resume = True
        if args.reset_early_stopping_on_resume is None:
            config.trainer.reset_early_stopping_on_resume = True

    _set_when_provided(config.data, "source_sar_transform", getattr(args, "source_sar_transform", None))
    _set_when_provided(config.model, "foundation_input_size", getattr(args, "foundation_input_size", None))
    _set_when_provided(config.model, "foundation_pyramid_channels", getattr(args, "foundation_pyramid_channels", None))

    from floods.pretrained import apply_resolved_model_to_config, normalize_source_name, resolve_model_spec
    explicit_source = getattr(args, "weights_source", None) is not None
    source_value = normalize_source_name(
        getattr(args, "weights_source", None) or getattr(config.model, "weights_source", "random")
    )
    if source_value == "random" and bool(config.model.pretrained) and not explicit_source:
        source_value = "imagenet"

    # Empty modality lists are valid in authoritative legacy configurations.  They
    # continue to be resolved later by prepare_datasets.  Registered EO providers,
    # however, require an explicit channel contract and fail here before training.
    modalities = list(getattr(config.data, "input_modalities", None) or [])
    provider_source = source_value not in {"random", "imagenet"}
    if provider_source and not modalities:
        from floods.modalities import resolve_input_modalities
        modalities = resolve_input_modalities(
            None,
            in_channels=config.data.in_channels,
            include_dem=config.data.include_dem,
            use_rgb=False,
        )

    requested_pretrained = getattr(args, "pretrained", None)
    if requested_pretrained is None and not explicit_source:
        requested_pretrained = config.model.pretrained
    resolved = resolve_model_spec(
        weights_source=source_value,
        modalities=modalities,
        encoder=(getattr(args, "encoder", None) if provider_source else config.model.encoder),
        decoder=(getattr(args, "decoder", None) if provider_source else config.model.decoder),
        pretrained=requested_pretrained,
    )
    apply_resolved_model_to_config(
        config,
        resolved,
        evaluation=False,
        force_normalization=bool(explicit_source or provider_source),
    )

    # The epoch summary always reports the key segmentation metrics.
    required_metrics = [Metrics.f1, Metrics.iou, Metrics.mcc, Metrics.precision, Metrics.recall]
    for metric in required_metrics:
        if metric not in config.trainer.val_metrics:
            config.trainer.val_metrics.append(metric)
    if config.trainer.monitor not in config.trainer.val_metrics:
        config.trainer.val_metrics.append(config.trainer.monitor)

    if hasattr(config.data, "refresh_cache_hash"):
        config.data.refresh_cache_hash()

    return config


def _apply_preprocess_overrides(config: PreparationConfig, args: argparse.Namespace) -> PreparationConfig:
    _set_when_provided(config, "data_source", args.raw_data_dir)
    _set_when_provided(config, "data_processed", args.processed_data_dir)
    _set_when_provided(config, "summary_file", args.summary_file)
    _set_when_provided(config, "tile_size", args.tile_size)
    _set_when_provided(config, "tile_max_overlap", args.tile_max_overlap)
    _set_when_provided(config, "nan_threshold", args.nan_threshold)
    _set_when_provided(config, "morph_kernel", args.morph_kernel)
    _set_when_provided(config, "decibel", args.decibel)
    _set_when_provided(config, "colorize", args.colorize)
    _set_when_provided(config, "sar_transform", args.sar_transform)
    _set_when_provided(config, "preserve_mask_ignore", args.preserve_mask_ignore)
    _set_when_provided(config, "align_to_reference_grid", args.align_to_reference_grid)
    if args.mask_flood_values:
        config.mask_flood_values = [int(v) for v in args.mask_flood_values]
    if args.mask_background_values:
        config.mask_background_values = [int(v) for v in args.mask_background_values]
    if args.mask_ignore_values:
        config.mask_ignore_values = [int(v) for v in args.mask_ignore_values]
    _set_when_provided(config, "clip_dem", args.clip_dem)
    _set_when_provided(config, "morphology", args.morphology)
    _set_when_provided(config, "tiling", args.tiling)
    _set_when_provided(config, "make_context", args.make_context)
    if args.subset:
        config.subset = set(args.subset)
    if args.scale:
        config.scale = args.scale
    return config


def _apply_stats_overrides(config: StatsConfig, args: argparse.Namespace) -> StatsConfig:
    _set_when_provided(config, "data_root", args.processed_data_dir)
    _set_when_provided(config, "subset", args.subset)
    return config




def _fill_missing_training_args(args: argparse.Namespace) -> argparse.Namespace:
    """Populate optional train override fields that a specialised parser omits."""
    defaults = {
        "epochs": None, "mask_body_ratio": None, "weighted_sampling": None,
        "foreground_balanced_sampling": None, "foreground_sample_ratio": None, "foreground_min_ratio": None,
        "weighted_samples_multiplier": None, "stratified_sampling": None, "event_balanced_sampling": None, "event_balance_power": None, "event_tile_weight_cap": None,
        "hard_example_sampling": None, "hard_example_csv": None, "hard_example_categories": None,
        "hard_example_fg_bins": None, "hard_example_max_f1": None, "hard_example_weight": None,
        "hard_example_max_fraction": None, "hard_negative_region_sampling": None,
        "hard_positive_region_sampling": None, "hard_positive_manifest": None,
        "hard_positive_region_weight": None, "hard_positive_region_max_fraction": None,
        "hard_positive_crop_probability": None,
        "hard_negative_manifest": None, "hard_negative_region_weight": None,
        "hard_negative_region_max_fraction": None, "hard_negative_crop_probability": None,
        "fg_bin_edges": None, "fg_bin_sample_weights": None,
        "init_checkpoint": None, "init_channel_adaptation": None, "resume": None, "resume_from": None, "extend_epochs": None, "reset_early_stopping_on_resume": None,
        "save_last": None, "save_epoch_checkpoints": None, "detect_anomaly": None,
        "amp_full_precision_retry": None, "max_skipped_batch_fraction": None, "visualize": False,
        "group_dro": None, "group_dro_eta": None, "group_dro_min_weight": None,
        "group_dro_warmup_epochs": None,
        "monitor_threshold_sweep": None, "clear_cache": None, "cache_dir": None,
        "sparse_crop_supervision": None, "sparse_crop_normal_fraction": None,
        "sparse_crop_flood_fraction": None, "sparse_crop_hard_background_fraction": None,
        "sparse_crop_sizes": None, "sparse_crop_attempts": None,
        "weights_source": None, "source_sar_transform": None,
        "foundation_input_size": None, "foundation_pyramid_channels": None,
        "sparse_crop_hard_background_max_fg_ratio": None, "sparse_crop_min_valid_ratio": None,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="floodmap",
        description="CLI for flood-extent preprocessing, training, evaluation, auditing, and deployment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a segmentation model from a YAML config.")
    train_p.add_argument("--config", type=Path, required=True, help="Path to a trusted training YAML.")
    train_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", help="Processed tiled dataset root.")
    train_p.add_argument("--output-folder", "--artifacts-dir", dest="output_folder", help="Run/artifacts root folder.")
    train_p.add_argument("--run-id", "--name", dest="run_id", help="Stable run identifier under the output folder.")
    train_p.add_argument("--seed", type=int)
    train_p.add_argument("--image-size", type=int, choices=[128, 256, 512], help="Model input size; normally match preprocessing --tile-size.")
    train_p.add_argument("--epochs", "--max-epochs", dest="epochs", type=int)
    train_p.add_argument("--batch-size", type=int)
    train_p.add_argument("--num-workers", type=int)
    train_p.add_argument("--patience", type=int)
    train_p.add_argument("--lr", type=float)
    train_p.add_argument("--encoder-lr", type=float)
    train_p.add_argument("--decoder-lr", type=float)
    train_p.add_argument("--weight-decay", type=float)
    train_p.add_argument("--optimizer", choices=[e.name for e in Optimizers])
    train_p.add_argument("--scheduler", choices=[e.name for e in Schedulers])
    train_p.add_argument("--loss", choices=[e.name for e in Losses])
    train_p.add_argument("--loss-alpha", type=float, help="Tversky alpha / false-positive penalty, e.g. 0.3")
    train_p.add_argument("--loss-beta", type=float, help="Tversky beta / false-negative penalty, e.g. 0.7")
    train_p.add_argument("--loss-gamma", type=float, help="Tversky/focal gamma used by the selected loss")
    train_p.add_argument("--focal-gamma", type=float, help="Focal BCE gamma for focal_tversky_combo")
    train_p.add_argument("--focal-alpha", type=float, help="Focal BCE alpha multiplier for focal_tversky_combo")
    train_p.add_argument("--bce-weight", type=float, help="BCE term weight for bce_tversky")
    train_p.add_argument("--focal-weight", type=float, help="Focal BCE term weight for focal_tversky_combo")
    train_p.add_argument("--tversky-weight", type=float, help="Tversky term weight for combined losses")
    train_p.add_argument("--monitor", choices=[e.name for e in Metrics] + [f"val_{e.name}" for e in Metrics])
    train_p.add_argument("--selection-metric", help="Alias for --monitor, e.g. val_f1.")
    train_p.add_argument("--encoder", "--backbone", dest="encoder", help="Encoder/backbone, e.g. resnet34. Alias: --backbone.")
    train_p.add_argument("--decoder", "--architecture", dest="decoder", help="Segmentation architecture/decoder: unet, unetpp, deeplabv3p, deeplabv3, pspnet, or segformer. Alias: --architecture.")
    train_p.add_argument("--weights-source", help="Registered initialisation source; see `floodmap model-catalog`.")
    train_p.add_argument("--foundation-input-size", type=int, help="Internal provider input size; normally resolved automatically.")
    train_p.add_argument("--foundation-pyramid-channels", type=int, help="Foundation decoder projection width.")
    train_p.add_argument("--source-sar-transform", choices=["auto", "linear", "log1p", "db10"], help="Processed SAR representation before provider normalization.")
    train_p.add_argument("--pretrained", dest="pretrained", action="store_true", default=None, help="Use pretrained encoder weights.")
    train_p.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Train encoder from scratch.")
    train_p.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true", default=None, help="Freeze the encoder/backbone.")
    train_p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false", help="Do not freeze the encoder/backbone.")
    train_p.add_argument("--output-stride", type=int, choices=[8, 16, 32], help="Encoder output stride where supported by the architecture.")
    train_p.add_argument("--activation", choices=["ident", "relu", "lrelu"], help="Activation layer used by configurable decoders.")
    train_p.add_argument("--norm-layer", choices=["std", "iabn", "iabn_sync"], help="Normalization layer used by configurable decoders.")
    train_p.add_argument("--dropout2d", dest="dropout2d", action="store_true", default=None, help="Enable 2D dropout in supported decoders.")
    train_p.add_argument("--no-dropout2d", dest="dropout2d", action="store_false", help="Disable 2D dropout in supported decoders.")
    train_p.add_argument("--in-channels", type=int)
    train_p.add_argument("--input-modalities", nargs="+", help="Example: --input-modalities vv vh dem")
    train_p.add_argument("--mask-body-ratio", type=float, help="Alias for --train-mask-body-ratio. Validation remains unfiltered unless --val-mask-body-ratio is set.")
    train_p.add_argument("--train-mask-body-ratio", type=float, help="Minimum foreground-mask ratio to keep a training tile. Use 0.0 to disable filtering.")
    train_p.add_argument("--val-mask-body-ratio", type=float, help="Optional validation foreground-mask filter. Defaults to 0.0 so validation remains unfiltered.")
    train_p.add_argument("--test-mask-body-ratio", type=float, help="Optional test foreground-mask filter for compatible evaluation commands.")
    train_p.add_argument("--cache-dir", help="Directory for mask-filter and sampler caches.")
    train_p.add_argument("--clear-cache", action="store_true", default=None, help="Recompute mask-filter and sampler caches for this run.")
    train_p.add_argument("--init-checkpoint", help="Warm-start model weights from a .pth/.ckpt while resetting optimiser, scheduler, epoch counter, and early stopping.")
    train_p.add_argument("--init-channel-adaptation", choices=["strict", "zero_extra"], help="Allow a 3-channel checkpoint to initialise a wider model by copying existing input weights and zero-initialising added channels.")
    train_p.add_argument("--resume", dest="resume", action="store_true", default=None, help="Resume from models/last.ckpt if it exists.")
    train_p.add_argument("--no-resume", dest="resume", action="store_false", help="Disable automatic resume.")
    train_p.add_argument("--resume-from", help="Explicit resumable checkpoint path, normally last.ckpt.")
    train_p.add_argument("--extend-epochs", type=int, help="When resuming, run this many additional epochs beyond the checkpoint epoch. Example: a completed 20-epoch run plus --extend-epochs 10 trains to epoch 30.")
    train_p.add_argument("--reset-early-stopping", dest="reset_early_stopping_on_resume", action="store_true", default=None, help="Reset the early-stopping patience counter after loading a resume checkpoint.")
    train_p.add_argument("--keep-early-stopping-state", dest="reset_early_stopping_on_resume", action="store_false", help="Keep the early-stopping patience counter stored in the resume checkpoint.")
    train_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    train_p.add_argument("--no-amp", dest="amp", action="store_false")
    train_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    train_p.add_argument("--gpu", dest="cpu", action="store_false")
    train_p.add_argument("--include-dem", dest="include_dem", action="store_true", default=None)
    train_p.add_argument("--no-include-dem", dest="include_dem", action="store_false")
    train_p.add_argument("--weighted-sampling", dest="weighted_sampling", action="store_true", default=None)
    train_p.add_argument("--no-weighted-sampling", dest="weighted_sampling", action="store_false")
    train_p.add_argument("--foreground-balanced-sampling", dest="foreground_balanced_sampling", action="store_true", default=None, help="Use a replacement sampler that balances empty and non-empty flood tiles without deleting data.")
    train_p.add_argument("--no-foreground-balanced-sampling", dest="foreground_balanced_sampling", action="store_false")
    train_p.add_argument("--foreground-sample-ratio", type=float, help="Target fraction of sampled training tiles that contain foreground. Example: 0.65")
    train_p.add_argument("--foreground-min-ratio", type=float, help="Minimum foreground ratio used to classify a tile as foreground for balanced sampling.")
    train_p.add_argument("--weighted-samples-multiplier", type=float, help="Samples per epoch as a multiplier of the training set length. Use 1.0 to avoid longer epochs.")
    train_p.add_argument("--stratified-sampling", dest="stratified_sampling", action="store_true", default=None, help="Use foreground-ratio bins for replacement sampling: empty/tiny/small/medium/large.")
    train_p.add_argument("--no-stratified-sampling", dest="stratified_sampling", action="store_false")
    train_p.add_argument("--event-balanced-sampling", dest="event_balanced_sampling", action="store_true", default=None, help="Balance by EMSR event, then foreground-ratio bin within event.")
    train_p.add_argument("--event-balance-power", type=float, help="Event-mass exponent. Use 0.5 for square-root tempered event balancing; 0.0 reproduces equal event mass.")
    train_p.add_argument("--event-tile-weight-cap", type=float, help="Cap event-balanced tile weights at this multiple of the median positive weight. Use 5 for the audited tempered profile.")
    train_p.add_argument("--no-event-balanced-sampling", dest="event_balanced_sampling", action="store_false")
    train_p.add_argument("--group-dro", dest="group_dro", action="store_true", default=None, help="Optimise an event-level GroupDRO objective using EMSR event IDs from training tile names.")
    train_p.add_argument("--no-group-dro", dest="group_dro", action="store_false", help="Disable event-level GroupDRO and use the configured ordinary loss objective.")
    train_p.add_argument("--group-dro-eta", type=float, help="Exponentiated-gradient step size for event weights. Default: 0.01.")
    train_p.add_argument("--group-dro-min-weight", type=float, help="Minimum probability retained for each event. Default: 0.001.")
    train_p.add_argument("--group-dro-warmup-epochs", type=int, help="ERM warm-up epochs before GroupDRO weighting starts. Default: 1.")
    train_p.add_argument("--hard-example-sampling", dest="hard_example_sampling", action="store_true", default=None, help="Oversample tiles identified by a train-split error-audit CSV.")
    train_p.add_argument("--no-hard-example-sampling", dest="hard_example_sampling", action="store_false")
    train_p.add_argument("--hard-example-csv", help="Path to train-split tile_error_metrics.csv from floodmap error-audit.")
    train_p.add_argument("--hard-example-categories", nargs="+", help="Error categories to oversample from the audit CSV.")
    train_p.add_argument("--hard-example-fg-bins", nargs="+", help="Foreground bins to oversample when tile F1 is low, e.g. tiny small.")
    train_p.add_argument("--hard-example-max-f1", type=float, help="Maximum tile F1 for hard foreground-bin selection.")
    train_p.add_argument("--hard-example-weight", type=float, help="Sampling weight multiplier for selected hard examples.")
    train_p.add_argument("--hard-example-max-fraction", type=float, help="Maximum probability mass assigned to hard examples after weighting.")
    train_p.add_argument("--hard-positive-region-sampling", dest="hard_positive_region_sampling", action="store_true", default=None, help="Oversample tiles with mined false-negative flood regions and crop directly around those audited errors.")
    train_p.add_argument("--no-hard-positive-region-sampling", dest="hard_positive_region_sampling", action="store_false")
    train_p.add_argument("--hard-positive-manifest", help="Path to hard_positive_regions.csv produced by floodmap mine-hard-positives.")
    train_p.add_argument("--hard-positive-region-weight", type=float, help="Relative sampling weight for tiles with hard-positive regions. Default: 3.0.")
    train_p.add_argument("--hard-positive-region-max-fraction", type=float, help="Maximum hard-positive probability mass. Default: 0.20.")
    train_p.add_argument("--hard-positive-crop-probability", type=float, help="Probability of using a mined positive crop on a matching tile. Default: 1.0.")
    train_p.add_argument("--hard-negative-region-sampling", dest="hard_negative_region_sampling", action="store_true", default=None, help="Oversample tiles with mined false-positive regions and crop directly around those audited errors.")
    train_p.add_argument("--no-hard-negative-region-sampling", dest="hard_negative_region_sampling", action="store_false")
    train_p.add_argument("--hard-negative-manifest", help="Path to hard_negative_regions.csv produced by floodmap mine-hard-negatives.")
    train_p.add_argument("--hard-negative-region-weight", type=float, help="Sampling weight multiplier for tiles with mined hard-negative regions. Default: 4.0.")
    train_p.add_argument("--hard-negative-region-max-fraction", type=float, help="Maximum probability mass assigned to region tiles. Default: 0.35.")
    train_p.add_argument("--hard-negative-crop-probability", type=float, help="Probability of applying a mined crop when a region tile is sampled. Default: 1.0.")
    train_p.add_argument("--pos-weight-from-train", dest="pos_weight_from_train", action="store_true", default=None, help="Compute a clipped BCE/focal positive-class weight from the filtered training masks.")
    train_p.add_argument("--no-pos-weight-from-train", dest="pos_weight_from_train", action="store_false")
    train_p.add_argument("--pos-weight-max", type=float, help="Maximum value for train-derived positive-class BCE/focal weight.")
    train_p.add_argument("--fg-bin-edges", nargs=4, type=float, help="Foreground-ratio bin edges for stratified sampling, e.g. 0 0.005 0.02 0.10")
    train_p.add_argument("--fg-bin-sample-weights", nargs=5, type=float, help="Target sampling weights for empty tiny small medium large bins, e.g. 0.20 0.20 0.25 0.25 0.10")
    train_p.add_argument("--augmentation-profile", help="Training augmentation profile: none, geometric, sar_radiometric, standard, crop_aware, deformation, or composite.")
    train_p.add_argument("--disable-random-crop", dest="disable_random_crop", action="store_true", default=None, help="Disable RandomSizedCrop in training augmentation.")
    train_p.add_argument("--enable-random-crop", dest="disable_random_crop", action="store_false")
    train_p.add_argument("--disable-elastic", dest="disable_elastic", action="store_true", default=None, help="Disable ElasticTransform in training augmentation.")
    train_p.add_argument("--enable-elastic", dest="disable_elastic", action="store_false")
    train_p.add_argument("--disable-grid-distortion", dest="disable_grid_distortion", action="store_true", default=None, help="Disable GridDistortion in training augmentation.")
    train_p.add_argument("--enable-grid-distortion", dest="disable_grid_distortion", action="store_false")
    train_p.add_argument("--disable-sar-noise", dest="disable_sar_noise", action="store_true", default=None, help="Disable SAR GaussianBlur/MultiplicativeNoise augmentation.")
    train_p.add_argument("--enable-sar-noise", dest="disable_sar_noise", action="store_false")
    train_p.add_argument("--sparse-crop-supervision", dest="sparse_crop_supervision", action="store_true", default=None, help="Enable 50/25/25 full-tile, flood-centred, and hard-background supervision on flood-containing tiles.")
    train_p.add_argument("--no-sparse-crop-supervision", dest="sparse_crop_supervision", action="store_false", help="Disable sparse-flood crop supervision.")
    train_p.add_argument("--sparse-crop-normal-fraction", type=float, help="Full-tile fraction within the flood-tile crop mixture. Default: 0.50.")
    train_p.add_argument("--sparse-crop-flood-fraction", type=float, help="Flood-centred crop fraction within the flood-tile crop mixture. Default: 0.25.")
    train_p.add_argument("--sparse-crop-hard-background-fraction", type=float, help="Hard-background crop fraction within the flood-tile crop mixture. Default: 0.25.")
    train_p.add_argument("--sparse-crop-sizes", nargs="+", type=int, help="Square crop sizes before resizing to image size, e.g. 256 320 384 448.")
    train_p.add_argument("--sparse-crop-attempts", type=int, help="Maximum hard-background crop search attempts. Default: 24.")
    train_p.add_argument("--sparse-crop-hard-background-max-fg-ratio", type=float, help="Maximum flood ratio allowed in a hard-background crop. Default: 0.001.")
    train_p.add_argument("--sparse-crop-min-valid-ratio", type=float, help="Minimum non-ignore-pixel ratio required in a crop. Default: 0.50.")
    train_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    train_p.add_argument("--normalization-mode", help="Normalization mode: fixed, stats, robust_percentile, or robust_minmax.")
    train_p.add_argument("--modality-dropout", dest="modality_dropout", action="store_true", default=None, help="Enable train-only modality dropout after normalization, usually for DEM regularisation.")
    train_p.add_argument("--no-modality-dropout", dest="modality_dropout", action="store_false")
    train_p.add_argument("--modality-dropout-prob", type=float, help="Probability of zeroing the selected modalities for each training sample, e.g. 0.30.")
    train_p.add_argument("--drop-modalities", nargs="+", help="Modalities to zero when dropout triggers, e.g. dem or vv vh. Active modalities only.")
    train_p.add_argument("--monitor-threshold-sweep", dest="monitor_threshold_sweep", action="store_true", default=None, help="Use best_<threshold_metric> for checkpoint selection when threshold sweep is enabled.")
    train_p.add_argument("--no-monitor-threshold-sweep", dest="monitor_threshold_sweep", action="store_false")
    train_p.add_argument("--save-last", dest="save_last", action="store_true", default=None)
    train_p.add_argument("--no-save-last", dest="save_last", action="store_false")
    train_p.add_argument("--save-epoch-checkpoints", dest="save_epoch_checkpoints", action="store_true", default=None)
    train_p.add_argument("--no-save-epoch-checkpoints", dest="save_epoch_checkpoints", action="store_false", help="Do not write an additional epoch_NNN.ckpt after every epoch.")
    train_p.add_argument("--grad-clip-norm", type=float, help="Clip gradients to this norm. Use 0 to disable.")
    train_p.add_argument("--detect-anomaly", dest="detect_anomaly", action="store_true", default=None, help="Enable PyTorch anomaly detection for debugging.")
    train_p.add_argument("--no-detect-anomaly", dest="detect_anomaly", action="store_false")
    train_p.add_argument("--skip-nonfinite-batches", dest="skip_nonfinite_batches", action="store_true", default=None)
    train_p.add_argument("--no-skip-nonfinite-batches", dest="skip_nonfinite_batches", action="store_false")
    train_p.add_argument("--amp-full-precision-retry", dest="amp_full_precision_retry", action="store_true", default=None, help="Retry an AMP-overflow batch once with autocast disabled before skipping it.")
    train_p.add_argument("--no-amp-full-precision-retry", dest="amp_full_precision_retry", action="store_false")
    train_p.add_argument("--max-skipped-batch-fraction", type=float, help="Abort when non-finite batches exceed this fraction of an epoch; 0 disables the budget.")
    train_p.add_argument("--visualize", dest="visualize", action="store_true", default=None)
    train_p.add_argument("--no-visualize", dest="visualize", action="store_false")
    train_p.add_argument("--progress", dest="progress_bar", action="store_true", default=None, help="Show live training and validation progress; plain mode reports roughly every 10%%.")
    train_p.add_argument("--quiet-progress", dest="progress_bar", action="store_false", help="Hide tqdm progress bars and print epoch summaries only unless --progress-log-interval is set.")
    train_p.add_argument("--progress-log-interval", type=int, help="When progress bars are hidden, optionally log training progress every N batches. Use 0 to disable heartbeat logs.")
    train_p.add_argument("--progress-label", help="Short label shown in the progress bar, e.g. Baseline.")
    train_p.add_argument("--threshold-sweep", dest="threshold_sweep", action="store_true", default=None, help="Report validation F1/IoU/MCC across multiple probability thresholds.")
    train_p.add_argument("--no-threshold-sweep", dest="threshold_sweep", action="store_false")
    train_p.add_argument("--thresholds", nargs="+", type=float, help="Thresholds for --threshold-sweep, e.g. 0.1 0.2 0.3 0.5")
    train_p.add_argument("--threshold-metric", choices=["f1", "iou", "mcc", "precision", "recall"], help="Metric used to select the best validation threshold.")
    train_p.add_argument("--metric-mode", choices=["global", "batch_average", "both"], help="Metric accounting mode. Training logs global metrics; evaluation can report batch-averaged metrics or both.")


    cl_p = sub.add_parser("continual-train", help="Train rehearsal-based continual-learning models over chronological EMSR tasks.")
    cl_p.add_argument("--config", type=Path, required=True, help="Path to a trusted training YAML.")
    cl_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", help="Processed tiled dataset root.")
    cl_p.add_argument("--output-folder", "--artifacts-dir", dest="output_folder", help="Run/artifacts root folder.")
    cl_p.add_argument("--run-id", "--name", dest="run_id", help="Base run identifier; the strategy name is appended automatically.")
    cl_p.add_argument("--activations-json-path", type=Path, required=True, help="MMFlood activations JSON with event start dates and split metadata.")
    cl_p.add_argument("--strategies", nargs="+", default=["random", "least_confidence", "margin", "entropy"], choices=["random", "least_confidence", "margin", "entropy"], help="Replay strategies to run.")
    cl_p.add_argument("--task-year-ranges", nargs="+", default=["2014-2017", "2018-2019", "2020-2021"], help="Chronological task ranges, e.g. 2014-2017 2018-2019 2020-2021.")
    cl_p.add_argument("--epochs-per-task", type=int, default=5, help="Number of epochs to train on each chronological task.")
    cl_p.add_argument("--replay-buffer-size", type=int, default=100, help="Maximum number of training samples retained for replay.")
    cl_p.add_argument("--replay-batch-size", type=int, default=16, help="Replay samples drawn from the buffer for each current-task batch.")
    cl_p.add_argument("--replay-mode", choices=["separate", "concat"], default="separate",
                      help="Memory-safe replay mode. 'separate' forwards current and replay batches separately; 'concat' preserves concatenated replay.")
    cl_p.add_argument("--uncertainty-subset-fraction", type=float, default=1.0, help="Fraction of the replay buffer scored by uncertainty strategies.")
    cl_p.add_argument("--resume", action="store_true", default=False,
                      help="Resume a continual-learning run from models/last.ckpt in the strategy/member output folder.")
    cl_p.add_argument("--resume-from", type=Path,
                      help="Resume from a specific CL last.ckpt or bare model_best*.pth checkpoint.")
    cl_p.add_argument("--resume-start-task", type=int,
                      help="One-based task number to start from when resuming from a bare model state dict.")
    cl_p.add_argument("--resume-start-epoch", type=int,
                      help="One-based epoch within the task to start from when resuming from a bare model state dict.")
    cl_p.add_argument("--cl-model-mode", choices=["single", "ensemble"], default="single", help="Train one CL model per strategy, or train multiple architecture members and evaluate them as an ensemble.")
    cl_p.add_argument("--ensemble-members", nargs="+", default=["unet:resnet50", "deeplabv3p:resnet50"], help="CL ensemble members as decoder:encoder[:label], e.g. unet:resnet50 deeplabv3p:resnet50.")
    cl_p.add_argument("--ensemble-method", choices=["mean_prob", "mean_logit"], default="mean_logit", help="How to combine CL ensemble members for validation evaluation.")
    cl_p.add_argument("--cl-eval-split", choices=["val", "test"], default="val", help="Split used for task-by-task CL evaluation matrix.")
    cl_p.add_argument("--cl-eval-inference-mode", choices=["direct", "sliding_window"], default="direct", help="Use sliding_window for full-raster test evaluation.")
    cl_p.add_argument("--window-size", type=int, default=512)
    cl_p.add_argument("--window-overlap", type=int, default=128)
    cl_p.add_argument("--window-batch-size", type=int, default=1)
    cl_p.add_argument("--seed", type=int)
    cl_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    cl_p.add_argument("--batch-size", type=int)
    cl_p.add_argument("--num-workers", type=int)
    cl_p.add_argument("--patience", type=int)
    cl_p.add_argument("--lr", type=float)
    cl_p.add_argument("--encoder-lr", type=float)
    cl_p.add_argument("--decoder-lr", type=float)
    cl_p.add_argument("--weight-decay", type=float)
    cl_p.add_argument("--optimizer", choices=[e.name for e in Optimizers])
    cl_p.add_argument("--scheduler", choices=[e.name for e in Schedulers])
    cl_p.add_argument("--loss", choices=[e.name for e in Losses])
    cl_p.add_argument("--loss-alpha", type=float)
    cl_p.add_argument("--loss-beta", type=float)
    cl_p.add_argument("--loss-gamma", type=float)
    cl_p.add_argument("--focal-gamma", type=float)
    cl_p.add_argument("--focal-alpha", type=float)
    cl_p.add_argument("--bce-weight", type=float)
    cl_p.add_argument("--focal-weight", type=float)
    cl_p.add_argument("--tversky-weight", type=float)
    cl_p.add_argument("--monitor", choices=[e.name for e in Metrics] + [f"val_{e.name}" for e in Metrics])
    cl_p.add_argument("--selection-metric", help="Alias for --monitor, e.g. val_f1.")
    cl_p.add_argument("--encoder", "--backbone", dest="encoder")
    cl_p.add_argument("--decoder", "--architecture", dest="decoder")
    cl_p.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    cl_p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    cl_p.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true", default=None)
    cl_p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    cl_p.add_argument("--output-stride", type=int, choices=[8, 16, 32])
    cl_p.add_argument("--activation", choices=["ident", "relu", "lrelu"])
    cl_p.add_argument("--norm-layer", choices=["std", "iabn", "iabn_sync"])
    cl_p.add_argument("--dropout2d", dest="dropout2d", action="store_true", default=None)
    cl_p.add_argument("--no-dropout2d", dest="dropout2d", action="store_false")
    cl_p.add_argument("--in-channels", type=int)
    cl_p.add_argument("--input-modalities", nargs="+")
    cl_p.add_argument("--include-dem", dest="include_dem", action="store_true", default=None)
    cl_p.add_argument("--no-include-dem", dest="include_dem", action="store_false")
    cl_p.add_argument("--train-mask-body-ratio", type=float)
    cl_p.add_argument("--val-mask-body-ratio", type=float)
    cl_p.add_argument("--test-mask-body-ratio", type=float)
    cl_p.add_argument("--cache-dir")
    cl_p.add_argument("--clear-cache", action="store_true", default=None)
    cl_p.add_argument("--pos-weight-from-train", dest="pos_weight_from_train", action="store_true", default=None)
    cl_p.add_argument("--no-pos-weight-from-train", dest="pos_weight_from_train", action="store_false")
    cl_p.add_argument("--pos-weight-max", type=float)
    cl_p.add_argument("--augmentation-profile")
    cl_p.add_argument("--disable-random-crop", dest="disable_random_crop", action="store_true", default=None)
    cl_p.add_argument("--enable-random-crop", dest="disable_random_crop", action="store_false")
    cl_p.add_argument("--disable-elastic", dest="disable_elastic", action="store_true", default=None)
    cl_p.add_argument("--enable-elastic", dest="disable_elastic", action="store_false")
    cl_p.add_argument("--disable-grid-distortion", dest="disable_grid_distortion", action="store_true", default=None)
    cl_p.add_argument("--enable-grid-distortion", dest="disable_grid_distortion", action="store_false")
    cl_p.add_argument("--disable-sar-noise", dest="disable_sar_noise", action="store_true", default=None)
    cl_p.add_argument("--enable-sar-noise", dest="disable_sar_noise", action="store_false")
    cl_p.add_argument("--sparse-crop-supervision", dest="sparse_crop_supervision", action="store_true", default=None)
    cl_p.add_argument("--no-sparse-crop-supervision", dest="sparse_crop_supervision", action="store_false")
    cl_p.add_argument("--sparse-crop-normal-fraction", type=float)
    cl_p.add_argument("--sparse-crop-flood-fraction", type=float)
    cl_p.add_argument("--sparse-crop-hard-background-fraction", type=float)
    cl_p.add_argument("--sparse-crop-sizes", nargs="+", type=int)
    cl_p.add_argument("--sparse-crop-attempts", type=int)
    cl_p.add_argument("--sparse-crop-hard-background-max-fg-ratio", type=float)
    cl_p.add_argument("--sparse-crop-min-valid-ratio", type=float)
    cl_p.add_argument("--normalization-stats-path")
    cl_p.add_argument("--normalization-mode")
    cl_p.add_argument("--modality-dropout", dest="modality_dropout", action="store_true", default=None)
    cl_p.add_argument("--no-modality-dropout", dest="modality_dropout", action="store_false")
    cl_p.add_argument("--modality-dropout-prob", type=float)
    cl_p.add_argument("--drop-modalities", nargs="+")
    cl_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    cl_p.add_argument("--no-amp", dest="amp", action="store_false")
    cl_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    cl_p.add_argument("--gpu", dest="cpu", action="store_false")
    cl_p.add_argument("--grad-clip-norm", type=float)
    cl_p.add_argument("--skip-nonfinite-batches", dest="skip_nonfinite_batches", action="store_true", default=None)
    cl_p.add_argument("--no-skip-nonfinite-batches", dest="skip_nonfinite_batches", action="store_false")
    cl_p.add_argument("--amp-full-precision-retry", dest="amp_full_precision_retry", action="store_true", default=None)
    cl_p.add_argument("--no-amp-full-precision-retry", dest="amp_full_precision_retry", action="store_false")
    cl_p.add_argument("--max-skipped-batch-fraction", type=float)
    cl_p.add_argument("--progress", dest="progress_bar", action="store_true", default=None)
    cl_p.add_argument("--quiet-progress", dest="progress_bar", action="store_false")
    cl_p.add_argument("--progress-log-interval", type=int)
    cl_p.add_argument("--progress-label")
    cl_p.add_argument("--threshold-sweep", dest="threshold_sweep", action="store_true", default=True)
    cl_p.add_argument("--no-threshold-sweep", dest="threshold_sweep", action="store_false")
    cl_p.add_argument("--thresholds", nargs="+", type=float)
    cl_p.add_argument("--threshold-metric", choices=["f1", "iou", "mcc", "precision", "recall"])
    cl_p.add_argument("--metric-mode", choices=["global", "batch_average", "both"])

    prep_p = sub.add_parser("preprocess", help="Run raw-data preprocessing/tiling.")
    prep_p.add_argument("--config", type=Path, required=True, help="Preparation YAML.")
    prep_p.add_argument("--raw-data-dir", "--data-source", dest="raw_data_dir", type=Path, help="Root containing EMSR*/s1_raw, DEM, and mask folders.")
    prep_p.add_argument("--processed-data-dir", "--data-processed", dest="processed_data_dir", type=Path, help="Output root for processed tiles.")
    prep_p.add_argument("--summary-file", type=str, help="Path to activations/split metadata JSON.")
    prep_p.add_argument("--subset", nargs="+", choices=["train", "val", "test"], help="Subsets to preprocess.")
    prep_p.add_argument("--scale", nargs="+", type=int, help="Tile scale factors, e.g. --scale 1 2.")
    prep_p.add_argument("--tile-size", type=int, choices=[128, 256, 512], help="Preprocessing tile size. Supported: 128, 256, 512.")
    prep_p.add_argument("--tile-max-overlap", type=int)
    prep_p.add_argument("--nan-threshold", type=float)
    prep_p.add_argument("--morph-kernel", type=int)
    prep_p.add_argument("--decibel", dest="decibel", action="store_true", default=None)
    prep_p.add_argument("--no-decibel", dest="decibel", action="store_false")
    prep_p.add_argument("--colorize", dest="colorize", action="store_true", default=None)
    prep_p.add_argument("--no-colorize", dest="colorize", action="store_false")
    prep_p.add_argument("--sar-transform", choices=["linear", "db10", "log1p"], help="SAR intensity transform used before tiling.")
    prep_p.add_argument("--mask-flood-values", nargs="+", type=int, help="Raw mask values mapped to flood/foreground value 1.")
    prep_p.add_argument("--mask-background-values", nargs="+", type=int, help="Raw mask values mapped to background value 0.")
    prep_p.add_argument("--mask-ignore-values", nargs="+", type=int, help="Raw mask values mapped to ignore value 255.")
    prep_p.add_argument("--preserve-mask-ignore", dest="preserve_mask_ignore", action="store_true", default=None, help="Restore ignored pixels after morphology.")
    prep_p.add_argument("--no-preserve-mask-ignore", dest="preserve_mask_ignore", action="store_false")
    prep_p.add_argument("--align-to-reference-grid", dest="align_to_reference_grid", action="store_true", default=None, help="Reproject DEM and mask rasters to the SAR grid before tiling.")
    prep_p.add_argument("--no-align-to-reference-grid", dest="align_to_reference_grid", action="store_false")
    prep_p.add_argument("--clip-dem", dest="clip_dem", action="store_true", default=None)
    prep_p.add_argument("--no-clip-dem", dest="clip_dem", action="store_false")
    prep_p.add_argument("--morphology", dest="morphology", action="store_true", default=None)
    prep_p.add_argument("--no-morphology", dest="morphology", action="store_false")
    prep_p.add_argument("--tiling", dest="tiling", action="store_true", default=None)
    prep_p.add_argument("--no-tiling", dest="tiling", action="store_false")
    prep_p.add_argument("--make-context", dest="make_context", action="store_true", default=None)
    prep_p.add_argument("--no-make-context", dest="make_context", action="store_false")

    norm_p = sub.add_parser("fit-normalization", help="Fit train-only robust normalization statistics for processed tiles.")
    norm_p.add_argument("--processed-data-dir", required=True, type=Path, help="Processed tiled dataset root.")
    norm_p.add_argument("--output-file", required=True, type=Path, help="Output normalization_stats.json path.")
    norm_p.add_argument("--split", choices=["train", "val", "test"], default="train")
    norm_p.add_argument(
        "--input-modalities",
        nargs="+",
        default=["vv", "vh"],
        help="Ordered modalities to fit, including derived channels such as vv_vh_log_ratio dem_slope dem_tpi.",
    )
    norm_p.add_argument(
        "--preserve-channel-stats-from",
        type=Path,
        help="Optional existing normalization JSON. Matching channels are copied exactly into the new file, preserving baseline preprocessing during channel expansion.",
    )
    norm_p.add_argument("--q-min", type=float, default=1.0, help="Lower clipping percentile fitted on the selected split.")
    norm_p.add_argument("--q-max", type=float, default=99.0, help="Upper clipping percentile fitted on the selected split.")
    norm_p.add_argument("--max-pixels-per-file", type=int, default=4096, help="Random finite pixels sampled per tile for robust fitting.")
    norm_p.add_argument("--seed", type=int, default=1337)

    derive_p = sub.add_parser("derive-features", help="Build reproducible VV/VH log-ratio, DEM slope, and local relative-elevation tiles.")
    derive_p.add_argument("--processed-data-dir", required=True, type=Path, help="Processed tiled dataset root containing split/sar and split/dem folders.")
    derive_p.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    derive_p.add_argument("--log-ratio-eps", type=float, default=1e-6, help="Positive floor used before the VV/VH logarithm.")
    derive_p.add_argument("--tpi-radius-pixels", type=int, default=15, help="Radius of the local-mean window used for DEM topographic position.")
    derive_p.add_argument("--overwrite", action="store_true", help="Rebuild existing derived GeoTIFFs instead of skipping them.")

    feature_audit_p = sub.add_parser("audit-feature-separability", help="Compare original and derived channels on train-to-validation sampled pixels before deep training.")
    feature_audit_p.add_argument("--processed-data-dir", required=True, type=Path)
    feature_audit_p.add_argument("--output-dir", required=True, type=Path)
    feature_audit_p.add_argument("--fit-split", choices=["train", "val", "test"], default="train")
    feature_audit_p.add_argument("--eval-split", choices=["train", "val", "test"], default="val")
    feature_audit_p.add_argument("--base-modalities", nargs="+", default=["vv", "vh", "dem"])
    feature_audit_p.add_argument("--extended-modalities", nargs="+", default=["vv", "vh", "dem", "vv_vh_log_ratio", "dem_slope", "dem_tpi"])
    feature_audit_p.add_argument("--max-pixels-per-class-per-tile", type=int, default=128)
    feature_audit_p.add_argument("--max-total-pixels-per-split", type=int, default=250000)
    feature_audit_p.add_argument("--seed", type=int, default=42)

    domain_shift_p = sub.add_parser(
        "audit-domain-shift",
        help="Compare a target EMSR event with the training domain across input distributions, label geometry, and event similarity.",
    )
    domain_shift_p.add_argument("--config", type=Path, help="Optional training config used to resolve processed data, modalities, and normalization metadata.")
    domain_shift_p.add_argument("--processed-data-dir", type=Path, help="Processed tiled dataset root. Overrides the config value.")
    domain_shift_p.add_argument("--output-dir", required=True, type=Path)
    domain_shift_p.add_argument("--reference-split", choices=["train", "val", "test"], default="train")
    domain_shift_p.add_argument("--target-split", choices=["train", "val", "test"], default="val")
    domain_shift_p.add_argument("--target-events", nargs="+", required=True, help="Target EMSR event IDs, e.g. EMSR342.")
    domain_shift_p.add_argument("--reference-events", nargs="+", help="Optional reference EMSR event subset.")
    domain_shift_p.add_argument("--exclude-reference-events", nargs="+", help="Optional reference EMSR events to exclude.")
    domain_shift_p.add_argument("--input-modalities", nargs="+", help="Ordered channels. Defaults to the training config or vv vh dem.")
    domain_shift_p.add_argument("--normalization-stats-path", type=Path, help="Optional train-fitted normalization JSON for clipping-rate diagnostics.")
    domain_shift_p.add_argument("--normalization-mode", help="Normalization mode associated with the checkpoint, e.g. robust_percentile.")
    domain_shift_p.add_argument("--max-reference-tiles", type=int, default=0, help="Optional event-stratified cap; 0 audits every reference tile.")
    domain_shift_p.add_argument("--max-target-tiles", type=int, default=0, help="Optional cap; 0 audits every target tile.")
    domain_shift_p.add_argument("--max-pixels-per-tile", type=int, default=256, help="Natural-distribution pixels sampled per tile and modality.")
    domain_shift_p.add_argument("--max-pixels-per-class-per-tile", type=int, default=128, help="Flood/background pixels sampled separately per tile and modality.")
    domain_shift_p.add_argument("--max-total-pixels-per-domain", type=int, default=250000, help="Maximum sampled pixels per modality/stratum/domain.")
    domain_shift_p.add_argument("--domain-classifier-reference-ratio", type=float, default=4.0, help="Reference tiles retained per target tile for the diagnostic classifier.")
    domain_shift_p.add_argument("--seed", type=int, default=42)
    domain_shift_p.add_argument("--write-plots", dest="write_plots", action="store_true", default=True)
    domain_shift_p.add_argument("--no-write-plots", dest="write_plots", action="store_false")

    domain_failure_p = sub.add_parser(
        "audit-domain-failure-link",
        help="Link target-event model failures to domain features and quantify training-domain analogue coverage.",
    )
    domain_failure_p.add_argument("--tile-features-csv", required=True, type=Path, help="tile_features.csv from audit-domain-shift.")
    domain_failure_p.add_argument("--tile-error-metrics-csv", required=True, type=Path, help="tile_error_metrics.csv from error-audit or ensemble-error-audit.")
    domain_failure_p.add_argument("--output-dir", required=True, type=Path)
    domain_failure_p.add_argument("--target-events", nargs="+", help="Optional target EMSR event IDs, e.g. EMSR342.")
    domain_failure_p.add_argument("--max-recall", type=float, default=0.25, help="Non-empty target tiles below this recall are failures.")
    domain_failure_p.add_argument("--neighbours", type=int, default=5, help="Training analogues exported per failing target tile.")
    domain_failure_p.add_argument("--seed", type=int, default=42)
    domain_failure_p.add_argument("--write-plots", dest="write_plots", action="store_true", default=True)
    domain_failure_p.add_argument("--no-write-plots", dest="write_plots", action="store_false")

    modality_ablation_p = sub.add_parser(
        "audit-modality-ablation",
        help="Measure checkpoint dependence on VV, VH, and DEM by zeroing normalized channels at inference time.",
    )
    modality_ablation_p.add_argument("--config", required=True, type=Path, help="Training YAML or saved run config.yaml.")
    modality_ablation_p.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", required=True, type=Path)
    modality_ablation_p.add_argument("--processed-data-dir", required=True, type=Path, help="Processed tiled dataset root.")
    modality_ablation_p.add_argument("--output-dir", required=True, type=Path)
    modality_ablation_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    modality_ablation_p.add_argument("--target-events", nargs="+", help="Report a separate target-event result, e.g. EMSR342.")
    modality_ablation_p.add_argument("--include-events", nargs="+", help="Optionally restrict the evaluated split to these events.")
    modality_ablation_p.add_argument("--exclude-events", nargs="+", help="Exclude these events from the evaluated split.")
    modality_ablation_p.add_argument("--ablations", nargs="+", default=["none", "dem", "vv", "vh"], help="Normalized channels to zero: none dem vv vh or combinations such as vv+vh.")
    modality_ablation_p.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    modality_ablation_p.add_argument("--threshold-metric", choices=["f1", "iou", "mcc", "precision", "recall"], default="f1")
    modality_ablation_p.add_argument("--operating-threshold", type=float, default=0.50, help="Shared fixed threshold reported alongside independently optimized sweeps.")
    modality_ablation_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    modality_ablation_p.add_argument("--batch-size", type=int)
    modality_ablation_p.add_argument("--num-workers", type=int)
    modality_ablation_p.add_argument("--input-modalities", nargs="+", help="Checkpoint channel order, e.g. vv vh dem.")
    _add_pretrained_override(modality_ablation_p)
    modality_ablation_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    modality_ablation_p.add_argument("--no-amp", dest="amp", action="store_false")
    modality_ablation_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    modality_ablation_p.add_argument("--gpu", dest="cpu", action="store_false")
    modality_ablation_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    modality_ablation_p.add_argument("--normalization-mode", help="Normalization mode used by the checkpoint.")
    modality_ablation_p.add_argument("--max-samples", type=int, help="Optional smoke-test cap; omit for a full audit.")


    water_prior_p = sub.add_parser(
        "audit-water-prior",
        help="Evaluate JRC long-term surface-water occurrence as a hard exclusion or soft probability penalty.",
    )
    water_prior_p.add_argument("--config", required=True, type=Path, help="Training YAML or saved run config.yaml.")
    water_prior_p.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", required=True, type=Path)
    water_prior_p.add_argument("--processed-data-dir", required=True, type=Path, help="Processed tiled dataset root.")
    water_prior_p.add_argument("--output-dir", required=True, type=Path, help="Directory for audit CSV and JSON outputs.")
    water_prior_p.add_argument("--prior-cache-dir", required=True, type=Path, help="Persistent cache for aligned JRC occurrence GeoTIFFs.")
    water_prior_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    water_prior_p.add_argument("--include-events", nargs="+", help="Only audit these EMSR event IDs, e.g. EMSR342.")
    water_prior_p.add_argument("--exclude-events", nargs="+", help="Exclude these EMSR event IDs.")
    water_prior_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    water_prior_p.add_argument("--batch-size", type=int)
    water_prior_p.add_argument("--num-workers", type=int)
    water_prior_p.add_argument("--input-modalities", nargs="+", help="Ordered model inputs, e.g. vv vh dem.")
    _add_pretrained_override(water_prior_p)
    water_prior_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    water_prior_p.add_argument("--no-amp", dest="amp", action="store_false")
    water_prior_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    water_prior_p.add_argument("--gpu", dest="cpu", action="store_false")
    water_prior_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    water_prior_p.add_argument("--normalization-mode", help="Normalization mode used by the retained checkpoint.")
    water_prior_p.add_argument("--model-thresholds", nargs="+", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    water_prior_p.add_argument("--occurrence-thresholds", nargs="+", type=int, default=[75, 90, 95, 99], help="JRC occurrence percentages treated as long-term water.")
    water_prior_p.add_argument("--penalty-strengths", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.00], help="Soft probability-penalty strengths in [0,1].")
    water_prior_p.add_argument("--min-component-areas", nargs="+", type=int, default=[96], help="Connected-component minimum areas included in the sweep.")
    water_prior_p.add_argument("--reference-threshold", type=float, default=0.50)
    water_prior_p.add_argument("--reference-min-component-area", type=int, default=96)
    water_prior_p.add_argument("--max-recall-drop", type=float, default=0.02, help="Maximum recall loss allowed for the guarded best setting.")
    water_prior_p.add_argument("--hard-exclusion", dest="include_hard_exclusion", action="store_true", default=True)
    water_prior_p.add_argument("--no-hard-exclusion", dest="include_hard_exclusion", action="store_false")
    water_prior_p.add_argument("--soft-penalty", dest="include_soft_penalty", action="store_true", default=True)
    water_prior_p.add_argument("--no-soft-penalty", dest="include_soft_penalty", action="store_false")
    water_prior_p.add_argument("--offline-prior-cache", action="store_true", help="Use only existing aligned prior files and make no catalogue requests.")
    water_prior_p.add_argument("--allow-incomplete-prior", action="store_true", help="Allow less than 99 percent JRC coverage; uncovered pixels are left unpenalised.")
    water_prior_p.add_argument("--max-samples", type=int, help="Optional smoke-test limit on audited tiles.")

    hysteresis_p = sub.add_parser(
        "audit-hysteresis-postprocess",
        help="Evaluate seeded hysteresis region growing against fixed probability thresholds.",
    )
    hysteresis_p.add_argument("--config", required=True, type=Path, help="Training YAML or saved run config.yaml.")
    hysteresis_p.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", required=True, type=Path)
    hysteresis_p.add_argument("--processed-data-dir", required=True, type=Path, help="Processed tiled dataset root.")
    hysteresis_p.add_argument("--output-dir", required=True, type=Path, help="Directory for audit CSV and JSON outputs.")
    hysteresis_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    hysteresis_p.add_argument("--include-events", nargs="+", help="Only audit these EMSR event IDs, e.g. EMSR342.")
    hysteresis_p.add_argument("--exclude-events", nargs="+", help="Exclude these EMSR event IDs.")
    hysteresis_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    hysteresis_p.add_argument("--batch-size", type=int)
    hysteresis_p.add_argument("--num-workers", type=int)
    hysteresis_p.add_argument("--input-modalities", nargs="+", help="Ordered model inputs, e.g. vv vh dem.")
    _add_pretrained_override(hysteresis_p)
    hysteresis_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    hysteresis_p.add_argument("--no-amp", dest="amp", action="store_false")
    hysteresis_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    hysteresis_p.add_argument("--gpu", dest="cpu", action="store_false")
    hysteresis_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    hysteresis_p.add_argument("--normalization-mode", help="Normalization mode used by the retained checkpoint.")
    hysteresis_p.add_argument("--fixed-thresholds", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    hysteresis_p.add_argument("--low-thresholds", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    hysteresis_p.add_argument("--high-thresholds", nargs="+", type=float, default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    hysteresis_p.add_argument("--min-seed-pixels", nargs="+", type=int, default=[1, 16, 64])
    hysteresis_p.add_argument("--min-component-areas", nargs="+", type=int, default=[96])
    hysteresis_p.add_argument("--reference-threshold", type=float, default=0.50)
    hysteresis_p.add_argument("--reference-min-component-area", type=int, default=96)
    hysteresis_p.add_argument("--max-recall-drop", type=float, default=0.02)
    hysteresis_p.add_argument("--max-empty-fp-rate-increase", type=float, default=0.0)
    hysteresis_p.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    hysteresis_p.add_argument("--max-samples", type=int, help="Optional smoke-test limit on audited tiles.")

    derived_config_p = sub.add_parser(
        "prepare-derived-experiment",
        help="Create a controlled six-channel warm-start configuration from a retained VV/VH/DEM baseline.",
    )
    derived_config_p.add_argument("--base-config", required=True, type=Path)
    derived_config_p.add_argument("--baseline-checkpoint", required=True, type=Path)
    derived_config_p.add_argument("--normalization-stats-path", required=True, type=Path)
    derived_config_p.add_argument("--output-config", required=True, type=Path)
    derived_config_p.add_argument("--run-id", required=True)
    derived_config_p.add_argument("--artifacts-dir", required=True, type=Path)
    derived_config_p.add_argument("--batch-size", type=int, default=4)
    derived_config_p.add_argument("--epochs", type=int, default=20)
    derived_config_p.add_argument("--patience", type=int, default=6)
    derived_config_p.add_argument("--encoder-lr", type=float, default=1e-5)
    derived_config_p.add_argument("--decoder-lr", type=float, default=1e-5)
    derived_config_p.add_argument("--max-skipped-batch-fraction", type=float, default=0.02)

    audit_data_p = sub.add_parser("audit-training-data", help="Audit train/val/test split leakage and flood-target composition before training.")
    audit_data_p.add_argument("--processed-data-dir", required=True, type=Path, help="Processed tiled dataset root containing train/val/test folders.")
    audit_data_p.add_argument("--output-dir", required=True, type=Path, help="Directory for JSON and CSV audit outputs.")
    audit_data_p.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"], help="Splits to compare.")
    audit_data_p.add_argument("--fail-on-leakage", action="store_true", help="Exit with an error when exact-tile or source-scene overlap is found.")

    exposure_p = sub.add_parser("audit-training-exposure", help="Simulate the configured training sampler and audit batches, tile exposure, and hard-negative proxy coverage.")
    exposure_p.add_argument("--config", type=Path, required=True, help="Training YAML or saved run config.")
    exposure_p.add_argument("--processed-data-dir", type=str, default=None, help="Override the processed dataset root from the config.")
    exposure_p.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV and JSON audit outputs.")
    exposure_p.add_argument("--epochs", type=int, default=10, help="Number of sampler epochs to simulate.")
    exposure_p.add_argument("--seed", type=int, default=42, help="Deterministic audit seed.")
    exposure_p.add_argument("--negative-max-ratio", type=float, default=0.001, help="Maximum foreground ratio for hard-negative proxy analysis.")
    exposure_p.add_argument("--negative-clusters", type=int, default=8, help="Number of unsupervised SAR/DEM negative groups.")
    exposure_p.add_argument("--sampler-profiles", nargs="+", choices=["configured", "tempered", "shuffle"], default=["configured", "tempered", "shuffle"], help="Sampler profiles to compare. Tempered uses square-root event balancing with a 5x median tile-weight cap.")
    stats_p = sub.add_parser("stats", help="Compute dataset statistics.")
    stats_p.add_argument("--config", type=Path, required=True, help="Stats YAML.")
    stats_p.add_argument("--processed-data-dir", "--data-root", dest="processed_data_dir", type=Path)
    stats_p.add_argument("--subset", choices=["train", "val", "test"])

    pseudo_p = sub.add_parser("pseudolabel", help="Generate threshold-based pseudo-labels.")
    pseudo_p.add_argument("--config", type=Path, required=True, help="Preparation YAML.")

    test_p = sub.add_parser("test", help="Evaluate a trained model/checkpoint.")
    test_p.add_argument("--config", type=Path, required=True, help="Test YAML.")
    test_p.add_argument("--checkpoint-path")
    test_p.add_argument("--data-root")
    test_p.add_argument("--store-predictions", dest="store_predictions", action="store_true", default=None)
    test_p.add_argument("--no-store-predictions", dest="store_predictions", action="store_false")

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint on a processed split with an optional threshold sweep.")
    eval_p.add_argument("--config", type=Path, required=True, help="Training YAML or saved run config.yaml.")
    eval_p.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", type=Path, required=True)
    eval_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", help="Processed tiled dataset root.")
    eval_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    eval_p.add_argument("--include-events", nargs="+", help="Only evaluate these EMSR event IDs, e.g. EMSR342")
    eval_p.add_argument("--exclude-events", nargs="+", help="Exclude these EMSR event IDs from evaluation")
    eval_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    eval_p.add_argument("--batch-size", type=int)
    eval_p.add_argument("--num-workers", type=int)
    eval_p.add_argument("--input-modalities", nargs="+", help="Example: --input-modalities vv vh dem")
    _add_pretrained_override(eval_p)
    eval_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    eval_p.add_argument("--no-amp", dest="amp", action="store_false")
    eval_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    eval_p.add_argument("--gpu", dest="cpu", action="store_false")
    eval_p.add_argument("--thresholds", nargs="+", type=float, help="Probability thresholds for evaluation.")
    eval_p.add_argument("--threshold-metric", choices=["f1", "iou", "mcc", "precision", "recall"], default="f1")
    eval_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    eval_p.add_argument("--normalization-mode", help="Normalization mode used for evaluation.")
    eval_p.add_argument("--metric-mode", choices=["global", "batch_average", "both"], default=None, help="Report global threshold-sweep metrics, batch-averaged metrics, or both.")
    eval_p.add_argument("--inference-mode", choices=["direct", "sliding_window"], default="direct", help="Use direct batched inference or sliding-window full-scene inference.")
    eval_p.add_argument("--window-size", type=int, default=256, help="Sliding-window inference tile size.")
    eval_p.add_argument("--window-overlap", type=int, default=64, help="Sliding-window overlap in pixels.")
    eval_p.add_argument("--window-batch-size", type=int, default=4, help="Number of windows forwarded at once during sliding-window inference.")

    ens_p = sub.add_parser("ensemble-evaluate", help="Evaluate an ensemble of checkpoints by averaging per-pixel probabilities or logits.")
    ens_p.add_argument("--configs", "--config", dest="configs", nargs="+", type=Path, required=True, help="Training YAML/run config paths, one per ensemble member.")
    ens_p.add_argument("--checkpoints", "--checkpoint", dest="checkpoints", nargs="+", type=Path, required=True, help="Checkpoint paths, one per ensemble member and in the same order as --configs.")
    ens_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", help="Processed tiled dataset root.")
    ens_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    ens_p.add_argument("--include-events", nargs="+", help="Only evaluate these EMSR event IDs, e.g. EMSR342")
    ens_p.add_argument("--exclude-events", nargs="+", help="Exclude these EMSR event IDs from evaluation")
    ens_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    ens_p.add_argument("--batch-size", type=int)
    ens_p.add_argument("--num-workers", type=int)
    ens_p.add_argument("--input-modalities", nargs="+", help="Example: --input-modalities vv vh dem")
    _add_pretrained_override(ens_p)
    ens_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    ens_p.add_argument("--no-amp", dest="amp", action="store_false")
    ens_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    ens_p.add_argument("--gpu", dest="cpu", action="store_false")
    ens_p.add_argument("--thresholds", nargs="+", type=float, help="Probability thresholds for ensemble evaluation.")
    ens_p.add_argument("--threshold-metric", choices=["f1", "iou", "mcc", "precision", "recall"], default="f1")
    ens_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    ens_p.add_argument("--normalization-mode", help="Normalization mode used for evaluation.")
    ens_p.add_argument("--metric-mode", choices=["global", "batch_average", "both"], default=None, help="Report global threshold-sweep metrics, batch-averaged metrics, or both.")
    ens_p.add_argument("--ensemble-method", choices=["mean_prob", "mean_logit"], default="mean_prob", help="How to combine model outputs. mean_prob is recommended for different architectures.")
    ens_p.add_argument("--inference-mode", choices=["direct", "sliding_window"], default="direct", help="Use direct batched inference or sliding-window full-scene inference.")
    ens_p.add_argument("--window-size", type=int, default=256, help="Sliding-window inference tile size.")
    ens_p.add_argument("--window-overlap", type=int, default=64, help="Sliding-window overlap in pixels.")
    ens_p.add_argument("--window-batch-size", type=int, default=4, help="Number of windows forwarded at once during sliding-window inference.")

    error_p = sub.add_parser("error-audit", help="Audit checkpoint errors with overlays, event metrics, foreground-bin metrics, and component filtering sweeps.")
    error_p.add_argument("--config", type=Path, required=True, help="Training YAML or saved run config.yaml.")
    error_p.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", type=Path, required=True)
    error_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", required=True, help="Processed tiled dataset root.")
    error_p.add_argument("--output-dir", required=True, type=Path, help="Directory where audit CSVs and overlays will be written.")
    error_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    error_p.add_argument("--include-events", nargs="+", help="Only audit these EMSR event IDs, e.g. EMSR342")
    error_p.add_argument("--exclude-events", nargs="+", help="Exclude these EMSR event IDs from the audit")
    error_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    error_p.add_argument("--batch-size", type=int)
    error_p.add_argument("--num-workers", type=int)
    error_p.add_argument("--input-modalities", nargs="+", help="Example: --input-modalities vv vh dem")
    _add_pretrained_override(error_p)
    error_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    error_p.add_argument("--no-amp", dest="amp", action="store_false")
    error_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    error_p.add_argument("--gpu", dest="cpu", action="store_false")
    error_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    error_p.add_argument("--normalization-mode", help="Normalization mode used for audit.")
    error_p.add_argument("--threshold", type=float, default=0.50, help="Operating threshold used for tile-level error categorisation and overlays.")
    error_p.add_argument("--thresholds", nargs="+", type=float, help="Thresholds for global and component-filter sweeps.")
    error_p.add_argument("--threshold-metric", choices=["f1", "iou", "mcc", "precision", "recall"], default="f1")
    error_p.add_argument("--min-component-area", type=int, default=0, help="Remove predicted connected components smaller than this area for tile-level audit.")
    error_p.add_argument("--sweep-component-areas", nargs="+", type=int, default=[0, 8, 16, 32, 64], help="Connected-component minimum areas to test in threshold_component_sweep.csv.")
    error_p.add_argument("--max-overlays-per-category", type=int, default=12, help="Number of PNG overlays to save per error category.")
    error_p.add_argument("--max-samples", type=int, help="Optional cap for small audit runs; normally omit for full audits.")

    mine_pos_p = sub.add_parser("mine-hard-positives", help="Mine false-negative flood crop regions from a labelled split for audit-guided fine-tuning.")
    mine_pos_p.add_argument("--config", type=Path, required=True)
    mine_pos_p.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", type=Path, required=True)
    mine_pos_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", required=True)
    mine_pos_p.add_argument("--output-dir", required=True, type=Path)
    mine_pos_p.add_argument("--split", choices=["train", "val", "test"], default="train")
    mine_pos_p.add_argument("--image-size", type=int, choices=[128,256,512])
    mine_pos_p.add_argument("--batch-size", type=int)
    mine_pos_p.add_argument("--num-workers", type=int)
    mine_pos_p.add_argument("--input-modalities", nargs="+")
    _add_pretrained_override(mine_pos_p)
    mine_pos_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    mine_pos_p.add_argument("--no-amp", dest="amp", action="store_false")
    mine_pos_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    mine_pos_p.add_argument("--gpu", dest="cpu", action="store_false")
    mine_pos_p.add_argument("--normalization-stats-path")
    mine_pos_p.add_argument("--normalization-mode")
    mine_pos_p.add_argument("--threshold", type=float, default=0.50)
    mine_pos_p.add_argument("--crop-sizes", nargs="+", type=int, default=[320,384])
    mine_pos_p.add_argument("--min-component-area", type=int, default=32)
    mine_pos_p.add_argument("--min-fn-pixels", type=int, default=64)
    mine_pos_p.add_argument("--min-label-fg-ratio", type=float, default=0.002)
    mine_pos_p.add_argument("--min-valid-ratio", type=float, default=0.50)
    mine_pos_p.add_argument("--max-regions-per-tile", type=int, default=3)
    mine_pos_p.add_argument("--nms-iou", type=float, default=0.30)
    mine_pos_p.add_argument("--max-samples", type=int)

    mine_p = sub.add_parser("mine-hard-negatives", help="Mine high-confidence false-positive crop regions from a labelled split for audit-guided fine-tuning.")
    mine_p.add_argument("--config", type=Path, required=True, help="Training YAML or saved run config.yaml.")
    mine_p.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", type=Path, required=True)
    mine_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", required=True, help="Processed tiled dataset root.")
    mine_p.add_argument("--output-dir", required=True, type=Path, help="Directory where hard_negative_regions.csv and summary.json will be written.")
    mine_p.add_argument("--split", choices=["train", "val", "test"], default="train", help="Use train for model-development mining; validation mining is diagnostic only.")
    mine_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    mine_p.add_argument("--batch-size", type=int)
    mine_p.add_argument("--num-workers", type=int)
    mine_p.add_argument("--input-modalities", nargs="+", help="Example: --input-modalities vv vh dem")
    _add_pretrained_override(mine_p)
    mine_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    mine_p.add_argument("--no-amp", dest="amp", action="store_false")
    mine_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    mine_p.add_argument("--gpu", dest="cpu", action="store_false")
    mine_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    mine_p.add_argument("--normalization-mode", help="Normalization mode used for mining.")
    mine_p.add_argument("--threshold", type=float, default=0.60, help="Probability threshold defining a high-confidence false positive.")
    mine_p.add_argument("--crop-sizes", nargs="+", type=int, default=[256, 320, 384], help="Candidate square crop sizes for mined regions.")
    mine_p.add_argument("--min-component-area", type=int, default=64, help="Ignore false-positive components smaller than this many pixels.")
    mine_p.add_argument("--min-fp-pixels", type=int, default=128, help="Minimum false-positive pixels required inside a mined crop.")
    mine_p.add_argument("--max-label-fg-ratio", type=float, default=0.001, help="Maximum labelled flood ratio allowed inside a hard-negative crop.")
    mine_p.add_argument("--min-valid-ratio", type=float, default=0.50, help="Minimum non-ignore fraction inside a crop.")
    mine_p.add_argument("--max-regions-per-tile", type=int, default=3)
    mine_p.add_argument("--nms-iou", type=float, default=0.30, help="Suppress overlapping mined crops above this IoU.")
    mine_p.add_argument("--max-samples", type=int, help="Optional cap for smoke tests; omit for a full train-split mining pass.")


    ens_error_p = sub.add_parser("ensemble-error-audit", help="Audit an ensemble with overlays, event metrics, foreground-bin metrics, and component filtering sweeps.")
    ens_error_p.add_argument("--configs", "--config", dest="configs", nargs="+", type=Path, required=True, help="Training YAML/run config paths, one per ensemble member.")
    ens_error_p.add_argument("--checkpoints", "--checkpoint", dest="checkpoints", nargs="+", type=Path, required=True, help="Checkpoint paths, one per ensemble member.")
    ens_error_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", required=True, help="Processed dataset root.")
    ens_error_p.add_argument("--output-dir", required=True, type=Path, help="Directory where audit CSVs and overlays will be written.")
    ens_error_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    ens_error_p.add_argument("--include-events", nargs="+", help="Only audit these EMSR event IDs.")
    ens_error_p.add_argument("--exclude-events", nargs="+", help="Exclude these EMSR event IDs.")
    ens_error_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    ens_error_p.add_argument("--batch-size", type=int)
    ens_error_p.add_argument("--num-workers", type=int)
    ens_error_p.add_argument("--input-modalities", nargs="+", help="Example: --input-modalities vv vh dem")
    _add_pretrained_override(ens_error_p)
    ens_error_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    ens_error_p.add_argument("--no-amp", dest="amp", action="store_false")
    ens_error_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    ens_error_p.add_argument("--gpu", dest="cpu", action="store_false")
    ens_error_p.add_argument("--normalization-stats-path", help="Path to train-fitted normalization_stats.json.")
    ens_error_p.add_argument("--normalization-mode", help="Normalization mode used for audit.")
    ens_error_p.add_argument("--ensemble-method", choices=["mean_prob", "mean_logit"], default="mean_logit")
    ens_error_p.add_argument("--inference-mode", choices=["direct", "sliding_window"], default="direct", help="Use direct batched inference or sliding-window full-scene inference.")
    ens_error_p.add_argument("--window-size", type=int, default=256, help="Sliding-window inference tile size.")
    ens_error_p.add_argument("--window-overlap", type=int, default=64, help="Sliding-window overlap in pixels.")
    ens_error_p.add_argument("--window-batch-size", type=int, default=4, help="Number of windows forwarded at once during sliding-window inference.")
    ens_error_p.add_argument("--threshold", type=float, default=0.50, help="Operating threshold used for tile-level error categorisation and overlays.")
    ens_error_p.add_argument("--thresholds", nargs="+", type=float, help="Thresholds for global and component-filter sweeps.")
    ens_error_p.add_argument("--threshold-metric", choices=["f1", "iou", "mcc", "precision", "recall"], default="f1")
    ens_error_p.add_argument("--min-component-area", type=int, default=0)
    ens_error_p.add_argument("--sweep-component-areas", nargs="+", type=int, default=[0, 8, 16, 32, 64])
    ens_error_p.add_argument("--max-overlays-per-category", type=int, default=12)
    ens_error_p.add_argument("--max-samples", type=int)


    cmp_p = sub.add_parser("compare-models", help="Evaluate and compare any number of single models and ensembles, writing CSV/JSON summaries and plots.")
    cmp_p.add_argument("--output-dir", required=True, type=Path, help="Directory for comparison CSV, JSON and PNG plots.")
    cmp_p.add_argument("--model", nargs=3, action="append", metavar=("NAME", "CONFIG", "CHECKPOINT"), help="Single model specification. Repeat as needed.")
    cmp_p.add_argument("--ensemble", nargs="+", action="append", metavar="ITEM", help="Ensemble specification: NAME METHOD CONFIG:CHECKPOINT CONFIG:CHECKPOINT ... . Repeat as needed.")
    cmp_p.add_argument("--processed-data-dir", "--data-path", dest="processed_data_dir", help="Processed tiled dataset root.")
    cmp_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    cmp_p.add_argument("--include-events", nargs="+", help="Only evaluate these EMSR event IDs.")
    cmp_p.add_argument("--exclude-events", nargs="+", help="Exclude these EMSR event IDs.")
    cmp_p.add_argument("--image-size", type=int, choices=[128, 256, 512])
    cmp_p.add_argument("--batch-size", type=int)
    cmp_p.add_argument("--num-workers", type=int)
    cmp_p.add_argument("--input-modalities", nargs="+", help="Example: --input-modalities vv vh dem")
    _add_pretrained_override(cmp_p)
    cmp_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    cmp_p.add_argument("--no-amp", dest="amp", action="store_false")
    cmp_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    cmp_p.add_argument("--gpu", dest="cpu", action="store_false")
    cmp_p.add_argument("--thresholds", nargs="+", type=float)
    cmp_p.add_argument("--threshold-metric", choices=["f1", "iou", "mcc", "precision", "recall"], default="f1")
    cmp_p.add_argument("--normalization-stats-path")
    cmp_p.add_argument("--normalization-mode")
    cmp_p.add_argument("--metric-mode", choices=["global", "batch_average", "both"], default="global")
    cmp_p.add_argument("--inference-mode", choices=["direct", "sliding_window"], default="direct")
    cmp_p.add_argument("--window-size", type=int, default=512)
    cmp_p.add_argument("--window-overlap", type=int, default=128)
    cmp_p.add_argument("--window-batch-size", type=int, default=1)
    cmp_p.add_argument("--plot-metrics", nargs="+", default=["f1", "iou", "mcc"], choices=["f1", "iou", "mcc", "precision", "recall"], help="Best-metric bar plots to write.")
    cmp_p.add_argument("--confusion-matrix-plots", dest="confusion_matrix_plots", action="store_true", default=True, help="Write one best-threshold confusion-matrix PNG per model/ensemble.")
    cmp_p.add_argument("--no-confusion-matrix-plots", dest="confusion_matrix_plots", action="store_false", help="Do not write confusion-matrix PNGs.")


    dep_p = sub.add_parser("export-deployment", help="Write a deployment manifest for a single model or ensemble operating point.")
    dep_p.add_argument("--output-file", required=True, type=Path, help="Manifest path, .yaml/.yml or .json. By default the parent directory becomes a portable deployment bundle.")
    dep_p.add_argument("--model-name", default="flood_model")
    dep_bundle = dep_p.add_mutually_exclusive_group()
    dep_bundle.add_argument("--portable", dest="copy_assets", action="store_true", default=True, help="Copy configs, checkpoints and normalization assets beside the manifest and use relative paths (default).")
    dep_bundle.add_argument("--reference-only", dest="copy_assets", action="store_false", help="Write external file references without copying assets. This manifest is environment-dependent.")
    dep_p.add_argument("--assets-directory", default="assets", help="Relative asset directory inside a portable deployment bundle (default: assets).")
    dep_p.add_argument("--configs", "--config", nargs="+", type=Path, required=True)
    dep_p.add_argument("--checkpoints", "--checkpoint", nargs="+", type=Path, required=True)
    dep_p.add_argument("--ensemble-method", choices=["mean_prob", "mean_logit"], default="mean_logit")
    dep_p.add_argument("--threshold", type=float, required=True)
    dep_p.add_argument("--min-component-area", type=int, default=0)
    dep_p.add_argument("--input-modalities", nargs="+", default=["vv", "vh", "dem"])
    dep_p.add_argument("--normalization-stats-path")
    dep_p.add_argument("--normalization-mode")
    dep_p.add_argument("--inference-mode", choices=["direct", "sliding_window"], default="sliding_window")
    dep_p.add_argument("--window-size", type=int, default=512)
    dep_p.add_argument("--window-overlap", type=int, default=128)
    dep_p.add_argument("--window-batch-size", type=int, default=1)
    dep_p.add_argument("--window-blend", choices=["uniform", "cosine"], default="uniform", help="Overlap blending. cosine downweights window borders.")
    dep_p.add_argument("--notes")

    disc_p = sub.add_parser("discover-scene", help="Inventory an EMSR/scene folder and group deployable SAR VV/VH candidates.")
    disc_p.add_argument("--scene-dir", required=True, type=Path, help="Folder containing one or more analysis-ready SAR GeoTIFF files.")
    disc_p.add_argument("--output-file", type=Path, help="Optional CSV path for the discovered SAR inventory.")
    disc_p.add_argument("--scene-id", help="Optional scene/event identifier used when constructing candidate names.")
    disc_p.add_argument("--candidate-prefix", help="Optional prefix to prepend to discovered candidate IDs.")
    disc_p.add_argument("--candidate-name-template", help="Optional Python format template for candidate IDs. Available fields: {scene_id}, {date}, {stem}, {index}, {kind}.")

    pred_p = sub.add_parser("predict-scene", help="Run deployment inference on a direct SAR/DEM scene or a folder with multiple SAR acquisitions.")
    pred_p.add_argument("--manifest", required=True, type=Path, help="Deployment manifest from export-deployment.")
    pred_p.add_argument("--sar-path", type=Path, help="Input two-band SAR GeoTIFF. Expected band 1=VV and band 2=VH.")
    pred_p.add_argument("--scene-dir", type=Path, help="Folder containing one or more deployable SAR files. Use discover-scene to inspect it first.")
    pred_p.add_argument("--input-csv", type=Path, help="Explicit deployment CSV. Columns may include candidate_id,sar_path,vv_path,vh_path,dem_path,mask_path,date,mosaic_group.")
    pred_p.add_argument("--dem-path", type=Path, help="Input DEM GeoTIFF. Required when the manifest includes dem unless a DEM is discoverable in --scene-dir/--dem-dir.")
    pred_p.add_argument("--dem-dir", type=Path, help="Directory to search for a DEM if --dem-path is not supplied.")
    pred_p.add_argument("--mask-path", type=Path, help="Optional ground-truth mask GeoTIFF for labelled-scene evaluation.")
    pred_p.add_argument("--mask-dir", type=Path, help="Directory to search for ground-truth masks when deploying from a scene folder.")
    pred_p.add_argument("--evaluate", action="store_true", help="Calculate F1/IoU/precision/recall/MCC and confusion matrix when a mask is supplied.")
    pred_p.add_argument("--sar-selection", default="all", help="For --scene-dir: all, latest, earliest, or a candidate_id from discover-scene.")
    pred_p.add_argument("--sar-date", help="For --scene-dir: process only candidates matching YYYY-MM-DD or YYYYMMDD.")
    pred_p.add_argument("--output-dir", type=Path, help="Output directory for GeoTIFFs, previews, reports and metadata.")
    pred_p.add_argument("--output-prefix", help="Filename prefix for a direct scene or single selected candidate.")
    pred_p.add_argument("--scene-id", help="Optional scene/event identifier used when constructing candidate names.")
    pred_p.add_argument("--candidate-prefix", help="Optional prefix to prepend to discovered candidate IDs.")
    pred_p.add_argument("--candidate-name-template", help="Optional Python format template for candidate IDs. Available fields: {scene_id}, {date}, {stem}, {index}, {kind}.")
    pred_p.add_argument("--mosaic-mode", choices=["smart", "off", "plan", "auto", "force"], default="smart", help="How to handle possible SAR tile mosaics. smart is the default: mosaic safe prediction-only groups, and mosaic SAR+matching masks for labelled evaluation when safe. plan prints decisions without mosaicking; off keeps candidates separate; force relaxes name/date grouping rules but still checks raster compatibility.")
    pred_p.add_argument("--mosaic-compatible-sar-tiles", action="store_true", help="Alias for --mosaic-mode auto.")
    pred_p.add_argument("--mosaic-undated", action="store_true", help="Allow undated selected multiband SAR files to be grouped when compatible. Prefer --mosaic-mode auto/force for new workflows.")
    pred_p.add_argument("--output-mask", type=Path, help="Optional explicit output binary flood mask GeoTIFF for single-scene prediction.")
    pred_p.add_argument("--output-probability", type=Path, help="Optional explicit output float32 flood-probability GeoTIFF for single-scene prediction.")
    pred_p.add_argument("--output-mode", choices=["concise", "standard", "verbose"], default="standard", help="Console verbosity. Use concise for user-facing prediction results only; verbose for debugging.")
    pred_p.add_argument("--prediction-only", action="store_true", help="Only write prediction rasters and the final console table; skip previews, reports and explanations.")
    pred_p.add_argument("--write-probability", action="store_true", help="Write a float32 flood-probability GeoTIFF.")
    pred_p.add_argument("--write-previews", action="store_true", help="Write input preview PNGs for VV, VH, DEM and false-colour composites.")
    pred_p.add_argument("--write-overlay", action="store_true", help="Write colorized probability and binary-mask overlay PNGs.")
    pred_p.add_argument("--write-uncertainty", action="store_true", help="Write ensemble disagreement maps when using an ensemble manifest.")
    pred_p.add_argument("--write-html-report", action="store_true", help="Write an HTML report with input, prediction and explanation panels.")
    pred_p.add_argument("--display-inline", action="store_true", help="Display the visual report inline when running inside Colab/Jupyter.")
    pred_p.add_argument("--explain", action="store_true", help="Write colorized positive-evidence panels for the prediction.")
    pred_p.add_argument("--explain-per-modality", action="store_true", help="Run optional occlusion explanations for VV, VH and DEM. This is slower.")
    pred_p.add_argument("--write-window-diagnostics", action="store_true", help="Write sliding-window grid, overlap-count map, per-window statistics and seam diagnostics without rerunning inference.")
    pred_p.add_argument("--window-blend", choices=["uniform", "cosine"], help="Override manifest overlap blending for this run.")
    pred_p.add_argument("--cpu", dest="deployment_device", action="store_const", const="cpu", default=None)
    pred_p.add_argument("--gpu", dest="deployment_device", action="store_const", const="cuda", default=None)

    catalog_p = sub.add_parser("model-catalog", help="List registered model and pretrained-weight sources.")
    catalog_p.add_argument("--json", action="store_true", help="Print the catalog as JSON.")

    doctor_p = sub.add_parser(
        "doctor",
        help="Validate the installed NumPy/SciPy/Rasterio/PyTorch runtime before training or deployment.",
    )
    doctor_p.add_argument("--json-output", type=Path, help="Optional JSON health-report path.")
    doctor_p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error when any runtime probe fails.",
    )

    cv_p = sub.add_parser(
        "event-cv",
        help="Run leakage-controlled event-level cross-validation across architectures and modality sets.",
    )
    cv_p.add_argument("--config", type=Path, required=True, help="Base training YAML. Runs start from normal initialisation, never from a checkpoint.")
    cv_p.add_argument("--processed-data-dir", type=Path, required=True, help="Processed MMFlood dataset root.")
    cv_p.add_argument("--output-dir", type=Path, required=True, help="Root for fold plan, normalization stats, runs, and ranked summaries.")
    cv_p.add_argument("--candidates", nargs="+", help="SOURCE[:DECODER[:ENCODER[:LABEL]]] candidates, e.g. terramind_v1_tiny or imagenet:unet:resnet34.")
    cv_p.add_argument("--architectures", nargs="+", help="Legacy decoder:encoder[:label] candidates; mapped using --pretrained.")
    cv_p.add_argument("--modality-sets", nargs="+", default=["vv+vh", "vv+vh+dem"], help="Input sets such as vv+vh and vv+vh+dem.")
    cv_p.add_argument("--folds", type=int, default=5, help="Number of event-separated folds.")
    cv_p.add_argument("--fold-indices", nargs="+", type=int, help="Run only selected zero-based folds; omit to run all folds.")
    cv_p.add_argument("--source-split", choices=["train"], default="train", help="Processed split used to form internal event folds.")
    cv_p.add_argument("--seed", type=int, default=42)
    cv_p.add_argument("--epochs", type=int, help="Maximum epochs per model.")
    cv_p.add_argument("--patience", type=int, help="Early-stopping patience per model.")
    cv_p.add_argument("--batch-size", type=int)
    cv_p.add_argument("--num-workers", type=int)
    cv_p.add_argument("--encoder-lr", type=float, help="Encoder learning rate applied consistently to every selected fold.")
    cv_p.add_argument("--decoder-lr", type=float, help="Decoder learning rate applied consistently to every selected fold.")
    cv_p.add_argument("--weight-decay", type=float, help="Optimizer weight decay applied consistently to every selected fold.")
    cv_p.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], help="Optimizer shared by the compared candidates.")
    cv_p.add_argument("--scheduler", choices=["plateau", "exp", "cosine", "poly"], help="Scheduler shared by the compared candidates.")
    cv_p.add_argument("--loss", choices=["bce", "focal", "tversky", "bce_tversky", "focal_tversky_combo", "lovasz", "combo"], help="Loss shared by the compared candidates.")
    cv_p.add_argument("--loss-alpha", type=float)
    cv_p.add_argument("--loss-beta", type=float)
    cv_p.add_argument("--bce-weight", type=float)
    cv_p.add_argument("--tversky-weight", type=float)
    cv_p.add_argument("--augmentation-profile", choices=["none", "geometric", "sar_radiometric", "standard", "crop_aware", "deformation", "composite"], help="Training augmentation shared by ordinary candidates. Registered providers may enforce a safer provider-compatible profile.")
    cv_p.add_argument("--pretrained", dest="pretrained", action="store_true", default=None, help="Use pretrained encoder weights as the normal initialisation.")
    cv_p.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Initialise every model from random weights.")
    cv_p.add_argument("--amp", dest="amp", action="store_true", default=None)
    cv_p.add_argument("--no-amp", dest="amp", action="store_false")
    cv_p.add_argument("--cpu", dest="cpu", action="store_true", default=None)
    cv_p.add_argument("--gpu", dest="cpu", action="store_false")
    cv_p.add_argument("--thresholds", nargs="+", type=float, help="Thresholds used for event-macro checkpoint selection.")
    cv_p.add_argument("--q-min", type=float, default=1.0, help="Lower percentile for fold-specific normalization.")
    cv_p.add_argument("--q-max", type=float, default=99.0, help="Upper percentile for fold-specific normalization.")
    cv_p.add_argument("--max-pixels-per-file", type=int, default=4096, help="Pixel sample cap per tile for fold-specific normalization.")
    cv_p.add_argument("--plan-only", action="store_true", help="Write and display the fold/run plan without training.")
    cv_p.add_argument("--skip-completed", dest="skip_completed", action="store_true", default=True, help="Resume the matrix by skipping only successfully completed runs with a validated cv_result.json marker.")
    cv_p.add_argument("--rerun-completed", dest="skip_completed", action="store_false", help="Run completed matrix entries again.")

    cv_cal_p = sub.add_parser(
        "calibrate-event-cv",
        help="Calibrate one operating threshold from completed out-of-fold event-CV checkpoints.",
    )
    cv_cal_p.add_argument("--cv-dir", type=Path, required=True, help="Completed event-CV directory containing cv_results.csv and runs/.")
    cv_cal_p.add_argument("--processed-data-dir", type=Path, required=True, help="Processed MMFlood dataset root used by the CV runs.")
    cv_cal_p.add_argument("--output-dir", type=Path, required=True, help="Directory for OOF calibration CSV, JSON, and log outputs.")
    cv_cal_p.add_argument("--candidate", help="Candidate label or weights-source name from cv_results.csv.")
    cv_cal_p.add_argument("--architecture", help="Legacy architecture label from older cv_results.csv.")
    cv_cal_p.add_argument("--modalities", default="vv+vh", help="Modality label from cv_results.csv, for example vv+vh.")
    cv_cal_p.add_argument("--fold-indices", nargs="+", type=int, help="Optional zero-based fold subset; omit to use every completed matching fold.")
    cv_cal_p.add_argument("--thresholds", nargs="+", type=float, required=True, help="Probability thresholds evaluated uniformly across all OOF folds.")
    cv_cal_p.add_argument("--batch-size", type=int)
    cv_cal_p.add_argument("--num-workers", type=int)
    cv_cal_p.add_argument("--amp", dest="amp", action="store_true", default=False)
    cv_cal_p.add_argument("--no-amp", dest="amp", action="store_false")
    cv_cal_p.add_argument("--cpu", dest="cpu", action="store_true", default=False)
    cv_cal_p.add_argument("--gpu", dest="cpu", action="store_false")

    dep_audit_p = sub.add_parser("audit-deployment-errors", help="Rank and visualise false-positive/false-negative failure patterns from labelled deployment outputs.")
    dep_audit_p.add_argument("--deployment-dir", dest="deployment_dirs", required=True, action="append", type=Path, help="Deployment output directory. Repeat for multiple EMSR runs.")
    dep_audit_p.add_argument("--output-dir", required=True, type=Path, help="Directory for ranked CSV, JSON, HTML and forensic montages.")
    dep_audit_p.add_argument("--max-montages", type=int, default=30, help="Maximum number of worst-candidate montages to write.")

    raw_audit_p = sub.add_parser("audit-raw-alignment", help="Audit SAR/DEM/mask CRS, transform, shape, and nodata alignment before preprocessing.")
    raw_audit_p.add_argument("--raw-data-dir", required=True, type=Path)
    raw_audit_p.add_argument("--output-dir", required=True, type=Path)
    raw_audit_p.add_argument("--include-events", nargs="+", help="Optional EMSR event IDs to audit.")

    code_audit_p = sub.add_parser("audit-code", help="Scan public source/config/docs for development-history terms.")
    code_audit_p.add_argument("--project-root", type=Path, default=Path("."))
    code_audit_p.add_argument("--output-dir", required=True, type=Path)

    audit_p = sub.add_parser("audit-dataset", help="Audit processed tiles for foreground sparsity and mask integrity.")
    audit_p.add_argument("--processed-data-dir", required=True, type=Path)
    audit_p.add_argument("--output-dir", required=True, type=Path)
    audit_p.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    audit_p.add_argument("--samples-per-split", type=int, default=8)
    audit_p.add_argument("--write-plots", dest="write_plots", action="store_true", default=True)
    audit_p.add_argument("--no-write-plots", dest="write_plots", action="store_false")

    # Runtime logging options are accepted after every subcommand so notebook and
    # shell environments use the same ``floodmap <command> ...`` interface.
    for command_parser in sub.choices.values():
        if "--log-file" not in command_parser._option_string_actions:
            command_parser.add_argument(
                "--log-file",
                type=Path,
                help="Optional command log path. Console logging always remains enabled.",
            )
        if "--log-level" not in command_parser._option_string_actions:
            command_parser.add_argument(
                "--log-level",
                choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                default="INFO",
                help="Console and file logging level.",
            )
        if "--heartbeat-seconds" not in command_parser._option_string_actions:
            command_parser.add_argument(
                "--heartbeat-seconds",
                type=float,
                default=30.0,
                help="Emit a heartbeat after N seconds without recent progress; use 0 to disable.",
            )
        if "--plain-progress" not in command_parser._option_string_actions:
            command_parser.add_argument(
                "--plain-progress",
                dest="plain_progress",
                action="store_true",
                default=None,
                help="Force concise newline progress logs, roughly every 10%% of a finite operation.",
            )
            command_parser.add_argument(
                "--dynamic-progress",
                dest="plain_progress",
                action="store_false",
                help="Allow dynamic progress bars when the terminal supports them.",
            )

    return parser


def _dispatch_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command == "doctor":
        from floods.environment import (
            format_environment_report,
            run_environment_checks,
            write_environment_report,
        )

        payload = run_environment_checks()
        print(format_environment_report(payload))
        if args.json_output:
            path = write_environment_report(payload, args.json_output)
            LOG.info("Environment report written to: %s", path)
        if args.strict and payload["status"] != "ok":
            raise RuntimeError("Flood Extent Mapping environment checks failed")
        return
    if args.command == "model-catalog":
        import json
        from floods.pretrained import list_weight_sources
        rows = list_weight_sources()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print("weights_source | provider | encoder | decoder | normalization | modalities")
            for row in rows:
                modality_text = ",".join("+".join(v) for v in row["allowed_modality_sets"]) or "configurable"
                print(f"{row['name']} | {row['provider']} | {row['default_encoder'] or '-'} | {row['default_decoder'] or '-'} | {row['normalization_mode']} | {modality_text}")
        return
    if args.command == "calibrate-event-cv":
        from floods.event_cv_calibration import run_event_cv_calibration
        run_event_cv_calibration(
            cv_dir=args.cv_dir,
            processed_data_dir=args.processed_data_dir,
            output_dir=args.output_dir,
            candidate=args.candidate,
            architecture=args.architecture,
            modalities=args.modalities,
            fold_indices=args.fold_indices,
            thresholds=args.thresholds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            cpu=args.cpu,
        )
        return
    if args.command == "event-cv":
        config = _load_config_from_yaml(args.config, TrainConfig)
        from floods.event_cv import run_event_cross_validation
        run_event_cross_validation(
            config,
            processed_data_dir=args.processed_data_dir,
            output_dir=args.output_dir,
            candidates=args.candidates,
            architectures=args.architectures,
            modality_sets=args.modality_sets,
            folds=args.folds,
            fold_indices=args.fold_indices,
            source_split=args.source_split,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            encoder_lr=args.encoder_lr,
            decoder_lr=args.decoder_lr,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            loss=args.loss,
            loss_alpha=args.loss_alpha,
            loss_beta=args.loss_beta,
            bce_weight=args.bce_weight,
            tversky_weight=args.tversky_weight,
            augmentation_profile=args.augmentation_profile,
            pretrained=args.pretrained,
            amp=args.amp,
            cpu=args.cpu,
            thresholds=args.thresholds,
            q_min=args.q_min,
            q_max=args.q_max,
            max_pixels_per_file=args.max_pixels_per_file,
            plan_only=args.plan_only,
            skip_completed=args.skip_completed,
        )
        return
    if args.command == "derive-features":
        from floods.derived_features import build_derived_features
        build_derived_features(
            processed_data_dir=args.processed_data_dir,
            splits=args.splits,
            log_ratio_eps=args.log_ratio_eps,
            tpi_radius_pixels=args.tpi_radius_pixels,
            overwrite=args.overwrite,
        )
        return
    
    if args.command == "audit-feature-separability":
        from floods.feature_audit import audit_feature_separability
        audit_feature_separability(
            processed_data_dir=args.processed_data_dir,
            output_dir=args.output_dir,
            fit_split=args.fit_split,
            eval_split=args.eval_split,
            base_modalities=args.base_modalities,
            extended_modalities=args.extended_modalities,
            max_pixels_per_class_per_tile=args.max_pixels_per_class_per_tile,
            max_total_pixels_per_split=args.max_total_pixels_per_split,
            seed=args.seed,
        )
        return
    
    if args.command == "audit-water-prior":
        config = _load_config_from_yaml(args.config, TrainConfig)
        _set_when_provided(config.data, "path", str(args.processed_data_dir))
        _set_when_provided(config, "image_size", args.image_size)
        _set_when_provided(config.trainer, "batch_size", args.batch_size)
        _set_when_provided(config.trainer, "num_workers", args.num_workers)
        _set_when_provided(config.trainer, "amp", args.amp)
        _set_when_provided(config.trainer, "cpu", args.cpu)
        _set_when_provided(config.data, "normalization_stats_path", args.normalization_stats_path)
        _set_when_provided(config.data, "normalization_mode", args.normalization_mode)
        _apply_eval_model_overrides(config, args)
        _set_input_modalities(config, args.input_modalities)
        from floods.water_prior_audit import audit_water_prior
        audit_water_prior(
            config=config,
            checkpoint_path=args.checkpoint_path,
            processed_data_dir=args.processed_data_dir,
            output_dir=args.output_dir,
            prior_cache_dir=args.prior_cache_dir,
            split=args.split,
            include_events=args.include_events,
            exclude_events=args.exclude_events,
            model_thresholds=args.model_thresholds,
            occurrence_thresholds=args.occurrence_thresholds,
            penalty_strengths=args.penalty_strengths,
            min_component_areas=args.min_component_areas,
            reference_threshold=args.reference_threshold,
            reference_min_component_area=args.reference_min_component_area,
            max_recall_drop=args.max_recall_drop,
            include_hard_exclusion=args.include_hard_exclusion,
            include_soft_penalty=args.include_soft_penalty,
            offline_prior_cache=args.offline_prior_cache,
            allow_incomplete_prior=args.allow_incomplete_prior,
            max_samples=args.max_samples,
        )
        return
    
    if args.command == "audit-hysteresis-postprocess":
        config = _load_config_from_yaml(args.config, TrainConfig)
        _set_when_provided(config.data, "path", str(args.processed_data_dir))
        _set_when_provided(config, "image_size", args.image_size)
        _set_when_provided(config.trainer, "batch_size", args.batch_size)
        _set_when_provided(config.trainer, "num_workers", args.num_workers)
        _set_when_provided(config.trainer, "amp", args.amp)
        _set_when_provided(config.trainer, "cpu", args.cpu)
        _set_when_provided(config.data, "normalization_stats_path", args.normalization_stats_path)
        _set_when_provided(config.data, "normalization_mode", args.normalization_mode)
        _apply_eval_model_overrides(config, args)
        _set_input_modalities(config, args.input_modalities)
        from floods.hysteresis_audit import audit_hysteresis_postprocess
        audit_hysteresis_postprocess(
            config=config,
            checkpoint_path=args.checkpoint_path,
            processed_data_dir=args.processed_data_dir,
            output_dir=args.output_dir,
            split=args.split,
            include_events=args.include_events,
            exclude_events=args.exclude_events,
            fixed_thresholds=args.fixed_thresholds,
            low_thresholds=args.low_thresholds,
            high_thresholds=args.high_thresholds,
            min_seed_pixels=args.min_seed_pixels,
            min_component_areas=args.min_component_areas,
            reference_threshold=args.reference_threshold,
            reference_min_component_area=args.reference_min_component_area,
            max_recall_drop=args.max_recall_drop,
            max_empty_fp_rate_increase=args.max_empty_fp_rate_increase,
            connectivity=args.connectivity,
            max_samples=args.max_samples,
        )
        return
    
    if args.command == "prepare-derived-experiment":
        from floods.derived_experiment import prepare_derived_experiment_config
        base_config = _load_config_from_yaml(args.base_config, TrainConfig)
        prepare_derived_experiment_config(
            base_config,
            output_config=args.output_config,
            run_id=args.run_id,
            artifacts_dir=args.artifacts_dir,
            baseline_checkpoint=args.baseline_checkpoint,
            normalization_stats_path=args.normalization_stats_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            encoder_lr=args.encoder_lr,
            decoder_lr=args.decoder_lr,
            max_skipped_batch_fraction=args.max_skipped_batch_fraction,
        )
        return
    
    if args.command == "audit-deployment-errors":
        from floods.deployment_error_audit import audit_deployment_errors
        audit_deployment_errors(args.deployment_dirs, args.output_dir, max_montages=args.max_montages)
        return
    
    if args.command == "audit-raw-alignment":
        from floods.pipeline_audit import audit_raw_alignment
        audit_raw_alignment(args.raw_data_dir, args.output_dir, include_events=args.include_events)
        return
    
    if args.command == "audit-code":
        from floods.pipeline_audit import audit_code_quality
        audit_code_quality(args.project_root, args.output_dir)
        return
    
    if args.command == "ensemble-error-audit":
        if len(args.configs) != len(args.checkpoints):
            parser.error(f"--configs and --checkpoints must have the same length; got {len(args.configs)} and {len(args.checkpoints)}")
        configs = []
        for path in args.configs:
            cfg = _load_config_from_yaml(path, TrainConfig)
            _set_when_provided(cfg.data, "path", args.processed_data_dir)
            _set_when_provided(cfg, "image_size", args.image_size)
            _set_when_provided(cfg.trainer, "batch_size", args.batch_size)
            _set_when_provided(cfg.trainer, "num_workers", args.num_workers)
            _set_when_provided(cfg.trainer, "amp", args.amp)
            _set_when_provided(cfg.trainer, "cpu", args.cpu)
            _set_when_provided(cfg.data, "normalization_stats_path", args.normalization_stats_path)
            _set_when_provided(cfg.data, "normalization_mode", args.normalization_mode)
            _apply_eval_model_overrides(cfg, args)
            _set_input_modalities(cfg, args.input_modalities)
            configs.append(cfg)
        from floods.ensemble_error_audit import ensemble_error_audit_checkpoints
        ensemble_error_audit_checkpoints(configs=configs,
                                         checkpoint_paths=args.checkpoints,
                                         output_dir=args.output_dir,
                                         split=args.split,
                                         thresholds=args.thresholds,
                                         threshold=args.threshold,
                                         threshold_metric=args.threshold_metric,
                                         min_component_area=args.min_component_area,
                                         sweep_component_areas=args.sweep_component_areas,
                                         max_overlays_per_category=args.max_overlays_per_category,
                                         max_samples=args.max_samples,
                                         include_events=args.include_events,
                                         exclude_events=args.exclude_events,
                                         ensemble_method=args.ensemble_method,
                                         inference_mode=args.inference_mode,
                                         window_size=args.window_size,
                                         window_overlap=args.window_overlap,
                                         window_batch_size=args.window_batch_size)
        return
    
    if args.command == "ensemble-evaluate":
        if len(args.configs) != len(args.checkpoints):
            parser.error(f"--configs and --checkpoints must have the same length; got {len(args.configs)} and {len(args.checkpoints)}")
        configs = []
        for path in args.configs:
            cfg = _load_config_from_yaml(path, TrainConfig)
            _set_when_provided(cfg.data, "path", args.processed_data_dir)
            _set_when_provided(cfg, "image_size", args.image_size)
            _set_when_provided(cfg.trainer, "batch_size", args.batch_size)
            _set_when_provided(cfg.trainer, "num_workers", args.num_workers)
            _set_when_provided(cfg.trainer, "amp", args.amp)
            _set_when_provided(cfg.trainer, "cpu", args.cpu)
            _set_when_provided(cfg.data, "normalization_stats_path", args.normalization_stats_path)
            _set_when_provided(cfg.data, "normalization_mode", args.normalization_mode)
            _set_when_provided(cfg.trainer, "metric_mode", args.metric_mode)
            _apply_eval_model_overrides(cfg, args)
            _set_input_modalities(cfg, args.input_modalities)
            configs.append(cfg)
        from floods.ensemble_evaluation import ensemble_evaluate_checkpoints
        ensemble_evaluate_checkpoints(configs=configs,
                                      checkpoint_paths=args.checkpoints,
                                      split=args.split,
                                      thresholds=args.thresholds,
                                      threshold_metric=args.threshold_metric,
                                      metric_mode=args.metric_mode,
                                      include_events=args.include_events,
                                      exclude_events=args.exclude_events,
                                      ensemble_method=args.ensemble_method,
                                      inference_mode=args.inference_mode,
                                      window_size=args.window_size,
                                      window_overlap=args.window_overlap,
                                      window_batch_size=args.window_batch_size)
        return
    
    if args.command == "error-audit":
        config = _load_config_from_yaml(args.config, TrainConfig)
        _set_when_provided(config.data, "path", args.processed_data_dir)
        _set_when_provided(config, "image_size", args.image_size)
        _set_when_provided(config.trainer, "batch_size", args.batch_size)
        _set_when_provided(config.trainer, "num_workers", args.num_workers)
        _set_when_provided(config.trainer, "amp", args.amp)
        _set_when_provided(config.trainer, "cpu", args.cpu)
        _set_when_provided(config.data, "normalization_stats_path", args.normalization_stats_path)
        _set_when_provided(config.data, "normalization_mode", args.normalization_mode)
        _apply_eval_model_overrides(config, args)
        _set_input_modalities(config, args.input_modalities)
        from floods.error_audit import error_audit_checkpoint
        error_audit_checkpoint(config=config,
                               checkpoint_path=args.checkpoint_path,
                               output_dir=args.output_dir,
                               split=args.split,
                               thresholds=args.thresholds,
                               threshold=args.threshold,
                               threshold_metric=args.threshold_metric,
                               min_component_area=args.min_component_area,
                               sweep_component_areas=args.sweep_component_areas,
                               max_overlays_per_category=args.max_overlays_per_category,
                               max_samples=args.max_samples,
                               include_events=args.include_events,
                               exclude_events=args.exclude_events)
        return
    
    
    if args.command == "mine-hard-positives":
        config = _load_config_from_yaml(args.config, TrainConfig)
        _set_when_provided(config.data, "path", args.processed_data_dir)
        _set_when_provided(config, "image_size", args.image_size)
        _set_when_provided(config.trainer, "batch_size", args.batch_size)
        _set_when_provided(config.trainer, "num_workers", args.num_workers)
        _set_when_provided(config.trainer, "amp", args.amp)
        _set_when_provided(config.trainer, "cpu", args.cpu)
        _set_when_provided(config.data, "normalization_stats_path", args.normalization_stats_path)
        _set_when_provided(config.data, "normalization_mode", args.normalization_mode)
        _apply_eval_model_overrides(config, args)
        _set_input_modalities(config, args.input_modalities)
        from floods.hard_positive_regions import mine_hard_positive_regions
        mine_hard_positive_regions(config=config, checkpoint_path=args.checkpoint_path, output_dir=args.output_dir,
            split=args.split, threshold=args.threshold, crop_sizes=args.crop_sizes,
            min_component_area=args.min_component_area, min_fn_pixels=args.min_fn_pixels,
            min_label_fg_ratio=args.min_label_fg_ratio, min_valid_ratio=args.min_valid_ratio,
            max_regions_per_tile=args.max_regions_per_tile, nms_iou=args.nms_iou, max_samples=args.max_samples)
        return
    
    if args.command == "mine-hard-negatives":
        config = _load_config_from_yaml(args.config, TrainConfig)
        _set_when_provided(config.data, "path", args.processed_data_dir)
        _set_when_provided(config, "image_size", args.image_size)
        _set_when_provided(config.trainer, "batch_size", args.batch_size)
        _set_when_provided(config.trainer, "num_workers", args.num_workers)
        _set_when_provided(config.trainer, "amp", args.amp)
        _set_when_provided(config.trainer, "cpu", args.cpu)
        _set_when_provided(config.data, "normalization_stats_path", args.normalization_stats_path)
        _set_when_provided(config.data, "normalization_mode", args.normalization_mode)
        _apply_eval_model_overrides(config, args)
        _set_input_modalities(config, args.input_modalities)
        from floods.hard_negative_regions import mine_hard_negative_regions
        mine_hard_negative_regions(
            config=config, checkpoint_path=args.checkpoint_path, output_dir=args.output_dir,
            split=args.split, threshold=args.threshold, crop_sizes=args.crop_sizes,
            min_component_area=args.min_component_area, min_fp_pixels=args.min_fp_pixels,
            max_label_fg_ratio=args.max_label_fg_ratio, min_valid_ratio=args.min_valid_ratio,
            max_regions_per_tile=args.max_regions_per_tile, nms_iou=args.nms_iou,
            max_samples=args.max_samples,
        )
        return
    
    
    if args.command == "discover-scene":
        from floods.deployment import discover_scene
        rows = discover_scene(scene_dir=args.scene_dir,
                              output_file=args.output_file,
                              scene_id=args.scene_id,
                              candidate_prefix=args.candidate_prefix,
                              candidate_name_template=args.candidate_name_template)
        ready = sum(1 for row in rows if row.get("status") == "ready")
        LOG.info("Discovered %d ready SAR candidate(s) from %d inventory row(s).", ready, len(rows))
        if args.output_file:
            LOG.info("Inventory written to: %s", args.output_file)
        return
    
    if args.command == "predict-scene":
        from floods.deployment import predict_scene
        predict_scene(manifest_path=args.manifest,
                      sar_path=args.sar_path,
                      dem_path=args.dem_path,
                      mask_path=args.mask_path,
                      mask_dir=args.mask_dir,
                      evaluate=args.evaluate,
                      output_mask=args.output_mask,
                      output_probability=args.output_probability,
                      device=args.deployment_device or ("cuda" if not getattr(args, "cpu", False) else "cpu"),
                      scene_dir=args.scene_dir,
                      input_csv=args.input_csv,
                      dem_dir=args.dem_dir,
                      output_dir=args.output_dir,
                      output_prefix=args.output_prefix,
                      scene_id=args.scene_id,
                      candidate_prefix=args.candidate_prefix,
                      candidate_name_template=args.candidate_name_template,
                      sar_selection=args.sar_selection,
                      sar_date=args.sar_date,
                      mosaic_compatible_sar_tiles=args.mosaic_compatible_sar_tiles,
                      mosaic_undated=args.mosaic_undated,
                      mosaic_mode=args.mosaic_mode,
                      write_probability=args.write_probability,
                      write_previews=args.write_previews,
                      write_overlay=args.write_overlay,
                      write_uncertainty=args.write_uncertainty,
                      write_html_report=args.write_html_report,
                      display_inline=args.display_inline,
                      explain=args.explain,
                      explain_per_modality=args.explain_per_modality,
                      write_window_diagnostics=args.write_window_diagnostics,
                      window_blend=args.window_blend,
                      output_mode=args.output_mode,
                      prediction_only=args.prediction_only)
        return
    
    if args.command == "export-deployment":
        from floods.deployment import write_deployment_manifest
        write_deployment_manifest(output_file=args.output_file,
                                  configs=args.configs,
                                  checkpoints=args.checkpoints,
                                  model_name=args.model_name,
                                  ensemble_method=args.ensemble_method,
                                  threshold=args.threshold,
                                  min_component_area=args.min_component_area,
                                  input_modalities=args.input_modalities,
                                  normalization_stats_path=args.normalization_stats_path,
                                  normalization_mode=args.normalization_mode,
                                  inference_mode=args.inference_mode,
                                  window_size=args.window_size,
                                  window_overlap=args.window_overlap,
                                  window_batch_size=args.window_batch_size,
                                  window_blend=args.window_blend,
                                  notes=args.notes,
                                  copy_assets=args.copy_assets,
                                  assets_directory=args.assets_directory)
        return
    
    if args.command == "compare-models":
        from floods.model_comparison import SingleModelSpec, EnsembleModelSpec, compare_models
    
        def load_eval_config(path_text: str):
            cfg = _load_config_from_yaml(Path(path_text), TrainConfig)
            _set_when_provided(cfg.data, "path", args.processed_data_dir)
            _set_when_provided(cfg, "image_size", args.image_size)
            _set_when_provided(cfg.trainer, "batch_size", args.batch_size)
            _set_when_provided(cfg.trainer, "num_workers", args.num_workers)
            _set_when_provided(cfg.trainer, "amp", args.amp)
            _set_when_provided(cfg.trainer, "cpu", args.cpu)
            _set_when_provided(cfg.data, "normalization_stats_path", args.normalization_stats_path)
            _set_when_provided(cfg.data, "normalization_mode", args.normalization_mode)
            _set_when_provided(cfg.model, "pretrained", args.pretrained)
            _set_input_modalities(cfg, args.input_modalities)
            return cfg
    
        singles = []
        for item in args.model or []:
            name, config_path, checkpoint_path = item
            singles.append(SingleModelSpec(name=name, config=load_eval_config(config_path), checkpoint=Path(checkpoint_path)))
    
        ensembles = []
        for item in args.ensemble or []:
            if len(item) < 4:
                parser.error("--ensemble requires NAME METHOD CONFIG:CHECKPOINT CONFIG:CHECKPOINT ...")
            name, method, *members = item
            configs = []
            checkpoints = []
            for member in members:
                if ":" not in member:
                    parser.error(f"Invalid ensemble member '{member}'. Use CONFIG:CHECKPOINT")
                config_path, checkpoint_path = member.split(":", 1)
                configs.append(load_eval_config(config_path))
                checkpoints.append(Path(checkpoint_path))
            ensembles.append(EnsembleModelSpec(name=name, configs=configs, checkpoints=checkpoints, method=method))
    
        compare_models(single_models=singles,
                       ensembles=ensembles,
                       output_dir=args.output_dir,
                       split=args.split,
                       thresholds=args.thresholds,
                       threshold_metric=args.threshold_metric,
                       metric_mode=args.metric_mode,
                       include_events=args.include_events,
                       exclude_events=args.exclude_events,
                       inference_mode=args.inference_mode,
                       window_size=args.window_size,
                       window_overlap=args.window_overlap,
                       window_batch_size=args.window_batch_size,
                       plot_metrics=args.plot_metrics,
                       write_confusion_matrices=args.confusion_matrix_plots)
        return
    
    if args.command == "audit-domain-shift":
        config = _load_config_from_yaml(args.config, TrainConfig) if args.config else None
        processed_data_dir = args.processed_data_dir or (Path(config.data.path) if config is not None else None)
        if processed_data_dir is None:
            parser.error("audit-domain-shift requires --processed-data-dir or --config with data.path")
        if args.input_modalities:
            modalities = args.input_modalities
        elif config is not None and getattr(config.data, "input_modalities", None):
            modalities = list(config.data.input_modalities)
        else:
            modalities = ["vv", "vh", "dem"]
        normalization_stats_path = args.normalization_stats_path
        if normalization_stats_path is None and config is not None:
            configured_stats = getattr(config.data, "normalization_stats_path", None)
            normalization_stats_path = Path(configured_stats) if configured_stats else None
        normalization_mode = args.normalization_mode
        if normalization_mode is None and config is not None:
            normalization_mode = getattr(config.data, "normalization_mode", None)
        normalization_mode = normalization_mode or "robust_percentile"
        from floods.domain_shift_audit import audit_domain_shift
        audit_domain_shift(
            processed_data_dir=processed_data_dir,
            output_dir=args.output_dir,
            reference_split=args.reference_split,
            target_split=args.target_split,
            target_events=args.target_events,
            reference_events=args.reference_events,
            exclude_reference_events=args.exclude_reference_events,
            input_modalities=modalities,
            normalization_stats_path=normalization_stats_path,
            normalization_mode=normalization_mode,
            max_reference_tiles=args.max_reference_tiles,
            max_target_tiles=args.max_target_tiles,
            max_pixels_per_tile=args.max_pixels_per_tile,
            max_pixels_per_class_per_tile=args.max_pixels_per_class_per_tile,
            max_total_pixels_per_domain=args.max_total_pixels_per_domain,
            domain_classifier_reference_ratio=args.domain_classifier_reference_ratio,
            seed=args.seed,
            write_plots=args.write_plots,
        )
        return

    if args.command == "audit-domain-failure-link":
        from floods.domain_failure_audit import audit_domain_failure_link
        audit_domain_failure_link(
            tile_features_csv=args.tile_features_csv,
            tile_error_metrics_csv=args.tile_error_metrics_csv,
            output_dir=args.output_dir,
            target_events=args.target_events,
            max_recall=args.max_recall,
            neighbours=args.neighbours,
            seed=args.seed,
            write_plots=args.write_plots,
        )
        return

    if args.command == "audit-modality-ablation":
        config = _load_config_from_yaml(args.config, TrainConfig)
        _set_when_provided(config.data, "path", args.processed_data_dir)
        _set_when_provided(config, "image_size", args.image_size)
        _set_when_provided(config.trainer, "batch_size", args.batch_size)
        _set_when_provided(config.trainer, "num_workers", args.num_workers)
        _set_when_provided(config.trainer, "amp", args.amp)
        _set_when_provided(config.trainer, "cpu", args.cpu)
        _set_when_provided(config.data, "normalization_stats_path", args.normalization_stats_path)
        _set_when_provided(config.data, "normalization_mode", args.normalization_mode)
        _apply_eval_model_overrides(config, args)
        _set_input_modalities(config, args.input_modalities)
        from floods.modality_ablation_audit import audit_modality_ablation
        audit_modality_ablation(
            config=config,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            split=args.split,
            target_events=args.target_events,
            include_events=args.include_events,
            exclude_events=args.exclude_events,
            ablations=args.ablations,
            thresholds=args.thresholds,
            threshold_metric=args.threshold_metric,
            operating_threshold=args.operating_threshold,
            max_samples=args.max_samples,
        )
        return

    if args.command == "audit-dataset":
        from floods.audit import audit_dataset
        audit_dataset(args.processed_data_dir, args.output_dir, args.splits,
                      samples_per_split=args.samples_per_split, write_plots=args.write_plots)
        return
    if args.command == "fit-normalization":
        if args.output_file.exists() and args.output_file.is_dir():
            parser.error(f"--output-file must be a JSON file path, not a directory: {args.output_file}")
        if args.output_file.suffix.lower() != ".json":
            parser.error(f"--output-file should end with .json: {args.output_file}")
        from floods.normalization import fit_normalization_stats
        fit_normalization_stats(processed_data_dir=args.processed_data_dir,
                                output_file=args.output_file,
                                split=args.split,
                                input_modalities=args.input_modalities,
                                q_min=args.q_min,
                                q_max=args.q_max,
                                max_pixels_per_file=args.max_pixels_per_file,
                                seed=args.seed,
                                preserve_channel_stats_from=args.preserve_channel_stats_from)
        return
    
    
    if args.command == "continual-train":
        args = _fill_missing_training_args(args)
        config = _load_config_from_yaml(args.config, TrainConfig)
        config = _apply_training_overrides(config, args)
        if args.epochs_per_task:
            config.trainer.max_epochs = int(args.epochs_per_task)
        if not config.name:
            config.name = "continual_learning"
        from floods.continual import continual_train
        continual_train(config=config,
                        activations_json_path=args.activations_json_path,
                        strategies=args.strategies,
                        task_year_ranges=args.task_year_ranges,
                        epochs_per_task=args.epochs_per_task,
                        replay_buffer_size=args.replay_buffer_size,
                        replay_batch_size=args.replay_batch_size,
                        uncertainty_subset_fraction=args.uncertainty_subset_fraction,
                        eval_split=args.cl_eval_split,
                        eval_inference_mode=args.cl_eval_inference_mode,
                        window_size=args.window_size,
                        window_overlap=args.window_overlap,
                        window_batch_size=args.window_batch_size,
                        model_mode=args.cl_model_mode,
                        ensemble_members=args.ensemble_members,
                        ensemble_method=args.ensemble_method,
                        replay_mode=args.replay_mode,
                        resume=args.resume,
                        resume_from=args.resume_from,
                        resume_start_task=args.resume_start_task,
                        resume_start_epoch=args.resume_start_epoch)
        return
    
    with progress_logging_context():
        if args.command == "train":
            config = _load_training_config_for_args(args)
            from floods import training
            training.train(config)
        elif args.command == "preprocess":
            config = _load_config_from_yaml(args.config, PreparationConfig)
            config = _apply_preprocess_overrides(config, args)
            from floods import preproc
            preproc.preprocess_data(config=config)
        elif args.command == "audit-training-data":
            from floods.training_data_audit import audit_training_data
            audit_training_data(processed_data_dir=args.processed_data_dir, output_dir=args.output_dir, splits=args.splits, fail_on_leakage=args.fail_on_leakage)
        elif args.command == "audit-training-exposure":
            config = _load_config_from_yaml(args.config, TrainConfig)
            _set_when_provided(config.data, "path", args.processed_data_dir)
            from floods.training_exposure_audit import audit_training_exposure
            audit_training_exposure(config=config, output_dir=args.output_dir, epochs=args.epochs, seed=args.seed, negative_max_ratio=args.negative_max_ratio, negative_clusters=args.negative_clusters, sampler_profiles=args.sampler_profiles)
        elif args.command == "stats":
            config = _load_config_from_yaml(args.config, StatsConfig)
            config = _apply_stats_overrides(config, args)
            from floods import preproc
            preproc.compute_statistics(config=config)
        elif args.command == "pseudolabel":
            config = _load_config_from_yaml(args.config, PreparationConfig)
            from floods import preproc
            preproc.generate_pseudolabels(config=config)
        elif args.command == "test":
            config = _load_config_from_yaml(args.config, TestConfig)
            _set_when_provided(config, "checkpoint_path", args.checkpoint_path)
            _set_when_provided(config, "data_root", args.data_root)
            _set_when_provided(config, "store_predictions", args.store_predictions)
            from floods import testing
            testing.test(config)
        elif args.command == "evaluate":
            config = _load_config_from_yaml(args.config, TrainConfig)
            _set_when_provided(config.data, "path", args.processed_data_dir)
            _set_when_provided(config, "image_size", args.image_size)
            _set_when_provided(config.trainer, "batch_size", args.batch_size)
            _set_when_provided(config.trainer, "num_workers", args.num_workers)
            _set_when_provided(config.trainer, "amp", args.amp)
            _set_when_provided(config.trainer, "cpu", args.cpu)
            _set_when_provided(config.data, "normalization_stats_path", args.normalization_stats_path)
            _set_when_provided(config.data, "normalization_mode", args.normalization_mode)
            _set_when_provided(config.trainer, "metric_mode", args.metric_mode)
            _apply_eval_model_overrides(config, args)
            _set_input_modalities(config, args.input_modalities)
            from floods.evaluation import evaluate_checkpoint
            evaluate_checkpoint(config, checkpoint_path=args.checkpoint_path, split=args.split, thresholds=args.thresholds, threshold_metric=args.threshold_metric, metric_mode=args.metric_mode, include_events=args.include_events, exclude_events=args.exclude_events, inference_mode=args.inference_mode, window_size=args.window_size, window_overlap=args.window_overlap, window_batch_size=args.window_batch_size)
        else:
            parser.error(f"Unknown command: {args.command}")


def _default_command_log_file(args: argparse.Namespace) -> Optional[Path]:
    explicit = getattr(args, "log_file", None)
    if explicit is not None:
        return Path(explicit)
    command = str(getattr(args, "command", ""))
    if command in {"train", "continual-train"}:
        output_folder = getattr(args, "output_folder", None)
        run_id = getattr(args, "run_id", None)
        if output_folder and run_id:
            return Path(output_folder) / str(run_id) / "output.log"
        return None
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        return Path(output_dir) / "output.log"
    if command == "preprocess":
        processed = getattr(args, "processed_data_dir", None)
        if processed is not None:
            return Path(processed) / "preprocess.log"
    if command == "derive-features":
        processed = getattr(args, "processed_data_dir", None)
        if processed is not None:
            return Path(processed) / "derive_features.log"
    output_file = getattr(args, "output_file", None)
    if output_file is not None:
        output_file = Path(output_file)
        return output_file.with_suffix(output_file.suffix + ".log")
    output_mask = getattr(args, "output_mask", None)
    if output_mask is not None:
        output_mask = Path(output_mask)
        return output_mask.with_suffix(output_mask.suffix + ".log")
    return None


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_cli_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv_list)
    if getattr(args, "plain_progress", None) is True:
        os.environ["FLOODMAP_PLAIN_PROGRESS"] = "1"
    elif getattr(args, "plain_progress", None) is False:
        os.environ.pop("FLOODMAP_PLAIN_PROGRESS", None)
    prepare_logging(getattr(args, "log_level", "INFO"))
    command_line = shlex.join(["floodmap", *argv_list])
    log_file = _default_command_log_file(args)
    with command_logging(
        args.command,
        log_file=log_file,
        argv_text=command_line,
        heartbeat_seconds=getattr(args, "heartbeat_seconds", 30.0),
    ):
        with progress_logging_context():
            _dispatch_command(args, parser)


if __name__ == "__main__":
    main(sys.argv[1:])
