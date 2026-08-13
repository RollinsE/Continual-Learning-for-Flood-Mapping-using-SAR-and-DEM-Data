from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from floods.config import TrainConfig
from floods.logging.tensorboard import TensorBoardLogger
from floods.models.base import Segmenter
from floods.sampling_modes import validate_sampling_modes
from floods.prepare import compute_binary_pos_weight_from_labels, inverse_transform, prepare_datasets, prepare_event_balanced_stratified_sampler, prepare_foreground_balanced_sampler, prepare_metrics, prepare_hard_example_sampler, prepare_model, prepare_sampler, prepare_stratified_sampler
from floods.hard_negative_regions import prepare_hard_negative_region_sampler
from floods.hard_positive_regions import prepare_hard_positive_region_sampler
from floods.resume import active_sampler_name, build_resume_signature, resolve_resume_checkpoint
from floods.evaluation import load_checkpoint_state
from floods.trainer.callbacks import Checkpoint, DisplaySamples, EarlyStopping, EarlyStoppingCriterion
from floods.trainer.flood import (
    FloodTrainer, GroupDROFloodTrainer, GroupDROMultiBranchTrainer, MultiBranchTrainer,
)
from floods.group_dro import EventIndexedDataset
from floods.utils.common import code_revision, config_to_plain_dict, flatten_config, get_logger, init_experiment, store_config
from floods.utils.gis import as_image, rgb_ratio
from floods.utils.data_preflight import validate_training_data_path
from floods.utils.ml import load_class_weights, seed_everything, seed_worker
from floods.checkpoint_adaptation import adapt_input_state_dict

LOG = get_logger(__name__)


def prepare_training_sampler(config: TrainConfig, train_set):
    """Create the training sampler and return whether the DataLoader should shuffle."""
    validate_sampling_modes(config.data)
    training_shuffle = True
    training_sampler = None

    if config.data.hard_positive_region_sampling:
        training_shuffle = False
        training_sampler = prepare_hard_positive_region_sampler(
            dataset=train_set, manifest_path=config.data.hard_positive_manifest,
            weight=config.data.hard_positive_region_weight,
            max_fraction=config.data.hard_positive_region_max_fraction,
            samples_multiplier=config.data.weighted_samples_multiplier)
    elif config.data.hard_negative_region_sampling:
        training_shuffle = False
        training_sampler = prepare_hard_negative_region_sampler(
            dataset=train_set,
            manifest_path=config.data.hard_negative_manifest,
            weight=config.data.hard_negative_region_weight,
            max_fraction=config.data.hard_negative_region_max_fraction,
            samples_multiplier=config.data.weighted_samples_multiplier,
        )
    elif config.data.hard_example_sampling:
        training_shuffle = False
        training_sampler = prepare_hard_example_sampler(dataset=train_set,
                                                        hard_example_csv=config.data.hard_example_csv,
                                                        hard_example_categories=config.data.hard_example_categories,
                                                        hard_example_fg_bins=config.data.hard_example_fg_bins,
                                                        hard_example_max_f1=config.data.hard_example_max_f1,
                                                        hard_example_weight=config.data.hard_example_weight,
                                                        hard_example_max_fraction=config.data.hard_example_max_fraction,
                                                        samples_multiplier=config.data.weighted_samples_multiplier)
    elif config.data.event_balanced_sampling:
        training_shuffle = False
        training_sampler = prepare_event_balanced_stratified_sampler(dataset=train_set,
                                                                     fg_bin_edges=config.data.fg_bin_edges,
                                                                     fg_bin_sample_weights=config.data.fg_bin_sample_weights,
                                                                     cache_hash=config.data.cache_hash,
                                                                     cache_dir=config.data.cache_dir,
                                                                     force_recompute=config.data.clear_cache,
                                                                     samples_multiplier=config.data.weighted_samples_multiplier,
                                                                     event_balance_power=config.data.event_balance_power,
                                                                     tile_weight_cap=config.data.event_tile_weight_cap)
    elif config.data.stratified_sampling:
        training_shuffle = False
        training_sampler = prepare_stratified_sampler(dataset=train_set,
                                                      fg_bin_edges=config.data.fg_bin_edges,
                                                      fg_bin_sample_weights=config.data.fg_bin_sample_weights,
                                                      cache_hash=config.data.cache_hash,
                                                      cache_dir=config.data.cache_dir,
                                                      force_recompute=config.data.clear_cache,
                                                      samples_multiplier=config.data.weighted_samples_multiplier)
    elif config.data.foreground_balanced_sampling:
        training_shuffle = False
        training_sampler = prepare_foreground_balanced_sampler(dataset=train_set,
                                                              foreground_sample_ratio=config.data.foreground_sample_ratio,
                                                              foreground_min_ratio=config.data.foreground_min_ratio,
                                                              cache_hash=config.data.cache_hash,
                                                              cache_dir=config.data.cache_dir,
                                                              force_recompute=config.data.clear_cache,
                                                              samples_multiplier=config.data.weighted_samples_multiplier)
    elif config.data.weighted_sampling:
        training_shuffle = False
        training_sampler = prepare_sampler(dataset=train_set,
                                           smoothing=config.data.sample_smoothing,
                                           cache_hash=config.data.cache_hash,
                                           cache_dir=config.data.cache_dir,
                                           force_recompute=config.data.clear_cache,
                                           samples_multiplier=config.data.weighted_samples_multiplier)
    else:
        LOG.info("Training sampler: standard shuffled sampling.")
    return training_shuffle, training_sampler


def train(config: TrainConfig):
    validate_training_data_path(config.data)
    torch.autograd.set_detect_anomaly(bool(config.trainer.detect_anomaly))
    if config.trainer.amp and not config.trainer.cpu and torch.cuda.is_available():
        if not torch.backends.cudnn.enabled:
            raise RuntimeError("CUDA AMP requires the cuDNN backend to be enabled")
    log_name = "output.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=config, log_name=log_name)
    if config.init_checkpoint and (config.resume or config.resume_from):
        raise ValueError("--init-checkpoint cannot be combined with --resume or --resume-from")
    init_checkpoint = Path(config.init_checkpoint).expanduser() if config.init_checkpoint else None
    if init_checkpoint is not None and not init_checkpoint.exists():
        raise FileNotFoundError(f"Initialisation checkpoint not found: {init_checkpoint}")
    resume_checkpoint = resolve_resume_checkpoint(config=config, model_folder=model_folder)
    if resume_checkpoint is not None:
        # A complete training checkpoint already contains encoder weights, so avoid a
        # redundant Hugging Face download before the checkpoint is loaded.
        config.model.pretrained = False
    elif init_checkpoint is not None:
        config.model.pretrained = False
        LOG.info("Fresh fine-tune requested from model weights: %s", init_checkpoint)
    elif not config.trainer.save_last:
        LOG.warning("Resumable checkpoints are disabled (trainer.save_last=false). An interrupted session will require a fresh run.")
    config_path = out_folder / "config.yaml"
    LOG.info("Run started: %s", exp_id)
    LOG.info("Output folder: %s", out_folder)
    LOG.info("Configuration: %s", config_path)
    if resume_checkpoint is None:
        store_config(config, path=config_path)
    else:
        # Preserve the original effective run plan. A resume invocation is recorded
        # separately and must never overwrite config.yaml with baseline defaults.
        if not config_path.exists():
            store_config(config, path=config_path)
        store_config(config, path=out_folder / "last_resume_config.yaml")

    LOG.info(
        "Training plan: target_epochs=%d | patience=%d | sampler=%s | objective=%s | batch_size=%d | encoder_lr=%.3g | decoder_lr=%.3g",
        int(config.trainer.max_epochs), int(config.trainer.patience), active_sampler_name(config),
        "event_group_dro" if config.trainer.group_dro else "erm",
        int(config.trainer.batch_size), float(config.optimizer.encoder_lr), float(config.optimizer.decoder_lr),
    )
    LOG.info(
        "Precision policy: amp=%s | AMP float32 retry=%s | skip non-finite=%s | "
        "max skipped fraction=%.4f | gradient clip=%s",
        bool(config.trainer.amp),
        bool(config.trainer.amp_full_precision_retry),
        bool(config.trainer.skip_nonfinite_batches),
        float(config.trainer.max_skipped_batch_fraction),
        "disabled" if not config.trainer.grad_clip_norm else f"{float(config.trainer.grad_clip_norm):.4f}",
    )
    if config.data.hard_positive_region_sampling:
        LOG.info("Hard-positive manifest: %s", config.data.hard_positive_manifest)
    elif config.data.hard_negative_region_sampling:
        LOG.info("Hard-negative manifest: %s", config.data.hard_negative_manifest)

    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed, deterministic=True)

    LOG.info("Loading datasets")
    num_classes = 1
    configured_modalities = [str(m).lower() for m in getattr(config.data, "input_modalities", [])]
    use_rgb = configured_modalities[:3] == ["r", "g", "b"] if configured_modalities else (config.data.in_channels - int(config.data.include_dem)) == 3
    train_set, valid_set = prepare_datasets(config=config, use_rgb=use_rgb)
    LOG.info("Dataset ready: train samples: %d | validation samples: %d", len(train_set), len(valid_set))

    try:
        accelerator = Accelerator(mixed_precision="fp16" if config.trainer.amp else "no", cpu=config.trainer.cpu)
    except TypeError:
        accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)
    accelerator.wait_for_everyone()

    training_shuffle, training_sampler = prepare_training_sampler(config=config, train_set=train_set)
    train_loader_dataset = train_set
    group_dro_event_names = None
    if config.trainer.group_dro:
        train_loader_dataset = EventIndexedDataset(train_set)
        group_dro_event_names = list(train_loader_dataset.event_names)
        if config.trainer.group_dro_min_weight * len(group_dro_event_names) >= 1.0:
            raise ValueError(
                "group_dro_min_weight is too large for the number of training events: "
                f"floor={config.trainer.group_dro_min_weight} events={len(group_dro_event_names)}"
            )
        LOG.info(
            "Event GroupDRO enabled: events=%d | eta=%.5f | min_weight=%.6f | ERM warmup=%d epoch(s)",
            len(group_dro_event_names), float(config.trainer.group_dro_eta),
            float(config.trainer.group_dro_min_weight), int(config.trainer.group_dro_warmup_epochs),
        )
        LOG.info("GroupDRO event order: %s", ", ".join(group_dro_event_names))
    train_loader = DataLoader(dataset=train_loader_dataset,
                              sampler=training_sampler,
                              batch_size=config.trainer.batch_size,
                              shuffle=training_shuffle,
                              num_workers=config.trainer.num_workers,
                              worker_init_fn=seed_worker,
                              drop_last=True)
    validation_event_names = None
    valid_loader_dataset = valid_set
    if getattr(config.trainer, "event_macro_validation", False):
        valid_loader_dataset = EventIndexedDataset(valid_set, require_multiple=False)
        validation_event_names = list(valid_loader_dataset.event_names)
        LOG.info("Event-macro validation enabled: events=%d | %s", len(validation_event_names), ", ".join(validation_event_names))
    valid_loader = DataLoader(dataset=valid_loader_dataset,
                              batch_size=config.trainer.batch_size,
                              shuffle=False,
                              num_workers=config.trainer.num_workers,
                              worker_init_fn=seed_worker)
    LOG.info("Preparing model")
    model: Segmenter = prepare_model(config=config, num_classes=num_classes).to(accelerator.device)
    if init_checkpoint is not None:
        state = load_checkpoint_state(init_checkpoint)
        state, adapted_keys = adapt_input_state_dict(
            model, state, mode=getattr(config, "init_channel_adaptation", "strict")
        )
        model.load_state_dict(state, strict=not config.model.multibranch)
        if adapted_keys:
            LOG.info(
                "Adapted checkpoint input weights with zero-initialised added channels: %s",
                ", ".join(adapted_keys),
            )
        LOG.info("Initialised model weights from %s; optimiser, scheduler, epochs, and early stopping start fresh.", init_checkpoint)

    params = [{"params": model.encoder_params(), "lr": config.optimizer.encoder_lr},
              {"params": model.decoder_params(), "lr": config.optimizer.decoder_lr}]
    optimizer = config.optimizer.instantiate(params)
    scheduler = config.scheduler.instantiate(optimizer)
    weights = None
    if config.data.class_weights:
        weights = load_class_weights(Path(config.data.class_weights), device=accelerator.device, normalize=False)
        LOG.info("Using class weights: %s", str(weights))
    if config.data.pos_weight_from_train:
        pos_weight_value = compute_binary_pos_weight_from_labels(train_set.label_files,
                                                                 max_value=config.data.pos_weight_max,
                                                                 cache_hash=config.data.cache_hash,
                                                                 cache_dir=config.data.cache_dir,
                                                                 force_recompute=config.data.clear_cache)
        weights = torch.tensor([pos_weight_value], dtype=torch.float32, device=accelerator.device)
        LOG.info("Using train-derived positive-class weight for BCE/focal terms: %.4f (clipped max %.4f)",
                 pos_weight_value, float(config.data.pos_weight_max))
    loss = config.loss.instantiate(ignore_index=255, weight=weights)

    monitored = config.trainer.monitor.name
    if getattr(config.trainer, "event_macro_validation", False):
        if not config.trainer.threshold_sweep:
            raise ValueError("Event-macro validation requires threshold_sweep=true")
        monitored = "best_event_macro_f1"
        LOG.info("Model selection metric: %s (event-macro threshold sweep)", monitored)
    elif config.trainer.threshold_sweep and config.trainer.monitor_threshold_sweep:
        monitored = f"best_{config.trainer.threshold_metric}"
        LOG.info("Model selection metric: %s (threshold sweep)", monitored)
    else:
        LOG.info("Model selection metric: %s", monitored)
    train_metrics, valid_metrics = prepare_metrics(config, device=accelerator.device)
    logger = TensorBoardLogger(log_folder=logs_folder, comment=config.comment)
    LOG.debug("Logging flattened configuration to TensorBoard")
    logger.log_table("config", flatten_config(config_to_plain_dict(config)))

    num_samples = int(config.visualize) * config.num_samples
    if config.visualize:
        LOG.debug("Sample visualization enabled for %d validation batches", num_samples)

    if config.trainer.group_dro:
        trainer_cls = GroupDROMultiBranchTrainer if config.model.multibranch else GroupDROFloodTrainer
    else:
        trainer_cls = MultiBranchTrainer if config.model.multibranch else FloodTrainer
    trainer_kwargs = {}
    if config.trainer.group_dro:
        trainer_kwargs.update(
            event_names=group_dro_event_names,
            group_dro_eta=config.trainer.group_dro_eta,
            group_dro_min_weight=config.trainer.group_dro_min_weight,
            group_dro_warmup_epochs=config.trainer.group_dro_warmup_epochs,
        )
    trainer = trainer_cls(accelerator=accelerator,
                          model=model,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          criterion=loss,
                          categories=train_set.categories(),
                          train_metrics=train_metrics,
                          val_metrics=valid_metrics,
                          logger=logger,
                          sample_batches=num_samples,
                          debug=config.debug,
                          checkpoint_dir=model_folder,
                          resume_from=resume_checkpoint,
                          auto_resume=False,
                          save_last=config.trainer.save_last,
                          save_epoch_checkpoints=config.trainer.save_epoch_checkpoints,
                          extend_epochs=config.trainer.extend_epochs,
                          reset_early_stopping_on_resume=config.trainer.reset_early_stopping_on_resume,
                          grad_clip_norm=config.trainer.grad_clip_norm,
                          skip_nonfinite_batches=config.trainer.skip_nonfinite_batches,
                          amp_full_precision_retry=config.trainer.amp_full_precision_retry,
                          max_skipped_batch_fraction=config.trainer.max_skipped_batch_fraction,
                          progress_bar=config.trainer.progress_bar,
                          progress_log_interval=config.trainer.progress_log_interval,
                          progress_label=config.trainer.progress_label,
                          max_epochs=config.trainer.max_epochs,
                          threshold_sweep=config.trainer.threshold_sweep,
                          thresholds=config.trainer.thresholds,
                          threshold_metric=config.trainer.threshold_metric,
                          event_macro_validation=getattr(config.trainer, "event_macro_validation", False),
                          validation_event_names=validation_event_names,
                          resume_signature=build_resume_signature(config),
                          **trainer_kwargs)

    trainer.add_callback(EarlyStopping(call_every=1,
                                       metric=monitored,
                                       criterion=EarlyStoppingCriterion.maximum,
                                       patience=config.trainer.patience))
    trainer.add_callback(Checkpoint(call_every=1,
                                    monitor=monitored,
                                    model_folder=model_folder,
                                    save_best=True,
                                    verbose=False))
    if config.visualize:
        image_trf = as_image if use_rgb else rgb_ratio
        trainer.add_callback(DisplaySamples(inverse_transform=inverse_transform(mean=train_set.mean(), std=train_set.std()),
                                            image_transform=image_trf,
                                            slice_at=3 if use_rgb else 2,
                                            mask_palette=train_set.palette()))

    config.version = code_revision()
    trainer.fit(train_dataloader=train_loader, val_dataloader=valid_loader, max_epochs=config.trainer.max_epochs)
    best_score = trainer._scalar(trainer.best_score)
    if trainer.stop_reason == "interrupted":
        LOG.info("Training interrupted during epoch %d; last completed epoch: %d; best validation %s: %.4f",
                 trainer.current_epoch + 1, max(trainer.last_completed_epoch + 1, 0), monitored, best_score)
        completed_path = model_folder / "last.ckpt"
        partial_path = trainer.interrupted_checkpoint_path
        if trainer.last_completed_epoch >= 0 and completed_path.exists():
            LOG.info("Retained completed resume checkpoint at %s", completed_path)
            if partial_path is not None:
                LOG.info("Partial interrupted state saved separately to %s", partial_path)
        elif partial_path is not None:
            LOG.info("Partial first-epoch checkpoint saved to %s; resume will restart the interrupted epoch", partial_path)
        LOG.info("Experiment %s interrupted", exp_id)
    elif trainer.stop_reason == "early_stopping":
        LOG.info("Training stopped early at epoch %d (best validation %s: %.4f)",
                 trainer.current_epoch + 1, monitored, best_score)
        LOG.info("Experiment %s completed", exp_id)
    elif trainer.stop_reason == "nan_loss":
        LOG.info("Training stopped because a non-finite loss was detected at epoch %d (best validation %s: %.4f)",
                 trainer.current_epoch + 1, monitored, best_score)
        LOG.info("Experiment %s stopped", exp_id)
    else:
        LOG.info("Training completed at epoch %d (best validation %s: %.4f)",
                 trainer.current_epoch + 1, monitored, best_score)
        LOG.info("Experiment %s completed", exp_id)
    checkpoint_candidates = sorted(model_folder.glob(f"model-*_{monitored}-*.pth"))
    best_checkpoint = checkpoint_candidates[-1] if checkpoint_candidates else None
    return {
        "run_id": exp_id,
        "output_dir": str(out_folder),
        "model_dir": str(model_folder),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        "best_score": best_score,
        "best_epoch": int(trainer.best_epoch + 1) if trainer.best_epoch is not None else None,
        "monitor": monitored,
        "stop_reason": trainer.stop_reason or "completed",
    }

