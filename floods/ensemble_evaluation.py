from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from floods.utils.console import progress_iter

from floods.evaluation import (
    BinaryThresholdSweep,
    BatchAverageMetrics,
    _filter_dataset_by_events,
    load_checkpoint_state,
)
from floods.utils.common import get_logger
from floods.sliding_window import ensemble_sliding_window_logits

LOG = get_logger(__name__)


def _apply_common_eval_overrides(config: Any,
                                 processed_data_dir: Optional[str] = None,
                                 image_size: Optional[int] = None,
                                 batch_size: Optional[int] = None,
                                 num_workers: Optional[int] = None,
                                 amp: Optional[bool] = None,
                                 cpu: Optional[bool] = None,
                                 normalization_stats_path: Optional[str] = None,
                                 normalization_mode: Optional[str] = None,
                                 input_modalities: Optional[Sequence[str]] = None) -> Any:
    if processed_data_dir is not None:
        config.data.path = processed_data_dir
    if image_size is not None:
        config.image_size = int(image_size)
    if batch_size is not None:
        config.trainer.batch_size = int(batch_size)
    if num_workers is not None:
        config.trainer.num_workers = int(num_workers)
    if amp is not None:
        config.trainer.amp = bool(amp)
    if cpu is not None:
        config.trainer.cpu = bool(cpu)
    if normalization_stats_path is not None:
        config.data.normalization_stats_path = normalization_stats_path
    if normalization_mode is not None:
        config.data.normalization_mode = normalization_mode
    if input_modalities:
        from floods.modalities import canonicalize_modalities
        modalities = canonicalize_modalities(input_modalities)
        config.data.input_modalities = modalities
        config.data.in_channels = len(modalities)
        config.data.include_dem = "dem" in modalities
    return config


def _main_logits(output: Any) -> torch.Tensor:
    logits = BinaryThresholdSweep._main_prediction(output)
    if logits.ndim == 4 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 4 and logits.shape[1] > 1:
        return logits[:, 1]
    return logits


def _prob_to_logits(prob: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(prob.dtype).eps
    prob = prob.clamp(min=eps, max=1.0 - eps)
    return torch.log(prob / (1.0 - prob))


def _ensure_compatible_configs(configs: Sequence[Any]) -> None:
    if not configs:
        raise ValueError("At least one model config is required for ensemble evaluation")
    base = configs[0]
    keys = {
        "data.in_channels": base.data.in_channels,
        "data.include_dem": base.data.include_dem,
        "data.input_modalities": list(getattr(base.data, "input_modalities", []) or []),
        "image_size": base.image_size,
    }
    for idx, cfg in enumerate(configs[1:], start=2):
        mismatches = []
        if cfg.data.in_channels != keys["data.in_channels"]:
            mismatches.append(f"data.in_channels {cfg.data.in_channels} != {keys['data.in_channels']}")
        if cfg.data.include_dem != keys["data.include_dem"]:
            mismatches.append(f"data.include_dem {cfg.data.include_dem} != {keys['data.include_dem']}")
        if list(getattr(cfg.data, "input_modalities", []) or []) != keys["data.input_modalities"]:
            mismatches.append(
                f"data.input_modalities {list(getattr(cfg.data, 'input_modalities', []) or [])} "
                f"!= {keys['data.input_modalities']}"
            )
        if cfg.image_size != keys["image_size"]:
            mismatches.append(f"image_size {cfg.image_size} != {keys['image_size']}")
        if mismatches:
            raise ValueError(f"Config #{idx} is not input-compatible with config #1: " + "; ".join(mismatches))


def ensemble_evaluate_checkpoints(configs: Sequence[Any],
                                  checkpoint_paths: Sequence[Path],
                                  split: str = "val",
                                  thresholds: Optional[Iterable[float]] = None,
                                  threshold_metric: str = "f1",
                                  metric_mode: Optional[str] = None,
                                  include_events: Optional[Iterable[str]] = None,
                                  exclude_events: Optional[Iterable[str]] = None,
                                  ensemble_method: str = "mean_prob",
                                  inference_mode: str = "direct",
                                  window_size: int = 256,
                                  window_overlap: int = 64,
                                  window_batch_size: int = 4) -> Dict[str, float]:
    """Evaluate an ensemble by averaging per-pixel model outputs.

    The usual use case is averaging a U-Net checkpoint with a DeepLabV3+
    checkpoint trained with the same preprocessing and input modalities.
    ``mean_prob`` averages sigmoid probabilities and is the safest default for
    differently calibrated architectures. ``mean_logit`` averages raw logits.
    """
    from accelerate import Accelerator
    from floods.datasets.flood import FloodDataset, RGBFloodDataset
    from floods.normalization import describe_stats, load_normalization_stats
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.utils.ml import seed_everything, seed_worker
    from floods.eval_collate import pad_segmentation_batch

    if len(configs) != len(checkpoint_paths):
        raise ValueError(f"Number of configs ({len(configs)}) must match number of checkpoints ({len(checkpoint_paths)})")
    if not configs:
        raise ValueError("No configs supplied")
    method = str(ensemble_method or "mean_prob").lower().replace("-", "_")
    if method not in {"mean_prob", "mean_logit"}:
        raise ValueError("ensemble_method must be 'mean_prob' or 'mean_logit'")

    _ensure_compatible_configs(configs)
    config = configs[0]
    seed_everything(config.seed, deterministic=True)

    try:
        accelerator = Accelerator(mixed_precision="fp16" if config.trainer.amp else "no", cpu=config.trainer.cpu)
    except TypeError:
        accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)

    dataset, modalities, use_rgb = prepare_evaluation_dataset(config, split=split)
    _filter_dataset_by_events(dataset, include_events=include_events, exclude_events=exclude_events)
    loader = DataLoader(dataset=dataset,
                        batch_size=config.trainer.batch_size,
                        shuffle=False,
                        num_workers=config.trainer.num_workers,
                        worker_init_fn=seed_worker,
                        collate_fn=pad_segmentation_batch)

    models = []
    for idx, (cfg, ckpt_path) in enumerate(zip(configs, checkpoint_paths), start=1):
        model = prepare_model(config=cfg, num_classes=1, stage="eval")
        state = load_checkpoint_state(Path(ckpt_path))
        try:
            model.load_state_dict(state, strict=not cfg.model.multibranch)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load checkpoint #{idx} into its model. This usually means the checkpoint and config do not match. "
                f"Checkpoint: {ckpt_path} | decoder={cfg.model.decoder} | encoder={cfg.model.encoder}"
            ) from exc
        model = model.to(accelerator.device)
        model.eval()
        models.append(model)
        LOG.info("Loaded ensemble member %d/%d: decoder=%s encoder=%s checkpoint=%s",
                 idx, len(checkpoint_paths), cfg.model.decoder, cfg.model.encoder, ckpt_path)

    # Prepare models and loader together so device placement stays consistent.
    prepared = accelerator.prepare(*models, loader)
    *models, loader = prepared

    sweep = BinaryThresholdSweep(thresholds=thresholds, device=accelerator.device)
    metric_mode = str(metric_mode or getattr(config.trainer, "metric_mode", "global") or "global").lower()
    batch_average_metrics = BatchAverageMetrics(threshold=0.5, ignore_index=255, device=accelerator.device) if metric_mode in {"batch_average", "both"} else None
    inference_mode = str(inference_mode or "direct").lower().replace("-", "_")
    if inference_mode not in {"direct", "sliding_window"}:
        raise ValueError("inference_mode must be direct or sliding_window")
    if inference_mode == "sliding_window":
        LOG.info("Using sliding-window inference: window_size=%d overlap=%d window_batch_size=%d", int(window_size), int(window_overlap), int(window_batch_size))

    LOG.info("Evaluating %d-model ensemble using %s", len(models), method)
    LOG.info("Dataset: %s split, %d samples", split, len(dataset))
    with torch.no_grad():
        for x, y in progress_iter(loader, desc=f"Ensemble evaluate {split}", unit="batch", colour="green"):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            if inference_mode == "sliding_window":
                for sample_idx in range(x.shape[0]):
                    sample_x = x[sample_idx:sample_idx + 1]
                    sample_y = y[sample_idx:sample_idx + 1]
                    with accelerator.autocast():
                        ensemble_logits = ensemble_sliding_window_logits(models, sample_x, method=method, window_size=window_size, overlap=window_overlap, window_batch_size=window_batch_size)
                    y_true = accelerator.gather(sample_y)
                    y_pred = accelerator.gather(ensemble_logits)
                    sweep.update(y_true, y_pred)
                    if batch_average_metrics is not None:
                        batch_average_metrics.update(y_true, y_pred)
            else:
                member_logits = []
                member_probs = []
                with accelerator.autocast():
                    for model in models:
                        logits = _main_logits(model(x)).float()
                        if method == "mean_logit":
                            member_logits.append(logits)
                        else:
                            member_probs.append(torch.sigmoid(logits))
                if method == "mean_logit":
                    ensemble_logits = torch.stack(member_logits, dim=0).mean(dim=0)
                else:
                    mean_prob = torch.stack(member_probs, dim=0).mean(dim=0)
                    ensemble_logits = _prob_to_logits(mean_prob)
                y_true = accelerator.gather(y)
                y_pred = accelerator.gather(ensemble_logits)
                sweep.update(y_true, y_pred)
                if batch_average_metrics is not None:
                    batch_average_metrics.update(y_true, y_pred)

    LOG.info(
        "Ensemble threshold sweep (full table):\n%s",
        sweep.to_table(metric=threshold_metric),
        extra={"floodmap_file_only": True},
    )
    best = sweep.best(threshold_metric)
    LOG.info(
        "Ensemble threshold sweep best | metric=%s | threshold=%.2f | f1=%.4f | iou=%.4f | "
        "precision=%.4f | recall=%.4f | mcc=%.4f | empty_fp=%.4f | nonempty_recall=%.4f",
        threshold_metric,
        best.threshold,
        best.f1,
        best.iou,
        best.precision,
        best.recall,
        best.mcc,
        best.empty_tile_fp_rate,
        best.nonempty_tile_recall,
    )
    output = {
        "best_threshold": best.threshold,
        "best_f1": best.f1,
        "best_iou": best.iou,
        "best_mcc": best.mcc,
        "best_precision": best.precision,
        "best_recall": best.recall,
        "best_tp": best.tp,
        "best_tn": best.tn,
        "best_fp": best.fp,
        "best_fn": best.fn,
        "confusion_matrix": {
            "threshold": best.threshold,
            "tp": best.tp,
            "tn": best.tn,
            "fp": best.fp,
            "fn": best.fn,
        },
        "threshold_sweep": [
            {
                "threshold": row.threshold,
                "f1": row.f1,
                "iou": row.iou,
                "precision": row.precision,
                "recall": row.recall,
                "mcc": row.mcc,
                "empty_fp": row.empty_tile_fp_rate,
                "nonempty_recall": row.nonempty_tile_recall,
                "tp": row.tp,
                "tn": row.tn,
                "fp": row.fp,
                "fn": row.fn,
            }
            for row in sweep.compute()
        ],
    }
    if batch_average_metrics is not None:
        LOG.info(batch_average_metrics.to_line())
        nb = batch_average_metrics.compute()
        output.update({f"batch_average_{k}": v for k, v in nb.items()})
    return output
