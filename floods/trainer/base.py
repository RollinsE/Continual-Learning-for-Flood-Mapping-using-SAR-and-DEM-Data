from __future__ import annotations

import csv
import os
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from accelerate import Accelerator
from floods.logging import BaseLogger
from floods.logging.empty import EmptyLogger
from floods.evaluation import BinaryThresholdSweep, EventMacroThresholdSweep
from floods.metrics import Metric
from floods.utils.common import get_logger
from floods.utils.ml import get_rank, progressbar

if TYPE_CHECKING:
    from floods.trainer.callbacks import BaseCallback

LOG = get_logger(__name__)


class TrainerStage(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class Trainer:
    def __init__(self,
                 accelerator: Accelerator,
                 model: nn.Module,
                 optimizer: Optimizer,
                 scheduler: Any,
                 criterion: nn.Module,
                 categories: Dict[int, str],
                 train_metrics: Dict[str, Metric] = None,
                 val_metrics: Dict[str, Metric] = None,
                 logger: BaseLogger = None,
                 sample_batches: int = None,
                 stage: str = "train",
                 debug: bool = False,
                 checkpoint_dir: Optional[Path] = None,
                 resume_from: Optional[Path] = None,
                 auto_resume: bool = False,
                 save_last: bool = True,
                 save_epoch_checkpoints: bool = False,
                 extend_epochs: Optional[int] = None,
                 reset_early_stopping_on_resume: bool = False,
                 grad_clip_norm: Optional[float] = 1.0,
                 skip_nonfinite_batches: bool = True,
                 amp_full_precision_retry: bool = True,
                 max_skipped_batch_fraction: float = 0.0,
                 progress_bar: bool = True,
                 progress_log_interval: int = 0,
                 progress_label: str = "Training",
                 max_epochs: Optional[int] = None,
                 threshold_sweep: bool = False,
                 thresholds: Optional[List[float]] = None,
                 threshold_metric: str = "f1",
                 event_macro_validation: bool = False,
                 validation_event_names: Optional[List[str]] = None,
                 resume_signature: Optional[Dict[str, Any]] = None) -> None:
        self.accelerator = accelerator
        self.stage = stage
        self.debug = debug
        self.model = model
        self.criterion = criterion
        self.categories = categories or {}
        # Core optimization, scheduling, and logging components.
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger or EmptyLogger()
        # Metric collections are stored per trainer stage.
        self.metrics = dict()

        self.add_metrics(stage=TrainerStage.train, metrics=train_metrics)
        self.add_metrics(stage=TrainerStage.val, metrics=val_metrics)
        # Runtime state.
        self.rank = get_rank()
        self.is_main = self.rank == 0
        self.current_epoch = -1
        self.current_loss = None
        self.global_step = -1
        # Validation tracking and reporting state.
        self.current_scores = {TrainerStage.train.value: dict(), TrainerStage.val.value: dict()}
        self.best_epoch = None
        self.best_score = None
        self.best_state_dict = None
        self.sample_batches = sample_batches
        self.sample_content = list()
        self.callbacks: List[BaseCallback] = list()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.auto_resume = auto_resume
        self.save_last = save_last
        self.save_epoch_checkpoints = save_epoch_checkpoints
        self.extend_epochs = int(extend_epochs) if extend_epochs is not None else None
        if self.extend_epochs is not None and self.extend_epochs <= 0:
            raise ValueError("extend_epochs must be a positive integer when provided")
        self.reset_early_stopping_on_resume = bool(reset_early_stopping_on_resume)
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            self.grad_clip_norm = None
        self.skip_nonfinite_batches = bool(skip_nonfinite_batches)
        self.amp_full_precision_retry = bool(amp_full_precision_retry)
        self.amp_enabled = str(getattr(self.accelerator, "mixed_precision", "no")).lower() != "no"
        self.max_skipped_batch_fraction = float(max_skipped_batch_fraction or 0.0)
        if self.max_skipped_batch_fraction < 0.0 or self.max_skipped_batch_fraction >= 1.0:
            raise ValueError("max_skipped_batch_fraction must be 0 or a value below 1")
        self.progress_bar = bool(progress_bar)
        self.progress_log_interval = int(progress_log_interval or 0)
        if self.progress_log_interval < 0:
            self.progress_log_interval = 0
        self.progress_label = (progress_label or "Training").strip()
        self.max_epochs = int(max_epochs) if max_epochs else None
        self.threshold_sweep_enabled = bool(threshold_sweep)
        self.threshold_values = thresholds or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
        self.threshold_metric = threshold_metric if threshold_metric in {"f1", "iou", "mcc", "precision", "recall"} else "f1"
        self.event_macro_validation = bool(event_macro_validation)
        self.validation_event_names = list(validation_event_names or [])
        if self.event_macro_validation and not self.validation_event_names:
            raise ValueError("event_macro_validation requires validation_event_names")
        self.resume_signature = dict(resume_signature or {})
        self.threshold_sweep_meter: Optional[BinaryThresholdSweep] = None
        self.threshold_sweep_results = []
        self.best_threshold_result = None
        self.event_macro_sweep_meter: Optional[EventMacroThresholdSweep] = None
        self.best_event_macro_result = None
        self.skipped_batches = 0
        self.amp_overflow_batches = 0
        self.fp32_recovery_successes = 0
        self.fp32_recovery_failures = 0
        self.epoch_gradient_norms: list[float] = []
        self.start_epoch = 0
        self.last_completed_epoch = -1
        self.stop_reason: Optional[str] = None
        self.interrupted_checkpoint_path: Optional[Path] = None
        self.current_epoch_losses = {TrainerStage.train.value: {}, TrainerStage.val.value: {}}
        self.crop_requested_counts = Counter()
        self.crop_applied_counts = Counter()
        self.crop_size_counts = Counter()
        self.crop_fallbacks = 0

    def _prepare(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None) -> None:
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        train_dataloader = self.accelerator.prepare(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)
            # Accelerator wrapping can alter loader length, so sample indices are chosen here.
            if self.sample_batches is not None and self.sample_batches > 0:
                sample_count = min(int(self.sample_batches), len(val_dataloader))
                self.sample_batches = np.random.choice(len(val_dataloader), sample_count, replace=False)
            else:
                self.sample_batches = np.array([])
        return train_dataloader, val_dataloader

    def _update_metrics(self,
                        y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        stage: TrainerStage = TrainerStage.train) -> None:
        with torch.no_grad():
            for metric in self.metrics[stage.value].values():
                metric(y_true, y_pred)

    def _compute_metrics(self, stage: TrainerStage = TrainerStage.train) -> None:
        result = dict()
        with torch.no_grad():
            for name, metric in self.metrics[stage.value].items():
                result[name] = metric.compute()
        self.current_scores[stage.value] = result

    def _reset_metrics(self, stage: TrainerStage = TrainerStage.train) -> None:
        for metric in self.metrics[stage.value].values():
            metric.reset()

    def _log_metrics(self, stage: TrainerStage = TrainerStage.train, exclude: Iterable[str] = None) -> None:
        log_strings = []
        exclude = exclude or []
        scores = self.current_scores[stage.value]
        classwise = dict()
        # Log scalar metrics immediately and hold class-wise tensors for the table writer.
        for metric_name, score in scores.items():
            if metric_name in exclude:
                continue
            if score.ndim > 0:
                # Store class-wise metrics for grouped logging below.
                classwise[metric_name] = score
                continue
            scalar_score = self._scalar(score)
            self.logger.log_scalar(f"{stage.value}/{metric_name}", scalar_score)
            log_strings.append(f"{stage.value}/{metric_name}: {scalar_score:.4f}")
        # Scalar metrics are written to TensorBoard; console output stays focused on epoch summaries.
        if log_strings:
            LOG.debug(", ".join(log_strings))
        # Log class-wise results in a single table.
        if classwise:
            LOG.debug("Classwise: %s", str(classwise))
            header = list(self.categories.values())
            self.logger.log_results(f"{stage.value}/results", headers=header, results=classwise)

    def _debug_training(self, **kwargs: dict) -> None:
        LOG.debug("[Epoch %2d] - iteration: %d", self.current_epoch, self.global_step)
        for name, item in kwargs.items():
            LOG.debug("%8s: %s", name, str(item))

    def _store_samples(self, images: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        for i in range(images.size(0)):
            image = images[i].detach().cpu()
            true_mask = targets[i].detach().cpu()
            pred_mask = outputs[i].detach().cpu()
            self.sample_content.append((image, true_mask, pred_mask))

    def add_callback(self, callback: BaseCallback) -> Trainer:
        self.callbacks.append(callback)
        return self

    def setup_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.setup(self)

    def dispose_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.dispose(self)

    def add_metrics(self, stage: TrainerStage, metrics: Dict[str, Metric]) -> Trainer:
        if stage.value in self.metrics:
            raise ValueError(f"Metrics are already registered for stage: {stage.value}")
        self.metrics[stage.value] = metrics or dict()

    @staticmethod
    def _scalar(value: Any, default: float = float("nan")) -> float:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            value = value.detach().float().mean().cpu().item()
        return float(value)

    def _mean_losses(self, losses: dict) -> Dict[str, float]:
        return {name: float(np.mean(values)) for name, values in losses.items() if len(values) > 0}

    def _checkpoint_path(self) -> Optional[Path]:
        if self.resume_from is not None:
            return self.resume_from
        if self.auto_resume and self.checkpoint_dir is not None:
            candidate = self.checkpoint_dir / "last.ckpt"
            if candidate.exists():
                return candidate
        return None

    def _load_resume_checkpoint(self) -> None:
        checkpoint_path = self._checkpoint_path()
        if checkpoint_path is None:
            return
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location=self.accelerator.device)
        saved_signature = checkpoint.get("resume_signature")
        if saved_signature is not None and self.resume_signature:
            from floods.resume import diff_resume_signatures
            differences = diff_resume_signatures(saved_signature, self.resume_signature)
            if differences:
                detail = "\n  - ".join(differences[:20])
                suffix = "" if len(differences) <= 20 else f"\n  - ... and {len(differences) - 20} more"
                raise RuntimeError(
                    "Resume checkpoint training plan does not match the current run configuration. "
                    "The checkpoint will not be loaded because continuing would create a mixed or invalid experiment.\n"
                    f"  - {detail}{suffix}"
                )
        elif saved_signature is None:
            LOG.warning("Resume checkpoint has no embedded training-plan signature; using the preserved run config for compatibility.")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        model_state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        try:
            unwrapped_model.load_state_dict(model_state)
        except RuntimeError as exc:
            raise RuntimeError(
                "Resume checkpoint is incompatible with the current model architecture. "
                "This usually happens when --resume is used after changing --decoder, "
                "--encoder, or --input-modalities, or when a run folder from another model configuration is reused. "
                f"Checkpoint: {checkpoint_path}. Use --no-resume, delete the run folder/models/last.ckpt, "
                "or choose a fresh --run-id for architecture comparisons."
            ) from exc

        if checkpoint.get("optimizer_state_dict") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._restore_extra_checkpoint_state(checkpoint)

        restored_epoch = int(checkpoint.get("epoch", -1))
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        if epoch_complete:
            self.start_epoch = restored_epoch + 1
            self.last_completed_epoch = restored_epoch
        else:
            self.start_epoch = max(restored_epoch, 0)
            self.last_completed_epoch = restored_epoch - 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_epoch = checkpoint.get("best_epoch")
        self.best_score = checkpoint.get("best_score")
        self.best_state_dict = checkpoint.get("best_model_state_dict", model_state)

        early_stopping_counter = int(checkpoint.get("early_stopping_counter", 0))
        if self.reset_early_stopping_on_resume:
            early_stopping_counter = 0
        for callback in self.callbacks:
            if hasattr(callback, "patience_counter"):
                callback.patience_counter = early_stopping_counter
            if hasattr(callback, "best_epoch") and self.best_epoch is not None:
                callback.best_epoch = int(self.best_epoch)

        if self.extend_epochs is not None:
            self.max_epochs = self.start_epoch + self.extend_epochs
            LOG.info(
                "Resume extension requested: %d additional epochs from checkpoint epoch %d; target epoch is %d.",
                self.extend_epochs, restored_epoch + 1, self.max_epochs,
            )
        if self.reset_early_stopping_on_resume:
            LOG.info("Early-stopping patience counter reset after resume.")

        if epoch_complete:
            LOG.info("Resumed training from %s at epoch %d.", checkpoint_path, self.start_epoch + 1)
        else:
            LOG.info(
                "Resumed partial interrupted state from %s; restarting epoch %d rather than skipping it.",
                checkpoint_path, self.start_epoch + 1,
            )
        if self.best_score is not None:
            LOG.info("Restored best validation score: %.6f", self._scalar(self.best_score))

    def _early_stopping_counter(self) -> int:
        for callback in self.callbacks:
            if hasattr(callback, "patience_counter"):
                return int(callback.patience_counter or 0)
        return 0

    def _extra_checkpoint_state(self) -> dict:
        """Resumable numerical-stability state extended by subclasses."""
        return {
            "numerical_stability": {
                "skipped_batches": int(self.skipped_batches),
                "amp_overflow_batches": int(self.amp_overflow_batches),
                "fp32_recovery_successes": int(self.fp32_recovery_successes),
                "fp32_recovery_failures": int(self.fp32_recovery_failures),
            }
        }

    def _restore_extra_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore numerical-stability counters after core state is loaded."""
        state = (checkpoint.get("trainer_state") or {}).get("numerical_stability") or {}
        self.skipped_batches = int(state.get("skipped_batches", self.skipped_batches))
        self.amp_overflow_batches = int(state.get("amp_overflow_batches", self.amp_overflow_batches))
        self.fp32_recovery_successes = int(state.get("fp32_recovery_successes", self.fp32_recovery_successes))
        self.fp32_recovery_failures = int(state.get("fp32_recovery_failures", self.fp32_recovery_failures))

    def _save_resume_checkpoint(
        self,
        *,
        checkpoint_name: str = "last.ckpt",
        epoch_complete: bool = True,
        write_epoch_copy: bool = True,
    ) -> Optional[Path]:
        if not self.save_last or self.checkpoint_dir is None:
            return None
        # Every worker reaches the barrier; only the main process writes the file.
        self.accelerator.wait_for_everyone()
        if not self.is_main:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint = {
            "epoch": self.current_epoch,
            "epoch_complete": bool(epoch_complete),
            "global_step": self.global_step,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None and hasattr(self.scheduler, "state_dict") else None,
            "best_score": self._scalar(self.best_score) if self.best_score is not None else None,
            "best_epoch": self.best_epoch,
            "best_model_state_dict": self.best_state_dict,
            "early_stopping_counter": self._early_stopping_counter(),
            "resume_signature": self.resume_signature,
            "target_max_epochs": self.max_epochs,
            "trainer_state": self._extra_checkpoint_state(),
        }
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        temp_path = self.checkpoint_dir / f".{checkpoint_name}.tmp"
        # Write-then-replace prevents interrupted or delayed mounted-storage sync from
        # leaving a partially written checkpoint that looks valid but cannot be loaded.
        self.accelerator.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
        if write_epoch_copy and epoch_complete and self.save_epoch_checkpoints:
            epoch_path = self.checkpoint_dir / f"epoch_{self.current_epoch + 1:03d}.ckpt"
            self.accelerator.save(checkpoint, epoch_path)
        return checkpoint_path

    def _save_interrupted_checkpoint(self) -> Optional[Path]:
        """Preserve the last completed epoch and store partial work separately.

        A disconnection can occur after the model and optimizer have already been
        updated by part of the current epoch. Overwriting ``last.ckpt`` with that
        partial state used to make resume skip the remainder of the interrupted
        epoch. Keep the completed checkpoint authoritative and write the partial
        state to a separate file. If interruption happens in the first epoch, the
        partial checkpoint becomes ``last.ckpt`` but is marked incomplete so the
        same epoch restarts on resume.
        """
        if not self.save_last or self.checkpoint_dir is None:
            return None
        last_path = self.checkpoint_dir / "last.ckpt"
        if self.last_completed_epoch >= 0 and last_path.exists():
            return self._save_resume_checkpoint(
                checkpoint_name="interrupted_partial.ckpt",
                epoch_complete=False,
                write_epoch_copy=False,
            )
        return self._save_resume_checkpoint(
            checkpoint_name="last.ckpt",
            epoch_complete=False,
            write_epoch_copy=False,
        )

    @staticmethod
    def _metadata_values(value) -> list[int]:
        if value is None:
            return []
        if torch.is_tensor(value):
            return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, np.ndarray):
            return [int(v) for v in value.reshape(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return [int(value)]

    def _record_crop_usage(self, data: dict) -> None:
        requested = self._metadata_values(data.pop("_crop_requested_mode", None))
        applied = self._metadata_values(data.pop("_crop_applied_mode", None))
        sizes = self._metadata_values(data.pop("_crop_size", None))
        if not requested and not applied and not sizes:
            return
        self.crop_requested_counts.update(requested)
        self.crop_applied_counts.update(applied)
        self.crop_size_counts.update(size for size in sizes if size > 0)
        self.crop_fallbacks += sum(1 for req, actual in zip(requested, applied) if req != actual)

    def _reset_crop_usage(self) -> None:
        self.crop_requested_counts.clear()
        self.crop_applied_counts.clear()
        self.crop_size_counts.clear()
        self.crop_fallbacks = 0

    @staticmethod
    def _format_crop_mode_counts(counts: Counter) -> str:
        names = {0: "normal", 1: "flood-centred", 2: "hard-background", 3: "audit-hard-negative"}
        total = max(sum(int(v) for v in counts.values()), 1)
        modes = [mode for mode in (0, 1, 2, 3) if int(counts.get(mode, 0)) > 0 or mode == 0]
        return " ".join(
            f"{names[mode]}={int(counts.get(mode, 0))} ({100.0 * int(counts.get(mode, 0)) / total:.1f}%)"
            for mode in modes
        )

    def _log_crop_usage(self) -> None:
        if not self.crop_requested_counts and not self.crop_applied_counts:
            return
        LOG.info(
            "Training crop supervision requested: %s",
            self._format_crop_mode_counts(self.crop_requested_counts),
        )
        LOG.info(
            "Training crop supervision applied: %s | fallbacks to full tile: %d",
            self._format_crop_mode_counts(self.crop_applied_counts),
            int(self.crop_fallbacks),
        )
        if self.crop_size_counts:
            total = max(sum(int(v) for v in self.crop_size_counts.values()), 1)
            sizes = " ".join(
                f"{size}={int(count)} ({100.0 * int(count) / total:.1f}%)"
                for size, count in sorted(self.crop_size_counts.items())
            )
            LOG.info("Training crop sizes before resize: %s", sizes)

    def _epoch_summary(self) -> None:
        train_loss = self.current_epoch_losses.get(TrainerStage.train.value, {}).get("loss")
        val_loss = self.current_epoch_losses.get(TrainerStage.val.value, {}).get("loss")
        val_scores = self.current_scores.get(TrainerStage.val.value, {})
        fields = []
        if train_loss is not None:
            fields.append(f"train loss: {train_loss:.4f}")
        if val_loss is not None:
            fields.append(f"validation loss: {val_loss:.4f}")
        if self.best_threshold_result is not None:
            # Report one authoritative validation operating point: the same
            # threshold-swept result used for checkpoint selection. This avoids
            # mixing fixed-0.5 metrics with the monitored best-threshold score.
            best = self.best_threshold_result
            for metric_name in ("f1", "iou", "precision", "recall", "mcc"):
                fields.append(f"validation {metric_name}: {getattr(best, metric_name):.4f}")
            fields.append(f"selection threshold: {best.threshold:.2f}")
            fields.append(f"selection {self.threshold_metric}: {getattr(best, self.threshold_metric):.4f}")
        else:
            for metric_name in ("f1", "iou", "precision", "recall", "mcc"):
                if metric_name in val_scores:
                    fields.append(f"validation {metric_name}: {self._scalar(val_scores[metric_name]):.4f}")
        if self.best_event_macro_result is not None:
            event_best = self.best_event_macro_result
            fields.append(f"event-macro f1: {event_best.macro_f1:.4f}")
            fields.append(f"worst-event f1: {event_best.worst_f1:.4f}")
            fields.append(f"event threshold: {event_best.threshold:.2f}")
        if self.amp_enabled:
            fields.append(f"AMP overflows cumulative: {self.amp_overflow_batches}")
            fields.append(f"float32 recoveries cumulative: {self.fp32_recovery_successes}")
            fields.append(f"failed recoveries cumulative: {self.fp32_recovery_failures}")
        if self.skipped_batches:
            fields.append(f"skipped batches cumulative: {self.skipped_batches}")
        if self.epoch_gradient_norms:
            fields.append(
                "gradient norm min/mean/max: "
                f"{min(self.epoch_gradient_norms):.4f}/"
                f"{float(np.mean(self.epoch_gradient_norms)):.4f}/"
                f"{max(self.epoch_gradient_norms):.4f}"
            )
        if fields:
            LOG.info("Epoch %d complete | %s", self.current_epoch + 1, " | ".join(fields))

    def step(self) -> None:
        self.global_step += 1
        self.logger.step()

    def precision_context(self, *, full_precision: bool = False):
        """Return the configured autocast context or a forced float32 context."""
        return nullcontext() if full_precision else self.accelerator.autocast()

    def _gradients_are_finite(self) -> bool:
        return all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in self.model.parameters()
        )

    def _prepare_gradient_step(self) -> float:
        """Unscale, measure, optionally clip, and return the pre-clip norm."""
        if self.grad_clip_norm is not None:
            norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        else:
            unscale = getattr(self.accelerator, "unscale_gradients", None)
            if callable(unscale):
                unscale(self.optimizer)
            grads = [p.grad.detach().float().norm(2) for p in self.model.parameters() if p.grad is not None]
            norm = torch.stack(grads).norm(2) if grads else torch.tensor(0.0, device=self.accelerator.device)
        value = float(norm.detach().cpu()) if torch.is_tensor(norm) else float(norm)
        self.epoch_gradient_norms.append(value)
        return value

    def train_epoch_start(self):
        self._reset_metrics(stage=TrainerStage.train)
        self._reset_crop_usage()
        self.epoch_gradient_norms = []

    def train_batch(
        self,
        batch: Any,
        *,
        full_precision: bool = False,
        record_state: bool = True,
    ) -> torch.Tensor:
        del full_precision, record_state
        raise NotImplementedError("Implement in subclass")

    def train_epoch(self, epoch: int, train_dataloader: DataLoader) -> Any:
        timings = []
        losses = defaultdict(list)
        total_batches = len(train_dataloader) if hasattr(train_dataloader, "__len__") else None
        epoch_skipped_start = int(self.skipped_batches)

        def enforce_nonfinite_budget() -> None:
            if not self.max_skipped_batch_fraction or total_batches is None:
                return
            skipped_this_epoch = int(self.skipped_batches) - epoch_skipped_start
            allowed = max(1, int(np.floor(total_batches * self.max_skipped_batch_fraction)))
            if skipped_this_epoch > allowed:
                raise FloatingPointError(
                    f"Non-finite batch budget exceeded in epoch {epoch + 1}: "
                    f"skipped={skipped_this_epoch}, allowed={allowed}, total_batches={total_batches}, "
                    f"fraction_limit={self.max_skipped_batch_fraction:.4f}."
                )
        if self.is_main and not self.progress_bar:
            if total_batches is not None:
                LOG.info("Epoch %d training started | batches=%d", epoch + 1, total_batches)
            else:
                LOG.info("Epoch %d training started", epoch + 1)
        train_tqdm = progressbar(train_dataloader,
                                 epoch=epoch,
                                 stage=TrainerStage.train.value,
                                 disable=(not self.is_main) or (not self.progress_bar),
                                 total_epochs=self.max_epochs,
                                 label=self.progress_label)

        self.model.train()
        running_loss = []
        for batch_index, batch in enumerate(train_tqdm, start=1):
            start = time.time()
            self.optimizer.zero_grad()
            loss, data = self.train_batch(batch=batch)
            self._record_crop_usage(data)
            if not torch.isfinite(loss.detach()).all():
                self.skipped_batches += 1
                message = f"Skipping non-finite training loss at epoch {epoch + 1}, step {self.global_step + 1}."
                if self.skip_nonfinite_batches:
                    LOG.warning(message)
                    self.optimizer.zero_grad(set_to_none=True)
                    enforce_nonfinite_budget()
                    continue
                raise FloatingPointError(message)
            # Backpropagation and optimizer update. AMP overflow batches are
            # retried once with autocast disabled rather than silently dropped.
            recovered_in_float32 = False
            backward_failed = False
            try:
                self.accelerator.backward(loss)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "nan" in message or "inf" in message or "non-finite" in message:
                    backward_failed = True
                    LOG.warning(
                        "Non-finite backward pass detected | epoch=%d | batch=%d | step=%d | error=%s",
                        epoch + 1, batch_index, self.global_step + 1, str(exc).splitlines()[0],
                    )
                else:
                    raise

            gradients_are_finite = (not backward_failed) and self._gradients_are_finite()
            should_retry = (
                self.amp_enabled
                and self.amp_full_precision_retry
                and (backward_failed or not gradients_are_finite)
            )
            if should_retry:
                self.amp_overflow_batches += 1
                LOG.warning(
                    "AMP overflow detected | epoch=%d | batch=%d | step=%d | retry=float32",
                    epoch + 1, batch_index, self.global_step + 1,
                )
                self.optimizer.zero_grad(set_to_none=True)
                retry_loss, retry_data = self.train_batch(
                    batch=batch,
                    full_precision=True,
                    record_state=False,
                )
                retry_ok = bool(torch.isfinite(retry_loss.detach()).all())
                if retry_ok:
                    try:
                        self.accelerator.backward(retry_loss)
                    except RuntimeError as exc:
                        retry_ok = False
                        LOG.warning(
                            "Float32 retry backward failed | epoch=%d | batch=%d | step=%d | error=%s",
                            epoch + 1, batch_index, self.global_step + 1, str(exc).splitlines()[0],
                        )
                retry_ok = retry_ok and self._gradients_are_finite()
                if retry_ok:
                    self.fp32_recovery_successes += 1
                    recovered_in_float32 = True
                    loss, data = retry_loss, retry_data
                    LOG.info(
                        "Float32 recovery succeeded | epoch=%d | batch=%d | step=%d | recovered=%d",
                        epoch + 1, batch_index, self.global_step + 1, self.fp32_recovery_successes,
                    )
                else:
                    self.fp32_recovery_failures += 1
                    self.skipped_batches += 1
                    LOG.warning(
                        "Float32 recovery failed; batch skipped | epoch=%d | batch=%d | step=%d | "
                        "failed_recoveries=%d | skipped=%d",
                        epoch + 1, batch_index, self.global_step + 1,
                        self.fp32_recovery_failures, self.skipped_batches,
                    )
                    # Let Accelerate/GradScaler observe the overflow and reduce
                    # its scale. PyTorch guarantees that the wrapped step is
                    # skipped when gradients are non-finite.
                    if getattr(self.accelerator, "scaler", None) is not None and not self._gradients_are_finite():
                        try:
                            self.optimizer.step()
                        except Exception:
                            pass
                    self.optimizer.zero_grad(set_to_none=True)
                    enforce_nonfinite_budget()
                    continue
            elif backward_failed or not gradients_are_finite:
                self.skipped_batches += 1
                message = f"Non-finite gradients at epoch {epoch + 1}, step {self.global_step + 1}."
                if self.skip_nonfinite_batches:
                    LOG.warning(
                        "Skipping batch with non-finite gradients | epoch=%d | batch=%d | step=%d | skipped=%d",
                        epoch + 1, batch_index, self.global_step + 1, self.skipped_batches,
                    )
                    if getattr(self.accelerator, "scaler", None) is not None:
                        try:
                            self.optimizer.step()
                        except Exception:
                            pass
                    self.optimizer.zero_grad(set_to_none=True)
                    enforce_nonfinite_budget()
                    continue
                raise FloatingPointError(message)

            gradient_norm = self._prepare_gradient_step()
            self.optimizer.step()
            if recovered_in_float32:
                self.logger.log_scalar("train/fp32_recovery", 1.0)
            self.logger.log_scalar("train/gradient_norm", gradient_norm)
            # Measure elapsed time for throughput monitoring.
            elapsed = (time.time() - start)
            # Store training state and scalar logs.
            self.current_loss = loss.mean()
            loss_val = loss.mean().item()
            running_loss.append(loss_val)
            lr_value = self.optimizer.param_groups[0]["lr"]
            train_tqdm.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{lr_value:.2e}"})
            self.logger.log_scalar("train/loss_iter", loss_val)
            self.logger.log_scalar("train/lr", self.optimizer.param_groups[0]["lr"])
            self.logger.log_scalar("train/time_iter", elapsed)
            # Accumulate batch losses for the epoch summary.
            for name, val in data.items():
                losses[name].append(val.mean().item())
            timings.append(elapsed)
            # Emit compact heartbeat logs when tqdm progress bars are hidden.
            if (self.is_main and not self.progress_bar and self.progress_log_interval
                    and (batch_index % self.progress_log_interval == 0)):
                avg_loss = float(np.mean(running_loss[-self.progress_log_interval:]))
                if total_batches is not None:
                    LOG.info("Epoch %d training progress | batch=%d/%d | recent_loss=%.4f",
                             epoch + 1, batch_index, total_batches, avg_loss)
                else:
                    LOG.info("Epoch %d training progress | batches=%d | recent_loss=%.4f",
                             epoch + 1, batch_index, avg_loss)
            # Advance the logger step after each training batch.
            self.step()
        return losses, timings

    def train_epoch_end(self, train_losses: dict, train_times: list):
        with torch.no_grad():
            self._compute_metrics(stage=TrainerStage.train)
        self.current_epoch_losses[TrainerStage.train.value] = self._mean_losses(train_losses)
        for name, value in self.current_epoch_losses[TrainerStage.train.value].items():
            self.logger.log_scalar(f"train/{name}", value)
        self.logger.log_scalar("train/time", np.mean(train_times))
        self._log_metrics(stage=TrainerStage.train)
        self._log_crop_usage()

    def validation_epoch_start(self):
        self.sample_content.clear()
        self._reset_metrics(stage=TrainerStage.val)
        self._reset_threshold_sweep()
        self._reset_event_macro_sweep()

    def validation_batch(self, batch: Any, batch_index: int):
        raise NotImplementedError("Implement in subclass")

    def validation_epoch(self, epoch: int, val_dataloader: DataLoader) -> Any:
        total_batches = len(val_dataloader) if hasattr(val_dataloader, "__len__") else None
        if self.is_main and not self.progress_bar:
            if total_batches is not None:
                LOG.info("Epoch %d validation started | batches=%d", epoch + 1, total_batches)
            else:
                LOG.info("Epoch %d validation started", epoch + 1)
        val_tqdm = progressbar(val_dataloader,
                               epoch=epoch,
                               stage=TrainerStage.val.value,
                               disable=(not self.is_main) or (not self.progress_bar),
                               total_epochs=self.max_epochs,
                               label=self.progress_label)
        timings = []
        losses = defaultdict(list)
        running_loss = []

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(val_tqdm, start=1):
                start = time.time()
                loss, data = self.validation_batch(batch=batch, batch_index=i - 1)
                elapsed = (time.time() - start)
                # Gather validation loss values.
                loss_val = loss.mean().item()
                running_loss.append(loss_val)
                val_tqdm.set_postfix({"loss": f"{loss_val:.4f}"})
                # Validation logs are emitted at epoch level because the logger step is tied to training batches.
                for name, val in data.items():
                    losses[name].append(val.mean().item())
                timings.append(elapsed)
                if (self.is_main and not self.progress_bar and self.progress_log_interval
                        and total_batches is not None and total_batches >= self.progress_log_interval
                        and i % self.progress_log_interval == 0):
                    avg_loss = float(np.mean(running_loss[-self.progress_log_interval:]))
                    LOG.info("Epoch %d validation progress | batch=%d/%d | recent_loss=%.4f",
                             epoch + 1, i, total_batches, avg_loss)
        return losses, timings

    def validation_epoch_end(self, val_losses: list, val_times: list):
        with torch.no_grad():
            self._compute_metrics(stage=TrainerStage.val)
            self._finalise_threshold_sweep()
            if self.event_macro_validation:
                self._finalise_event_macro_sweep()
        self.current_epoch_losses[TrainerStage.val.value] = self._mean_losses(val_losses)
        for name, value in self.current_epoch_losses[TrainerStage.val.value].items():
            self.logger.log_scalar(f"val/{name}", value)
        self.logger.log_scalar("val/time", np.mean(val_times))
        self._log_metrics(stage=TrainerStage.val)


    def _reset_threshold_sweep(self) -> None:
        if self.threshold_sweep_enabled:
            self.threshold_sweep_meter = BinaryThresholdSweep(thresholds=self.threshold_values, device=self.accelerator.device)
            self.threshold_sweep_results = []
            self.best_threshold_result = None
        else:
            self.threshold_sweep_meter = None
            self.threshold_sweep_results = []
            self.best_threshold_result = None

    def _update_threshold_sweep(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        if self.threshold_sweep_meter is not None:
            self.threshold_sweep_meter.update(y_true, y_pred)

    def _finalise_threshold_sweep(self) -> None:
        if self.threshold_sweep_meter is None:
            return
        self.threshold_sweep_results = self.threshold_sweep_meter.compute()
        self.best_threshold_result = self.threshold_sweep_meter.best(self.threshold_metric)
        val_scores = self.current_scores.setdefault(TrainerStage.val.value, {})
        val_scores[f"best_{self.threshold_metric}"] = torch.tensor(getattr(self.best_threshold_result, self.threshold_metric), device=self.accelerator.device)
        for key in ("f1", "iou", "mcc", "precision", "recall"):
            val_scores[f"best_threshold_{key}"] = torch.tensor(getattr(self.best_threshold_result, key), device=self.accelerator.device)
        val_scores["best_threshold"] = torch.tensor(self.best_threshold_result.threshold, device=self.accelerator.device)
        self.logger.log_scalar("val/best_threshold", self.best_threshold_result.threshold)
        self.logger.log_scalar(f"val/best_threshold_{self.threshold_metric}", getattr(self.best_threshold_result, self.threshold_metric))
        LOG.info(
            "Validation threshold sweep (full table):\n%s",
            self.threshold_sweep_meter.to_table(metric=self.threshold_metric),
            extra={"floodmap_file_only": True},
        )
        best = self.best_threshold_result
        LOG.info(
            "Validation threshold sweep best | metric=%s | threshold=%.2f | f1=%.4f | iou=%.4f | "
            "precision=%.4f | recall=%.4f | mcc=%.4f | empty_fp=%.4f | nonempty_recall=%.4f",
            self.threshold_metric,
            best.threshold,
            best.f1,
            best.iou,
            best.precision,
            best.recall,
            best.mcc,
            best.empty_tile_fp_rate,
            best.nonempty_tile_recall,
        )

    def _step_scheduler(self) -> None:
        if self.scheduler is None:
            return
        try:
            self.scheduler.step()
        except TypeError:
            val_scores = self.current_scores.get(TrainerStage.val.value, {})
            monitor_value = None
            if val_scores:
                monitor_value = self._scalar(next(iter(val_scores.values())))
            elif self.current_loss is not None:
                monitor_value = self._scalar(self.current_loss)
            self.scheduler.step(monitor_value)

    def _reset_event_macro_sweep(self) -> None:
        if self.event_macro_validation:
            self.event_macro_sweep_meter = EventMacroThresholdSweep(
                event_names=self.validation_event_names,
                thresholds=self.threshold_values,
                device=self.accelerator.device,
            )
            self.best_event_macro_result = None
        else:
            self.event_macro_sweep_meter = None

    def _update_event_macro_sweep(self, y_true: torch.Tensor, y_pred: torch.Tensor, event_indices: Optional[torch.Tensor]) -> None:
        if self.event_macro_sweep_meter is not None:
            if event_indices is None:
                raise ValueError("Event-macro validation batch is missing event indices")
            self.event_macro_sweep_meter.update(y_true, y_pred, event_indices)

    def _finalise_event_macro_sweep(self) -> None:
        if self.event_macro_sweep_meter is None:
            return
        best = self.event_macro_sweep_meter.best()
        self.best_event_macro_result = best
        scores = self.current_scores.setdefault(TrainerStage.val.value, {})
        device = self.accelerator.device
        scores["best_event_macro_f1"] = torch.tensor(best.macro_f1, dtype=torch.float32, device=device)
        scores["best_event_macro_iou"] = torch.tensor(best.macro_iou, dtype=torch.float32, device=device)
        scores["best_event_worst_f1"] = torch.tensor(best.worst_f1, dtype=torch.float32, device=device)
        scores["best_event_threshold"] = torch.tensor(best.threshold, dtype=torch.float32, device=device)
        LOG.info(
            "Event-macro validation best | threshold=%.2f | macro_f1=%.4f | macro_iou=%.4f | "
            "worst_event_f1=%.4f | mean_precision=%.4f | mean_recall=%.4f",
            best.threshold, best.macro_f1, best.macro_iou, best.worst_f1,
            best.mean_precision, best.mean_recall,
        )
        if self.is_main and self.checkpoint_dir is not None:
            history_path = self.checkpoint_dir.parent / "event_validation_history.csv"
            write_header = not history_path.exists()
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "epoch", "threshold", "macro_f1", "macro_iou", "worst_event_f1",
                    "mean_precision", "mean_recall",
                ])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "epoch": self.current_epoch + 1,
                    "threshold": best.threshold,
                    "macro_f1": best.macro_f1,
                    "macro_iou": best.macro_iou,
                    "worst_event_f1": best.worst_f1,
                    "mean_precision": best.mean_precision,
                    "mean_recall": best.mean_recall,
                })
            event_path = self.checkpoint_dir.parent / "event_validation_metrics.csv"
            write_event_header = not event_path.exists()
            with event_path.open("a", newline="", encoding="utf-8") as handle:
                fieldnames = ["epoch", "event_id", "threshold", "f1", "iou", "precision", "recall", "tp", "fp", "fn"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_event_header:
                    writer.writeheader()
                for row in best.event_metrics:
                    writer.writerow({"epoch": self.current_epoch + 1, **row})

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None, max_epochs: int = 100):
        self.max_epochs = int(max_epochs)
        train_dataloader, val_dataloader = self._prepare(train_dataloader, val_dataloader)
        self.best_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        self.setup_callbacks()
        self.global_step = 0
        self._load_resume_checkpoint()

        if self.start_epoch >= self.max_epochs:
            LOG.info(
                "Resume checkpoint is already at or beyond the requested target epoch: checkpoint epoch %d, target epoch %d. Nothing to train.",
                self.start_epoch, self.max_epochs,
            )
            self.stop_reason = self.stop_reason or "completed"
            return self

        for curr_epoch in range(self.start_epoch, self.max_epochs):
            self.current_epoch = curr_epoch
            try:
                self.train_epoch_start()
                t_losses, t_times = self.train_epoch(epoch=self.current_epoch, train_dataloader=train_dataloader)
                self.train_epoch_end(t_losses, t_times)

                if val_dataloader is not None:
                    self.validation_epoch_start()
                    v_losses, v_times = self.validation_epoch(epoch=self.current_epoch, val_dataloader=val_dataloader)
                    self.validation_epoch_end(v_losses, v_times)

                self._step_scheduler()
                self._epoch_summary()
                self.last_completed_epoch = self.current_epoch

                for callback in self.callbacks:
                    callback(self)

                self._save_resume_checkpoint()

            except KeyboardInterrupt:
                if self.stop_reason is None:
                    self.stop_reason = "interrupted"
                self.interrupted_checkpoint_path = self._save_interrupted_checkpoint()
                break
        else:
            self.stop_reason = self.stop_reason or "completed"

        self.dispose_callbacks()
        return self

    def test_batch(self, batch: Any, batch_index: int):
        x, y = batch
        # Forward pass and loss computation with the configured precision policy.
        with self.accelerator.autocast():
            preds = self.model(x)
            loss = self.criterion(preds, y)
        # Gather predictions and targets across processes.
        images = self.accelerator.gather(x)
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(preds)
        # Store sample predictions when visualization callbacks are active.
        if self.sample_batches is not None and batch_index in self.sample_batches:
            self._store_samples(images, y_pred, y_true)
        # Update metrics and return batch outputs.
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.test)
        result_data = {"inputs": images.cpu(), "targets": y_true.cpu(), "preds": torch.argmax(y_pred, dim=1).cpu()}
        return loss, result_data

    def predict(self,
                test_dataloader: DataLoader,
                metrics: Dict[str, Metric],
                logger_exclude: Iterable[str] = None,
                return_predictions: bool = False,
                **kwargs: dict):
        logger_exclude = logger_exclude or []
        self.metrics[TrainerStage.test.value] = metrics
        self._reset_metrics(stage=TrainerStage.test)
        test_tqdm = progressbar(test_dataloader, stage=TrainerStage.test.value, disable=not self.is_main)
        losses, timings, results = [], [], []
        # Prepare the model and loader, reusing validation bookkeeping for sample counts.
        _, test_dataloader = self._prepare(train_dataloader=None, val_dataloader=test_dataloader)

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(test_tqdm):
                start = time.time()
                loss, data = self.test_batch(batch=batch, batch_index=i, **kwargs)
                elapsed = (time.time() - start)
                loss_value = loss.item()
                test_tqdm.set_postfix({"loss": f"{loss_value:.4f}"})
                # Test logs are emitted at dataset level, matching validation behaviour.
                losses.append(loss_value)
                timings.append(elapsed)
                if return_predictions:
                    results.append(data)

            self.logger.log_scalar("test/loss", np.mean(losses))
            self.logger.log_scalar("test/time", np.mean(timings))
            self._compute_metrics(stage=TrainerStage.test)
            self._log_metrics(stage=TrainerStage.test, exclude=logger_exclude)
        # Execute test callbacks such as sample visualization.
        for callback in self.callbacks:
            callback(self)
        return losses, results
