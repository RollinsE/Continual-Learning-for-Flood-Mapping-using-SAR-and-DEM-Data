from pathlib import Path
from typing import Any, Callable, Dict

import torch
from torch import nn
from torch.optim import Optimizer

from accelerate import Accelerator
from floods.logging import BaseLogger
from floods.metrics import Metric
from floods.trainer import Trainer, TrainerStage
from floods.group_dro import (
    append_event_weight_rows, effective_group_count, group_mean_losses,
    robust_present_group_loss, update_group_weights,
)
from floods.utils.common import get_logger

LOG = get_logger(__name__)

def _unpack_training_batch(batch: Any):
    """Return inputs, targets, optional crop metadata, and optional event groups."""
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Training batch must be a tuple/list, got {type(batch)!r}")
    if len(batch) == 2:
        x, y = batch
        return x, y, None, None
    if len(batch) == 3:
        x, y, group_indices = batch
        return x, y, None, group_indices
    if len(batch) == 5:
        x, y, requested_mode, applied_mode, crop_size = batch
        return x, y, {
            "_crop_requested_mode": requested_mode,
            "_crop_applied_mode": applied_mode,
            "_crop_size": crop_size,
        }, None
    if len(batch) == 6:
        x, y, requested_mode, applied_mode, crop_size, group_indices = batch
        return x, y, {
            "_crop_requested_mode": requested_mode,
            "_crop_applied_mode": applied_mode,
            "_crop_size": crop_size,
        }, group_indices
    raise ValueError(f"Unsupported training batch structure with {len(batch)} items")



def _unpack_validation_batch(batch: Any):
    """Return validation inputs, targets, and optional event indices."""
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Validation batch must be a tuple/list, got {type(batch)!r}")
    if len(batch) == 2:
        x, y = batch
        return x, y, None
    if len(batch) == 3:
        x, y, event_indices = batch
        return x, y, event_indices
    raise ValueError(f"Unsupported validation batch structure with {len(batch)} items")


def _gather_crop_metadata(accelerator: Accelerator, metadata):
    if not metadata:
        return {}
    return {name: accelerator.gather(value) for name, value in metadata.items()}


class FloodTrainer(Trainer):
    def __init__(self,
                 accelerator: Accelerator,
                 model: nn.Module,
                 criterion: nn.Module,
                 categories: Dict[int, str],
                 optimizer: Optimizer = None,
                 scheduler: Any = None,
                 tiler: Callable = None,
                 train_metrics: Dict[str, Metric] = None,
                 val_metrics: Dict[str, Metric] = None,
                 logger: BaseLogger = None,
                 sample_batches: int = None,
                 stage: str = "train",
                 debug: bool = False,
                 checkpoint_dir: Path = None,
                 resume_from: Path = None,
                 auto_resume: bool = False,
                 save_last: bool = True,
                 save_epoch_checkpoints: bool = False,
                 extend_epochs: int = None,
                 reset_early_stopping_on_resume: bool = False,
                 grad_clip_norm: float = 1.0,
                 skip_nonfinite_batches: bool = True,
                 amp_full_precision_retry: bool = True,
                 max_skipped_batch_fraction: float = 0.0,
                 progress_bar: bool = True,
                 progress_log_interval: int = 0,
                 progress_label: str = "Training",
                 max_epochs: int = None,
                 threshold_sweep: bool = False,
                 thresholds: list[float] = None,
                 threshold_metric: str = "f1",
                 event_macro_validation: bool = False,
                 validation_event_names: list[str] = None,
                 resume_signature: Dict[str, Any] = None) -> None:
        super().__init__(accelerator,
                         model,
                         optimizer,
                         scheduler,
                         criterion,
                         categories,
                         train_metrics=train_metrics,
                         val_metrics=val_metrics,
                         logger=logger,
                         sample_batches=sample_batches,
                         stage=stage,
                         debug=debug,
                         checkpoint_dir=checkpoint_dir,
                         resume_from=resume_from,
                         auto_resume=auto_resume,
                         save_last=save_last,
                         save_epoch_checkpoints=save_epoch_checkpoints,
                         extend_epochs=extend_epochs,
                         reset_early_stopping_on_resume=reset_early_stopping_on_resume,
                         grad_clip_norm=grad_clip_norm,
                         skip_nonfinite_batches=skip_nonfinite_batches,
                         amp_full_precision_retry=amp_full_precision_retry,
                         max_skipped_batch_fraction=max_skipped_batch_fraction,
                         progress_bar=progress_bar,
                         progress_log_interval=progress_log_interval,
                         progress_label=progress_label,
                         max_epochs=max_epochs,
                         threshold_sweep=threshold_sweep,
                         thresholds=thresholds,
                         threshold_metric=threshold_metric,
                         event_macro_validation=event_macro_validation,
                         validation_event_names=validation_event_names,
                         resume_signature=resume_signature)
        self.tiler = tiler

    def train_batch(
        self,
        batch: Any,
        *,
        full_precision: bool = False,
        record_state: bool = True,
    ) -> torch.Tensor:
        x, y, crop_metadata, _ = _unpack_training_batch(batch)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        with self.precision_context(full_precision=full_precision):
            out = self.model(x)
            loss = self.criterion(out, y.long())
        if record_state:
            y_true = self.accelerator.gather(y)
            y_pred = self.accelerator.gather(out)
            self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        if self.debug:
            self._debug_training(x=x.dtype, y=y.dtype, pred=out.dtype, loss=loss)
        data = {"loss": loss.detach()}
        if record_state:
            data.update(_gather_crop_metadata(self.accelerator, crop_metadata))
        return loss, data

    def validation_batch(self, batch: Any, batch_index: int):
        # Retrieve inputs, targets, and optional event indices.
        x, y, event_indices = _unpack_validation_batch(batch)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        # Forward pass and loss computation with the configured precision policy.
        with self.accelerator.autocast():
            out = self.model(x)
            loss = self.criterion(out, y.long())
        # Gather predictions and targets across processes.
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(out)
        gathered_event_indices = self.accelerator.gather(event_indices) if event_indices is not None else None
        # Store one representative sample from selected batches for visualization.
        if self.sample_batches is not None and batch_index in self.sample_batches:
            preds = (torch.sigmoid(y_pred) > 0.5).int()
            images = self.accelerator.gather(x)
            self._store_samples(images[:1], preds[:1], y_true[:1].int())
        # Update metrics and return validation loss.
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.val)
        self._update_threshold_sweep(y_true=y_true, y_pred=y_pred)
        self._update_event_macro_sweep(y_true=y_true, y_pred=y_pred, event_indices=gathered_event_indices)
        return loss, {"loss": loss.detach()}

    def test_batch(self, batch: Any, batch_index: int, output_path: Path = None):
        # Inputs and masks are full-size rasters with matching height and width.
        x, y = batch
        if x.shape[0] != 1:
            raise ValueError("Full-scene testing requires batch size 1")
        x = x.to(device=self.accelerator.device)
        y = y.to(device=self.accelerator.device)

        # Build the sliding-window prediction callback.
        def callback(patches: torch.Tensor) -> torch.Tensor:
            patch_preds = self.model(patches)
            return patch_preds

        y_pred = self.tiler(x[0], callback)
        y_pred = y_pred.unsqueeze(0)  # .permute(2, 0, 1)
        loss = self.criterion(y_pred, y.long())
        # Variable image dimensions prevent direct distributed gathering.
        # Store representative samples when visualisation callbacks are active.
        if output_path:
            if self.sample_batches is None or batch_index in self.sample_batches:
                self._store_samples(x, (torch.sigmoid(y_pred) > 0.5).int(), y.int())
                self.callbacks[0](self, filepath=output_path, filename=f"{batch_index:06d}-0")
                self.sample_content = []
        # Update metrics and return losses.
        self._update_metrics(y_true=y, y_pred=y_pred, stage=TrainerStage.test)
        return loss, {"loss": loss.detach()}


class MultiBranchTrainer(FloodTrainer):
    def train_batch(
        self,
        batch: Any,
        *,
        full_precision: bool = False,
        record_state: bool = True,
    ) -> torch.Tensor:
        x, y, crop_metadata, _ = _unpack_training_batch(batch)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        with self.precision_context(full_precision=full_precision):
            out, aux = self.model(x)
            loss = self.criterion(out, y.long())
            loss += self.criterion(aux, y.long()) * 0.4
        if record_state:
            y_true = self.accelerator.gather(y)
            y_pred = self.accelerator.gather(out)
            self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        if self.debug:
            self._debug_training(x=x.dtype, y=y.dtype, pred=out.dtype, loss=loss)
        data = {"loss": loss.detach()}
        if record_state:
            data.update(_gather_crop_metadata(self.accelerator, crop_metadata))
        return loss, data

    def validation_batch(self, batch: Any, batch_index: int):
        # Retrieve inputs, targets, and optional event indices.
        x, y, event_indices = _unpack_validation_batch(batch)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        # Compute the primary-task forward pass and loss under the configured precision policy.
        with self.accelerator.autocast():
            out, _ = self.model(x)
            loss = self.criterion(out, y.long())
        # Gather predictions and targets across processes.
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(out)
        gathered_event_indices = self.accelerator.gather(event_indices) if event_indices is not None else None
        # Store one representative sample from selected batches for visualization.
        if self.sample_batches is not None and batch_index in self.sample_batches:
            preds = (torch.sigmoid(y_pred) > 0.5).int()
            images = self.accelerator.gather(x)
            self._store_samples(images[:1], preds[:1], y_true[:1])
        # Update metrics and return validation loss.
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.val)
        self._update_threshold_sweep(y_true=y_true, y_pred=y_pred)
        self._update_event_macro_sweep(y_true=y_true, y_pred=y_pred, event_indices=gathered_event_indices)
        return loss, {"loss": loss.detach()}


class _GroupDROMixin:
    """Event-level GroupDRO objective shared by single- and multi-branch trainers."""

    def _init_group_dro(
        self,
        *,
        event_names,
        eta: float,
        min_weight: float,
        warmup_epochs: int,
    ) -> None:
        self.group_dro_event_names = list(event_names)
        if len(self.group_dro_event_names) < 2:
            raise ValueError("GroupDRO requires at least two training events")
        self.group_dro_eta = float(eta)
        self.group_dro_min_weight = float(min_weight)
        self.group_dro_warmup_epochs = int(warmup_epochs)
        if self.group_dro_eta < 0:
            raise ValueError("group_dro_eta must be non-negative")
        if self.group_dro_min_weight < 0:
            raise ValueError("group_dro_min_weight must be non-negative")
        if self.group_dro_warmup_epochs < 0:
            raise ValueError("group_dro_warmup_epochs must be non-negative")
        count = len(self.group_dro_event_names)
        if self.group_dro_min_weight * count >= 1.0:
            raise ValueError("group_dro_min_weight is too large for the number of events")
        self.group_dro_weights = torch.full(
            (count,), 1.0 / count, dtype=torch.float32, device=self.accelerator.device
        )
        self.group_dro_epoch_loss_sum = torch.zeros_like(self.group_dro_weights)
        self.group_dro_epoch_observations = torch.zeros_like(self.group_dro_weights)
        self.group_dro_history_path = (
            self.checkpoint_dir.parent / "group_dro_event_weights.csv"
            if self.checkpoint_dir is not None else None
        )

    def train_epoch_start(self):
        super().train_epoch_start()
        self.group_dro_epoch_loss_sum.zero_()
        self.group_dro_epoch_observations.zero_()

    def _group_dro_loss(
        self,
        per_sample_losses: torch.Tensor,
        group_indices: torch.Tensor,
        *,
        update_state: bool = True,
    ):
        if group_indices is None:
            raise ValueError("GroupDRO training batch is missing event group indices")
        group_indices = group_indices.reshape(-1).long().to(per_sample_losses.device)
        local_means, local_counts, local_present = group_mean_losses(
            per_sample_losses, group_indices, len(self.group_dro_event_names)
        )

        gathered_losses = self.accelerator.gather(per_sample_losses.detach().float())
        gathered_groups = self.accelerator.gather(group_indices.detach())
        observed_means, observed_counts, observed_present = group_mean_losses(
            gathered_losses, gathered_groups, len(self.group_dro_event_names)
        )

        if update_state:
            self.group_dro_epoch_loss_sum += observed_means.to(self.group_dro_epoch_loss_sum.device) * observed_counts.to(self.group_dro_epoch_loss_sum.device)
            self.group_dro_epoch_observations += observed_counts.to(self.group_dro_epoch_observations.device)

        active = self.current_epoch >= self.group_dro_warmup_epochs
        if active:
            if update_state:
                self.group_dro_weights = update_group_weights(
                    self.group_dro_weights,
                    observed_means,
                    observed_present,
                    eta=self.group_dro_eta,
                    min_weight=self.group_dro_min_weight,
                ).to(self.accelerator.device)
            loss = robust_present_group_loss(local_means, local_present, self.group_dro_weights)
        else:
            loss = per_sample_losses.mean()

        q = self.group_dro_weights
        entropy = -torch.sum(q.clamp_min(torch.finfo(torch.float32).tiny) * torch.log(q.clamp_min(torch.finfo(torch.float32).tiny)))
        return loss, {
            "erm_loss": per_sample_losses.mean().detach(),
            "group_dro_max_weight": q.max().detach(),
            "group_dro_entropy": entropy.detach(),
            "group_dro_active": torch.tensor(float(active), device=loss.device),
        }

    def train_epoch_end(self, train_losses: dict, train_times: list):
        super().train_epoch_end(train_losses, train_times)
        counts = self.group_dro_epoch_observations
        mean_losses = torch.zeros_like(counts)
        present = counts > 0
        mean_losses[present] = self.group_dro_epoch_loss_sum[present] / counts[present]
        order = torch.argsort(self.group_dro_weights, descending=True)
        top = []
        for index in order[: min(8, len(order))].tolist():
            observed = float(mean_losses[index].detach().cpu()) if bool(present[index]) else float("nan")
            top.append(
                f"{self.group_dro_event_names[index]}={float(self.group_dro_weights[index].detach().cpu()):.4f}"
                f"/loss={observed:.4f}"
            )
        LOG.info(
            "GroupDRO epoch %d: active=%s | effective groups=%.2f/%d | max weight=%.4f | top events: %s",
            self.current_epoch + 1,
            self.current_epoch >= self.group_dro_warmup_epochs,
            effective_group_count(self.group_dro_weights),
            len(self.group_dro_event_names),
            float(self.group_dro_weights.max().detach().cpu()),
            ", ".join(top),
        )
        if self.is_main and self.group_dro_history_path is not None:
            append_event_weight_rows(
                self.group_dro_history_path,
                epoch=self.current_epoch + 1,
                event_names=self.group_dro_event_names,
                weights=self.group_dro_weights,
                mean_losses=mean_losses,
                observation_counts=counts,
            )

    def _extra_checkpoint_state(self):
        state = dict(super()._extra_checkpoint_state())
        state["group_dro"] = {
            "event_names": list(self.group_dro_event_names),
            "weights": self.group_dro_weights.detach().cpu(),
            "eta": self.group_dro_eta,
            "min_weight": self.group_dro_min_weight,
            "warmup_epochs": self.group_dro_warmup_epochs,
        }
        return state

    def _restore_extra_checkpoint_state(self, checkpoint):
        super()._restore_extra_checkpoint_state(checkpoint)
        state = (checkpoint.get("trainer_state") or {}).get("group_dro")
        if not state:
            LOG.warning("Resume checkpoint has no GroupDRO state; event weights reset to uniform.")
            return
        saved_names = list(state.get("event_names") or [])
        if saved_names != self.group_dro_event_names:
            raise RuntimeError(
                "Resume checkpoint GroupDRO event order does not match the current training dataset. "
                f"saved={saved_names} current={self.group_dro_event_names}"
            )
        saved_weights = torch.as_tensor(state.get("weights"), dtype=torch.float32, device=self.accelerator.device)
        if saved_weights.shape != self.group_dro_weights.shape:
            raise RuntimeError("Resume checkpoint GroupDRO weight vector has the wrong shape")
        self.group_dro_weights = saved_weights / saved_weights.sum()
        LOG.info("Restored GroupDRO event weights from resumable checkpoint.")


class GroupDROFloodTrainer(_GroupDROMixin, FloodTrainer):
    def __init__(self, *args, event_names, group_dro_eta=0.01, group_dro_min_weight=0.001, group_dro_warmup_epochs=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_group_dro(
            event_names=event_names, eta=group_dro_eta, min_weight=group_dro_min_weight,
            warmup_epochs=group_dro_warmup_epochs,
        )

    def train_batch(
        self,
        batch: Any,
        *,
        full_precision: bool = False,
        record_state: bool = True,
    ) -> torch.Tensor:
        x, y, crop_metadata, group_indices = _unpack_training_batch(batch)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        with self.precision_context(full_precision=full_precision):
            out = self.model(x)
            per_sample_losses = torch.stack([
                self.criterion(out[index:index + 1], y[index:index + 1].long())
                for index in range(out.shape[0])
            ])
            loss, dro_data = self._group_dro_loss(
                per_sample_losses, group_indices, update_state=record_state
            )
        if record_state:
            y_true = self.accelerator.gather(y)
            y_pred = self.accelerator.gather(out)
            self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        if self.debug:
            self._debug_training(x=x.dtype, y=y.dtype, pred=out.dtype, loss=loss)
        data = {"loss": loss.detach(), **dro_data}
        if record_state:
            data.update(_gather_crop_metadata(self.accelerator, crop_metadata))
        return loss, data


class GroupDROMultiBranchTrainer(_GroupDROMixin, MultiBranchTrainer):
    def __init__(self, *args, event_names, group_dro_eta=0.01, group_dro_min_weight=0.001, group_dro_warmup_epochs=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_group_dro(
            event_names=event_names, eta=group_dro_eta, min_weight=group_dro_min_weight,
            warmup_epochs=group_dro_warmup_epochs,
        )

    def train_batch(
        self,
        batch: Any,
        *,
        full_precision: bool = False,
        record_state: bool = True,
    ) -> torch.Tensor:
        x, y, crop_metadata, group_indices = _unpack_training_batch(batch)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        with self.precision_context(full_precision=full_precision):
            out, aux = self.model(x)
            per_sample_losses = torch.stack([
                self.criterion(out[index:index + 1], y[index:index + 1].long())
                + 0.4 * self.criterion(aux[index:index + 1], y[index:index + 1].long())
                for index in range(out.shape[0])
            ])
            loss, dro_data = self._group_dro_loss(
                per_sample_losses, group_indices, update_state=record_state
            )
        if record_state:
            y_true = self.accelerator.gather(y)
            y_pred = self.accelerator.gather(out)
            self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        if self.debug:
            self._debug_training(x=x.dtype, y=y.dtype, pred=out.dtype, loss=loss)
        data = {"loss": loss.detach(), **dro_data}
        if record_state:
            data.update(_gather_crop_metadata(self.accelerator, crop_metadata))
        return loss, data
