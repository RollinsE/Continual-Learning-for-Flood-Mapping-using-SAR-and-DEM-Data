from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch
from matplotlib import pyplot as plt

from floods.logging.functional import make_grid, mask_to_rgb
from floods.utils.common import get_logger, prepare_folder

if TYPE_CHECKING:
    from floods.trainer.base import Trainer

LOG = get_logger(__name__)


class BaseCallback:
    def __init__(self, call_every: int = 1, call_once: int = None) -> None:
        if call_every is None and call_once is None:
            raise ValueError("Specify at least one of call_every or call_once")
        if call_every is not None:
            if call_every <= 0:
                raise ValueError("call_every must be at least 1")
        if call_once is not None:
            if call_once < 0:
                raise ValueError("call_once must be non-negative")
        self.call_every = call_every
        self.call_once = call_once
        self.expired = False

    def __call__(self, trainer: "Trainer", *args: Any, **kwds: Any) -> Any:
        # Return after the configured one-time callback has already run.
        if self.expired:
            return
        if self.call_once is not None and self.call_once == trainer.current_epoch:
            data = self.call(trainer, *args, **kwds)
            self.expired = True
            return data
        if self.call_every is not None:
            if (trainer.current_epoch % self.call_every) == 0:
                return self.call(trainer, *args, **kwds)

    def setup(self, trainer: "Trainer"):
        pass

    def call(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Callback not implemented!")

    def dispose(self, trainer: "Trainer"):
        pass


class EarlyStoppingCriterion(Enum):
    minimum = torch.lt
    maximum = torch.gt


class EarlyStopping(BaseCallback):

    criteria = {"min": torch.lt, "max": torch.gt}

    def __init__(self,
                 call_every: int,
                 metric: str,
                 criterion: EarlyStoppingCriterion = EarlyStoppingCriterion.minimum,
                 patience: int = 10) -> None:
        super().__init__(call_every=call_every)
        self.metric = metric
        self.criterion = criterion.value
        self.patience = patience
        self.patience_counter = None

    def setup(self, trainer: "Trainer"):
        metrics = trainer.metrics["val"]
        sweep_metric = str(self.metric).startswith("best_") or str(self.metric).startswith("best_threshold_")
        if self.metric not in metrics and not (sweep_metric and getattr(trainer, "threshold_sweep_enabled", False)):
            raise ValueError(f"Monitored metric '{self.metric}' not in validation metrics: {list(metrics.keys())}")
        self.patience_counter = 0

    def call(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> Any:
        # Stop safely if mixed-precision training produces an invalid loss.
        if trainer.current_loss is not None and torch.isnan(trainer.current_loss):
            LOG.info("Non-finite training loss detected | epoch=%d | action=stop", trainer.current_epoch + 1)
            trainer.stop_reason = "nan_loss"
            raise KeyboardInterrupt
        # Keep distributed workers aligned before updating shared best-score state.
        current_score = trainer.current_scores["val"][self.metric]
        previous_best = trainer.best_score
        if previous_best is None or self.criterion(current_score, previous_best):
            self.patience_counter = 0
            trainer.accelerator.wait_for_everyone()
            trainer.best_score = current_score
            trainer.best_epoch = trainer.current_epoch
            # Store the unwrapped model state so checkpointing and resume remain stable.
            trainer.best_state_dict = trainer.accelerator.unwrap_model(trainer.model).state_dict()
            current_value = trainer._scalar(current_score)
            if previous_best is None:
                LOG.info("Validation metric improved | metric=%s | value=%.4f | previous=-inf", self.metric, current_value)
            else:
                LOG.info("Validation metric improved | metric=%s | value=%.4f | delta=%+.6f", self.metric, current_value, current_value - trainer._scalar(previous_best))
        else:
            self.patience_counter += 1
            LOG.info("Early stopping patience | counter=%d/%d", self.patience_counter, self.patience)
            if self.patience_counter >= self.patience:
                LOG.info("Early stopping triggered")
                trainer.accelerator.unwrap_model(trainer.model).load_state_dict(trainer.best_state_dict)
                # The training loop catches this and saves the latest resumable checkpoint.
                trainer.stop_reason = "early_stopping"
                raise KeyboardInterrupt

    def dispose(self, trainer: "Trainer"):
        self.patience_counter = 0


class Checkpoint(BaseCallback):
    def __init__(self,
                 call_every: int,
                 model_folder: Path,
                 monitor: str,
                 name_suffix: str = "",
                 save_every: int = None,
                 save_best: bool = True,
                 verbose: bool = True) -> None:
        super().__init__(call_every=call_every)
        model_folder = prepare_folder(model_folder)
        if not model_folder.exists() or not model_folder.is_dir():
            raise FileNotFoundError(f"Invalid model directory: {model_folder}")
        if not (save_every or save_best):
            raise ValueError("Enable at least one of save_every or save_best")
        self.model_folder = model_folder
        self.name_format = name_suffix
        self.save_every = save_every
        self.save_best = save_best
        self.verbose = verbose
        self.monitor = monitor
        self.best_epoch = None

    def _should_save(self, trainer: "Trainer") -> bool:
        # Save on configured epoch intervals.
        if self.save_every and ((trainer.current_epoch + 1) % self.save_every == 0):
            return True
        # Save when the monitored validation score improves.
        if self.save_best and (self.best_epoch is None or self.best_epoch < trainer.best_epoch):
            self.best_epoch = trainer.best_epoch
            return True
        return False

    def setup(self, trainer: "Trainer"):
        self.best_epoch = None

    def call(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> Any:
        if self._should_save(trainer):
            # Include epoch and monitored score in the checkpoint filename.
            score = trainer.current_scores["val"][self.monitor]
            epoch = trainer.current_epoch
            safe_monitor = str(self.monitor).replace("/", "_")
            model_name = f"model-{epoch + 1:03d}_{safe_monitor}-{trainer._scalar(score):.4f}"
            filename = self.model_folder / f"{model_name}.pth"
            # Synchronize workers before saving from the main process.
            trainer.accelerator.wait_for_everyone()
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            trainer.accelerator.save(unwrapped_model.state_dict(), filename)
            if self.verbose:
                LOG.info("Best model checkpoint saved: %s", str(filename))
        else:
            if self.verbose:
                LOG.debug("No best-model checkpoint saved this epoch")

    def dispose(self, trainer: "Trainer"):
        self.best_epoch = None


class DisplaySamples(BaseCallback):
    def __init__(self,
                 inverse_transform: Callable,
                 mask_palette: Dict[int, tuple],
                 image_transform: Optional[Callable] = None,
                 slice_at: int = -1,
                 call_every: int = 1,
                 stage: str = "val") -> None:
        super().__init__(call_every=call_every)
        self.inverse_transform = inverse_transform
        self.image_transform = image_transform
        self.color_palette = mask_palette
        self.slice_at = slice_at
        self.stage = stage

    def setup(self, trainer: "Trainer"):
        if trainer.sample_batches is None or len(trainer.sample_batches) == 0:
            LOG.warn("An ImagePlotter callback is active, but no samples have been found, have you set them?")

    def call(self, trainer: "Trainer", *args: Any, filepath: str = None, filename: str = None, **kwargs: Any) -> Any:
        if not trainer.sample_content:
            LOG.warn("No content to be displayed")
        for i, (image, y_true, y_pred) in enumerate(trainer.sample_content):
            image = self.inverse_transform(image)
            if self.image_transform is not None:
                image = self.image_transform(image[:, :, :self.slice_at])
            true_masks = mask_to_rgb(y_true.numpy(), palette=self.color_palette)
            pred_masks = mask_to_rgb(y_pred.numpy(), palette=self.color_palette)
            grid = make_grid(image, true_masks, pred_masks)
            if filepath:
                plt.imsave(filepath / f"{filename}.png", grid)
            else:
                trainer.logger.log_image(f"{self.stage}/sample-{i}", image=grid, step=trainer.current_epoch)

    def dispose(self, trainer: "Trainer"):
        trainer.sample_content.clear()
