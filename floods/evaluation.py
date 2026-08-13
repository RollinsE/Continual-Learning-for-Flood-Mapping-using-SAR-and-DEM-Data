from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from floods.utils.console import progress_iter
from typing import TYPE_CHECKING

from floods.utils.common import get_logger

if TYPE_CHECKING:
    from floods.config.training import TrainConfig

LOG = get_logger(__name__)


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    iou: float
    mcc: float
    empty_tile_fp_rate: float
    nonempty_tile_recall: float
    tp: float
    tn: float
    fp: float
    fn: float


class BinaryThresholdSweep:
    """Accumulates binary segmentation metrics across multiple probability thresholds."""

    def __init__(self, thresholds: Optional[Iterable[float]] = None, ignore_index: int = 255, device: torch.device | str = "cpu") -> None:
        values = list(thresholds or DEFAULT_THRESHOLDS)
        if not values:
            values = [0.5]
        values = sorted(float(v) for v in values)
        self.thresholds = torch.tensor(values, dtype=torch.float32, device=device)
        self.ignore_index = ignore_index
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        n = len(self.thresholds)
        self.tp = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.fp = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.tn = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.fn = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.empty_tiles = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.empty_tile_fp = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.nonempty_tiles = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.nonempty_tile_detected = torch.zeros(n, dtype=torch.float64, device=self.device)

    @staticmethod
    def _main_prediction(output: Any) -> torch.Tensor:
        """Return the primary logits tensor from a model output structure."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            if not output:
                raise ValueError("Model returned an empty output sequence.")
            return BinaryThresholdSweep._main_prediction(output[0])
        if isinstance(output, dict):
            for key in ("logits", "out", "prediction", "pred", "mask"):
                if key in output:
                    return BinaryThresholdSweep._main_prediction(output[key])
            if output:
                first_value = next(iter(output.values()))
                return BinaryThresholdSweep._main_prediction(first_value)
        raise TypeError(f"Unsupported prediction output type: {type(output)!r}")

    @staticmethod
    def _squeeze_logits(y_pred: Any) -> torch.Tensor:
        y_pred = BinaryThresholdSweep._main_prediction(y_pred)
        if y_pred.ndim == 4 and y_pred.shape[1] == 1:
            return y_pred[:, 0]
        if y_pred.ndim == 4 and y_pred.shape[1] > 1:
            return y_pred[:, 1]
        return y_pred

    def update(self, y_true: torch.Tensor, y_pred: Any) -> None:
        with torch.no_grad():
            logits = self._squeeze_logits(y_pred).detach().float().to(self.device)
            target = y_true.detach().to(self.device)
            if target.ndim == 4 and target.shape[1] == 1:
                target = target[:, 0]
            valid = target != self.ignore_index
            target_fg = (target > 0) & valid
            prob = torch.sigmoid(logits)

            for i, threshold in enumerate(self.thresholds):
                pred_fg = (prob >= threshold) & valid
                self.tp[i] += torch.count_nonzero(pred_fg & target_fg).double()
                self.fp[i] += torch.count_nonzero(pred_fg & (~target_fg) & valid).double()
                self.tn[i] += torch.count_nonzero((~pred_fg) & (~target_fg) & valid).double()
                self.fn[i] += torch.count_nonzero((~pred_fg) & target_fg).double()

                flat_valid = valid.reshape(valid.shape[0], -1)
                flat_target = target_fg.reshape(target_fg.shape[0], -1)
                flat_pred = pred_fg.reshape(pred_fg.shape[0], -1)
                valid_tile = flat_valid.any(dim=1)
                true_any = flat_target.any(dim=1) & valid_tile
                pred_any = flat_pred.any(dim=1) & valid_tile
                empty = (~true_any) & valid_tile
                nonempty = true_any
                self.empty_tiles[i] += torch.count_nonzero(empty).double()
                self.empty_tile_fp[i] += torch.count_nonzero(empty & pred_any).double()
                self.nonempty_tiles[i] += torch.count_nonzero(nonempty).double()
                self.nonempty_tile_detected[i] += torch.count_nonzero(nonempty & pred_any).double()

    def compute(self) -> List[ThresholdResult]:
        eps = torch.tensor(1e-12, dtype=torch.float64, device=self.device)
        precision = self.tp / torch.clamp(self.tp + self.fp, min=eps)
        recall = self.tp / torch.clamp(self.tp + self.fn, min=eps)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=eps)
        iou = self.tp / torch.clamp(self.tp + self.fp + self.fn, min=eps)
        denom = torch.sqrt(torch.clamp((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn), min=eps))
        mcc = ((self.tp * self.tn) - (self.fp * self.fn)) / denom
        empty_fp_rate = self.empty_tile_fp / torch.clamp(self.empty_tiles, min=eps)
        nonempty_recall = self.nonempty_tile_detected / torch.clamp(self.nonempty_tiles, min=eps)

        results = []
        for i, threshold in enumerate(self.thresholds.detach().cpu().tolist()):
            results.append(ThresholdResult(
                threshold=float(threshold),
                precision=float(precision[i].detach().cpu()),
                recall=float(recall[i].detach().cpu()),
                f1=float(f1[i].detach().cpu()),
                iou=float(iou[i].detach().cpu()),
                mcc=float(mcc[i].detach().cpu()),
                empty_tile_fp_rate=float(empty_fp_rate[i].detach().cpu()),
                nonempty_tile_recall=float(nonempty_recall[i].detach().cpu()),
                tp=float(self.tp[i].detach().cpu()),
                tn=float(self.tn[i].detach().cpu()),
                fp=float(self.fp[i].detach().cpu()),
                fn=float(self.fn[i].detach().cpu()),
            ))
        return results

    def best(self, metric: str = "f1") -> ThresholdResult:
        results = self.compute()
        if metric not in {"f1", "iou", "mcc", "precision", "recall"}:
            metric = "f1"
        return max(results, key=lambda item: getattr(item, metric))

    def to_table(self, metric: str = "f1") -> str:
        results = self.compute()
        lines = ["threshold  f1      iou     precision  recall   mcc     empty_fp  nonempty_recall"]
        for row in results:
            lines.append(
                f"{row.threshold:>8.2f}  {row.f1:>6.4f}  {row.iou:>6.4f}  {row.precision:>9.4f}  "
                f"{row.recall:>6.4f}  {row.mcc:>6.4f}  {row.empty_tile_fp_rate:>8.4f}  {row.nonempty_tile_recall:>15.4f}"
            )
        best = self.best(metric)
        lines.append(f"Best {metric}: {getattr(best, metric):.4f} at threshold {best.threshold:.2f}")
        return "\n".join(lines)


class BatchAverageMetrics:
    """Batch-averaged binary segmentation metrics at a fixed threshold.

    Global pixel metrics are preferred for reporting. This class is retained
    only as an optional diagnostic and honours the ignore index by default.
    """

    def __init__(self, threshold: float = 0.5, ignore_index: int | None = 255, device: torch.device | str = "cpu") -> None:
        self.threshold = float(threshold)
        self.ignore_index = ignore_index
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.losses: List[float] = []
        self.f1: List[float] = []
        self.iou: List[float] = []
        self.precision: List[float] = []
        self.recall: List[float] = []
        self.mcc: List[float] = []
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, y_true: torch.Tensor, y_pred: Any, loss: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            logits = BinaryThresholdSweep._squeeze_logits(y_pred).detach().float().to(self.device)
            target = y_true.detach().to(self.device)
            if target.ndim == 4 and target.shape[1] == 1:
                target = target[:, 0]
            if self.ignore_index is None:
                valid = torch.ones_like(target, dtype=torch.bool)
            else:
                valid = target != self.ignore_index
            target_fg = (target > 0) & valid
            pred_fg = (torch.sigmoid(logits) > self.threshold) & valid

            tp = torch.count_nonzero(pred_fg & target_fg).double()
            fp = torch.count_nonzero(pred_fg & (~target_fg) & valid).double()
            tn = torch.count_nonzero((~pred_fg) & (~target_fg) & valid).double()
            fn = torch.count_nonzero((~pred_fg) & target_fg).double()
            eps = torch.tensor(1e-6, dtype=torch.float64, device=self.device)
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
            iou = (tp + eps) / (tp + fp + fn + eps)
            denom = torch.sqrt(torch.clamp((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), min=float(eps)))
            mcc = ((tp * tn) - (fp * fn)) / denom
            mcc = torch.nan_to_num(mcc, nan=0.0, posinf=0.0, neginf=0.0)

            if loss is not None:
                self.losses.append(float(loss.detach().float().cpu()))
            self.precision.append(float(precision.cpu()))
            self.recall.append(float(recall.cpu()))
            self.f1.append(float(f1.cpu()))
            self.iou.append(float(iou.cpu()))
            self.mcc.append(float(mcc.cpu()))
            self.tp += float(tp.cpu())
            self.tn += float(tn.cpu())
            self.fp += float(fp.cpu())
            self.fn += float(fn.cpu())

    def compute(self) -> Dict[str, float]:
        def avg(values: List[float]) -> float:
            return float(np.mean(values)) if values else float("nan")
        return {
            "loss": avg(self.losses),
            "f1": avg(self.f1),
            "iou": avg(self.iou),
            "precision": avg(self.precision),
            "recall": avg(self.recall),
            "mcc": avg(self.mcc),
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "threshold": self.threshold,
        }

    def to_line(self) -> str:
        m = self.compute()
        return (f"Batch-averaged metrics @ {self.threshold:.2f}: "
                f"f1={m['f1']:.4f} | iou={m['iou']:.4f} | precision={m['precision']:.4f} | "
                f"recall={m['recall']:.4f} | mcc={m['mcc']:.4f}")


def _event_id_from_path(path: str | Path) -> str:
    match = re.search(r"(EMSR\d+)", Path(path).name)
    return match.group(1) if match else "unknown"


def _filter_dataset_by_events(dataset: Any, include_events: Optional[Iterable[str]] = None, exclude_events: Optional[Iterable[str]] = None) -> None:
    include = {str(v).upper() for v in (include_events or [])}
    exclude = {str(v).upper() for v in (exclude_events or [])}
    if not include and not exclude:
        return
    mask = []
    for path in dataset.label_files:
        event = _event_id_from_path(path).upper()
        keep = True
        if include:
            keep = event in include
        if exclude and event in exclude:
            keep = False
        mask.append(keep)
    before = len(dataset)
    dataset.add_mask(mask)
    LOG.info("Event filter: kept %d/%d samples (include=%s exclude=%s)", len(dataset), before, sorted(include), sorted(exclude))
    if len(dataset) == 0:
        raise ValueError("Event filter removed every sample")


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, dict):
        if "best_model_state_dict" in checkpoint and checkpoint["best_model_state_dict"] is not None:
            return checkpoint["best_model_state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def evaluate_checkpoint(config: "TrainConfig",
                        checkpoint_path: Path,
                        split: str = "val",
                        thresholds: Optional[Iterable[float]] = None,
                        threshold_metric: str = "f1",
                        metric_mode: Optional[str] = None,
                        include_events: Optional[Iterable[str]] = None,
                        exclude_events: Optional[Iterable[str]] = None,
                        inference_mode: str = "direct",
                        window_size: int = 256,
                        window_overlap: int = 64,
                        window_batch_size: int = 4) -> Dict[str, float]:
    from accelerate import Accelerator
    from floods.datasets.flood import FloodDataset, RGBFloodDataset
    from floods.prepare import prepare_evaluation_dataset, prepare_model
    from floods.normalization import describe_stats, load_normalization_stats
    from floods.utils.ml import seed_everything, seed_worker
    from floods.eval_collate import pad_segmentation_batch
    from floods.sliding_window import sliding_window_logits

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

    model = prepare_model(config=config, num_classes=1, stage="eval")
    state = load_checkpoint_state(Path(checkpoint_path))
    model.load_state_dict(state, strict=not config.model.multibranch)
    model = model.to(accelerator.device)
    model, loader = accelerator.prepare(model, loader)
    sweep = BinaryThresholdSweep(thresholds=thresholds, device=accelerator.device)
    metric_mode = str(metric_mode or getattr(config.trainer, "metric_mode", "global") or "global").lower()
    batch_average_metrics = BatchAverageMetrics(threshold=0.5, ignore_index=255, device=accelerator.device) if metric_mode in {"batch_average", "both"} else None
    inference_mode = str(inference_mode or "direct").lower().replace("-", "_")
    if inference_mode not in {"direct", "sliding_window"}:
        raise ValueError("inference_mode must be direct or sliding_window")
    if inference_mode == "sliding_window":
        LOG.info("Using sliding-window inference: window_size=%d overlap=%d window_batch_size=%d", int(window_size), int(window_overlap), int(window_batch_size))

    LOG.info("Evaluating checkpoint: %s", checkpoint_path)
    LOG.info("Dataset: %s split, %d samples", split, len(dataset))
    model.eval()
    with torch.no_grad():
        for x, y in progress_iter(loader, desc=f"Evaluate {split}", unit="batch", colour="green"):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            if inference_mode == "sliding_window":
                for sample_idx in range(x.shape[0]):
                    sample_x = x[sample_idx:sample_idx + 1]
                    sample_y = y[sample_idx:sample_idx + 1]
                    with accelerator.autocast():
                        logits = sliding_window_logits(model, sample_x, window_size=window_size, overlap=window_overlap, window_batch_size=window_batch_size)
                    y_true = accelerator.gather(sample_y)
                    y_pred = accelerator.gather(logits)
                    sweep.update(y_true, y_pred)
                    if batch_average_metrics is not None:
                        batch_average_metrics.update(y_true, y_pred)
            else:
                with accelerator.autocast():
                    out = model(x)
                logits = BinaryThresholdSweep._main_prediction(out)
                y_true = accelerator.gather(y)
                y_pred = accelerator.gather(logits)
                sweep.update(y_true, y_pred)
                if batch_average_metrics is not None:
                    batch_average_metrics.update(y_true, y_pred)

    LOG.info(
        "Validation threshold sweep (full table):\n%s",
        sweep.to_table(metric=threshold_metric),
        extra={"floodmap_file_only": True},
    )
    best = sweep.best(threshold_metric)
    LOG.info(
        "Validation threshold sweep best | metric=%s | threshold=%.2f | f1=%.4f | iou=%.4f | "
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

@dataclass
class EventThresholdResult:
    threshold: float
    macro_f1: float
    macro_iou: float
    worst_f1: float
    mean_precision: float
    mean_recall: float
    event_metrics: List[Dict[str, float]]


class EventMacroThresholdSweep:
    """Accumulate threshold metrics independently for each validation event."""

    def __init__(self, event_names: Iterable[str], thresholds: Optional[Iterable[float]] = None,
                 ignore_index: int = 255, device: torch.device | str = "cpu") -> None:
        self.event_names = [str(v) for v in event_names]
        if not self.event_names:
            raise ValueError("Event-macro validation requires at least one validation event")
        values = sorted(float(v) for v in (thresholds or DEFAULT_THRESHOLDS))
        self.thresholds = torch.tensor(values or [0.5], dtype=torch.float32, device=device)
        self.ignore_index = int(ignore_index)
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        shape = (len(self.thresholds), len(self.event_names))
        self.tp = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.fp = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.fn = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.valid = torch.zeros(len(self.event_names), dtype=torch.float64, device=self.device)

    def update(self, y_true: torch.Tensor, y_pred: Any, event_indices: torch.Tensor) -> None:
        with torch.no_grad():
            logits = BinaryThresholdSweep._squeeze_logits(y_pred).detach().float().to(self.device)
            target = y_true.detach().to(self.device)
            if target.ndim == 4 and target.shape[1] == 1:
                target = target[:, 0]
            groups = event_indices.detach().reshape(-1).long().to(self.device)
            if groups.numel() != target.shape[0]:
                raise ValueError("Event index count does not match validation batch size")
            prob = torch.sigmoid(logits)
            valid = target != self.ignore_index
            truth = (target > 0) & valid
            for sample_idx in range(target.shape[0]):
                group = int(groups[sample_idx])
                if group < 0 or group >= len(self.event_names):
                    raise ValueError("Validation event index outside configured range")
                sample_valid = valid[sample_idx]
                if not bool(sample_valid.any()):
                    continue
                self.valid[group] += 1.0
                sample_truth = truth[sample_idx]
                sample_prob = prob[sample_idx]
                for threshold_idx, threshold in enumerate(self.thresholds):
                    pred = (sample_prob >= threshold) & sample_valid
                    self.tp[threshold_idx, group] += torch.count_nonzero(pred & sample_truth).double()
                    self.fp[threshold_idx, group] += torch.count_nonzero(pred & (~sample_truth) & sample_valid).double()
                    self.fn[threshold_idx, group] += torch.count_nonzero((~pred) & sample_truth).double()

    def compute(self) -> List[EventThresholdResult]:
        eps = torch.tensor(1e-12, dtype=torch.float64, device=self.device)
        results: List[EventThresholdResult] = []
        active = self.valid > 0
        for threshold_idx, threshold in enumerate(self.thresholds.detach().cpu().tolist()):
            tp = self.tp[threshold_idx]
            fp = self.fp[threshold_idx]
            fn = self.fn[threshold_idx]
            precision = tp / torch.clamp(tp + fp, min=eps)
            recall = tp / torch.clamp(tp + fn, min=eps)
            f1 = 2.0 * tp / torch.clamp(2.0 * tp + fp + fn, min=eps)
            iou = tp / torch.clamp(tp + fp + fn, min=eps)
            used = active
            if not bool(used.any()):
                raise RuntimeError("Event-macro validation observed no valid pixels")
            rows = []
            for event_idx, name in enumerate(self.event_names):
                if not bool(used[event_idx]):
                    continue
                rows.append({
                    "event_id": name,
                    "threshold": float(threshold),
                    "f1": float(f1[event_idx].detach().cpu()),
                    "iou": float(iou[event_idx].detach().cpu()),
                    "precision": float(precision[event_idx].detach().cpu()),
                    "recall": float(recall[event_idx].detach().cpu()),
                    "tp": float(tp[event_idx].detach().cpu()),
                    "fp": float(fp[event_idx].detach().cpu()),
                    "fn": float(fn[event_idx].detach().cpu()),
                })
            results.append(EventThresholdResult(
                threshold=float(threshold),
                macro_f1=float(f1[used].mean().detach().cpu()),
                macro_iou=float(iou[used].mean().detach().cpu()),
                worst_f1=float(f1[used].min().detach().cpu()),
                mean_precision=float(precision[used].mean().detach().cpu()),
                mean_recall=float(recall[used].mean().detach().cpu()),
                event_metrics=rows,
            ))
        return results

    def best(self) -> EventThresholdResult:
        return max(self.compute(), key=lambda item: (item.macro_f1, item.worst_f1))
