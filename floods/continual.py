from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from floods.utils.console import progress_iter

from floods.config import TrainConfig
from floods.datasets.flood import FloodDataset, RGBFloodDataset
from floods.eval_collate import pad_segmentation_batch
from floods.evaluation import BinaryThresholdSweep, ThresholdResult
from floods.normalization import describe_stats, load_normalization_stats
from floods.sliding_window import sliding_window_logits
from floods.utils.common import config_to_plain_dict, get_logger, init_experiment, store_config
from floods.utils.ml import seed_everything, seed_worker

LOG = get_logger(__name__)

DEFAULT_CL_STRATEGIES = ("random", "least_confidence", "margin", "entropy")
DEFAULT_TASK_YEAR_RANGES = ("2014-2017", "2018-2019", "2020-2021")


@dataclass
class CLTask:
    """Container for one chronological continual-learning task."""

    task_id: int
    name: str
    start_year: int
    end_year: int
    train_indices: List[int]
    eval_indices: List[int]


class ReplayBuffer:
    """Reservoir replay buffer storing indices into the training dataset."""

    def __init__(self, max_size: int, seed: int = 42) -> None:
        self.max_size = int(max_size)
        if self.max_size < 0:
            raise ValueError("replay_buffer_size must be >= 0")
        self.indices: List[int] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def add_many(self, indices: Iterable[int]) -> None:
        for index in indices:
            self.add(int(index))

    def add(self, index: int) -> None:
        if self.max_size == 0:
            self.seen += 1
            return
        self.seen += 1
        if len(self.indices) < self.max_size:
            self.indices.append(index)
            return
        replacement_position = self.rng.randint(0, self.seen - 1)
        if replacement_position < self.max_size:
            self.indices[replacement_position] = index

    def random_sample(self, count: int) -> List[int]:
        if len(self.indices) == 0 or count <= 0:
            return []
        return self.rng.sample(self.indices, min(int(count), len(self.indices)))


def parse_year_ranges(values: Sequence[str] | None) -> List[Tuple[int, int]]:
    """Parse year range strings such as '2014-2017' or '2020'."""
    values = list(values or DEFAULT_TASK_YEAR_RANGES)
    ranges: List[Tuple[int, int]] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        match = re.fullmatch(r"(\d{4})(?:\s*[-:]\s*(\d{4}))?", text)
        if not match:
            raise ValueError(f"Invalid task year range '{raw}'. Use values like 2014-2017 or 2021.")
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        if end < start:
            raise ValueError(f"Invalid task year range '{raw}': end year is before start year.")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("At least one CL task year range is required.")
    return ranges


def load_event_years(activations_json_path: Path) -> Dict[str, int]:
    """Load EMSR event start years from the MMFlood activations metadata."""
    path = Path(activations_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Activations JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    event_years: Dict[str, int] = {}
    for event_id, details in data.items():
        start = str((details or {}).get("start", ""))
        match = re.match(r"(\d{4})", start)
        if match:
            event_years[str(event_id).upper()] = int(match.group(1))
    if not event_years:
        raise ValueError(f"No event start years could be parsed from {path}")
    return event_years


def event_id_from_path(path: str | Path) -> str:
    match = re.search(r"(EMSR\d+)", Path(path).name, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def _dataset_file_list(dataset: Any) -> List[str]:
    if not hasattr(dataset, "image_files"):
        raise TypeError("Expected a FloodDataset-like object with image_files")
    return list(dataset.image_files)


def indices_for_year_range(dataset: Any, event_years: Dict[str, int], start_year: int, end_year: int) -> List[int]:
    """Return dataset indices whose EMSR event start year falls within the range."""
    indices: List[int] = []
    for idx, image_path in enumerate(_dataset_file_list(dataset)):
        event_id = event_id_from_path(image_path)
        year = event_years.get(event_id)
        if year is not None and start_year <= year <= end_year:
            indices.append(idx)
    return indices


def build_cl_tasks(train_dataset: Any,
                   eval_dataset: Any,
                   event_years: Dict[str, int],
                   year_ranges: Sequence[Tuple[int, int]]) -> List[CLTask]:
    """Create chronological tasks for train and evaluation datasets."""
    tasks: List[CLTask] = []
    for task_number, (start_year, end_year) in enumerate(year_ranges, start=1):
        train_indices = indices_for_year_range(train_dataset, event_years, start_year, end_year)
        eval_indices = indices_for_year_range(eval_dataset, event_years, start_year, end_year)
        name = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
        tasks.append(CLTask(task_id=task_number,
                            name=name,
                            start_year=start_year,
                            end_year=end_year,
                            train_indices=train_indices,
                            eval_indices=eval_indices))
    return tasks


def _active_modalities(config: TrainConfig, use_rgb: bool) -> List[str]:
    return (["r", "g", "b"] if use_rgb else ["vv", "vh"]) + (["dem"] if config.data.include_dem else [])


def _normalization_for_config(config: TrainConfig, use_rgb: bool):
    dataset_cls = RGBFloodDataset if use_rgb else FloodDataset
    modalities = _active_modalities(config, use_rgb)
    norm_mode = str(getattr(config.data, "normalization_mode", "fixed") or "fixed").lower()
    if norm_mode in {"stats", "robust_percentile", "notebook_robust", "robust_minmax"} or getattr(config.data, "normalization_stats_path", None):
        if not config.data.normalization_stats_path:
            raise ValueError(f"normalization_mode='{norm_mode}' requires --normalization-stats-path")
        stats_path = Path(config.data.normalization_stats_path)
        mean, std, clip_min, clip_max = load_normalization_stats(stats_path, modalities, mode=norm_mode)
        LOG.info("Using train-fitted normalization stats (%s): %s", norm_mode, describe_stats(stats_path))
    else:
        mean = dataset_cls.mean()[:config.data.in_channels]
        std = dataset_cls.std()[:config.data.in_channels]
        clip_min = tuple([-30.0] * config.data.in_channels)
        clip_max = tuple([30.0] * config.data.in_channels)
        LOG.info("Using fixed dataset normalization statistics")
    from floods.prepare import eval_transforms
    return eval_transforms(mean=mean, std=std, clip_min=clip_min, clip_max=clip_max, normalization_mode=norm_mode)


def prepare_eval_split_dataset(config: TrainConfig, split: str, use_rgb: bool, normalization=None):
    """Instantiate a non-augmented split dataset for CL evaluation/scoring."""
    dataset_cls = RGBFloodDataset if use_rgb else FloodDataset
    normalization = normalization or _normalization_for_config(config, use_rgb)
    return dataset_cls(path=Path(config.data.path),
                       subset=split,
                       include_dem=config.data.include_dem,
                       normalization=normalization)


def clone_train_as_scoring_dataset(config: TrainConfig, train_dataset: Any, use_rgb: bool, normalization=None):
    """Create a non-augmented train dataset aligned to the filtered training dataset."""
    scoring_dataset = prepare_eval_split_dataset(config, split="train", use_rgb=use_rgb, normalization=normalization)
    scoring_dataset.image_files = list(train_dataset.image_files)
    scoring_dataset.label_files = list(train_dataset.label_files)
    if getattr(train_dataset, "_include_dem", False):
        scoring_dataset.dem_files = list(train_dataset.dem_files)
    return scoring_dataset


def collate_replay_samples(dataset: Any, indices: Sequence[int], device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load replay samples by index and stack them into tensors."""
    if not indices:
        return None, None
    images: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for index in indices:
        sample = dataset[int(index)]
        image, label = sample[:2]
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        if not torch.is_tensor(label):
            label = torch.as_tensor(label)
        images.append(image.float())
        labels.append(label.long())
    if not images:
        return None, None
    return torch.stack(images).to(device), torch.stack(labels).to(device)


def _main_logits(output: Any) -> torch.Tensor:
    return BinaryThresholdSweep._main_prediction(output)


def score_uncertainty(model: nn.Module,
                      scoring_dataset: Any,
                      candidate_indices: Sequence[int],
                      strategy: str,
                      device: torch.device,
                      batch_size: int,
                      num_workers: int) -> Dict[int, float]:
    """Score replay candidates; higher values are more uncertain."""
    if not candidate_indices:
        return {}
    subset = Subset(scoring_dataset, list(candidate_indices))
    loader = DataLoader(subset,
                        batch_size=max(1, int(batch_size)),
                        shuffle=False,
                        num_workers=max(0, int(num_workers)),
                        worker_init_fn=seed_worker)
    scores: Dict[int, float] = {}
    cursor = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0).to(device)
            logits = _main_logits(model(x))
            prob = torch.sigmoid(logits.float())
            if prob.ndim == 4 and prob.shape[1] == 1:
                prob = prob[:, 0]
            if strategy == "entropy":
                p = prob.clamp(1e-6, 1.0 - 1e-6)
                batch_scores = (-(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))).mean(dim=(1, 2))
            elif strategy == "margin":
                batch_scores = (1.0 - torch.abs(2.0 * prob - 1.0)).mean(dim=(1, 2))
            elif strategy == "least_confidence":
                confidence = torch.maximum(prob, 1.0 - prob)
                batch_scores = (1.0 - confidence).mean(dim=(1, 2))
            else:
                batch_scores = torch.rand(prob.shape[0], device=prob.device)
            for value in batch_scores.detach().cpu().tolist():
                if cursor < len(candidate_indices):
                    scores[int(candidate_indices[cursor])] = float(value)
                cursor += 1
    if was_training:
        model.train()
    return scores


def sample_replay_indices(buffer: ReplayBuffer,
                          strategy: str,
                          model: nn.Module,
                          scoring_dataset: Any,
                          replay_batch_size: int,
                          uncertainty_subset_fraction: float,
                          device: torch.device,
                          score_batch_size: int,
                          num_workers: int) -> List[int]:
    """Select replay indices using random or uncertainty-guided scoring."""
    if len(buffer) == 0 or replay_batch_size <= 0:
        return []
    strategy = str(strategy).lower().replace("-", "_")
    if strategy == "random":
        return buffer.random_sample(replay_batch_size)
    candidates = list(buffer.indices)
    if 0.0 < float(uncertainty_subset_fraction) < 1.0 and len(candidates) > replay_batch_size:
        candidate_count = max(replay_batch_size, int(math.ceil(len(candidates) * float(uncertainty_subset_fraction))))
        candidates = buffer.rng.sample(candidates, min(candidate_count, len(candidates)))
    scores = score_uncertainty(model=model,
                               scoring_dataset=scoring_dataset,
                               candidate_indices=candidates,
                               strategy=strategy,
                               device=device,
                               batch_size=score_batch_size,
                               num_workers=num_workers)
    if not scores:
        return buffer.random_sample(replay_batch_size)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in ranked[:min(replay_batch_size, len(ranked))]]


def _loss_weights(config: TrainConfig, train_dataset: Any, device: torch.device) -> Optional[torch.Tensor]:
    weights = None
    if config.data.pos_weight_from_train:
        from floods.prepare import compute_binary_pos_weight_from_labels
        pos_weight_value = compute_binary_pos_weight_from_labels(train_dataset.label_files,
                                                                 max_value=config.data.pos_weight_max,
                                                                 cache_hash=config.data.cache_hash,
                                                                 cache_dir=config.data.cache_dir,
                                                                 force_recompute=config.data.clear_cache)
        weights = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        LOG.info("Using train-derived positive-class weight for BCE/focal terms: %.4f (clipped max %.4f)",
                 pos_weight_value, float(config.data.pos_weight_max))
    return weights


def evaluate_logits(model: nn.Module,
                    dataloader: DataLoader,
                    device: torch.device,
                    thresholds: Sequence[float],
                    threshold_metric: str,
                    inference_mode: str = "direct",
                    window_size: int = 512,
                    window_overlap: int = 128,
                    window_batch_size: int = 1,
                    description: str = "Evaluate") -> Tuple[ThresholdResult, List[ThresholdResult]]:
    """Evaluate a model with global pixel metrics and a threshold sweep."""
    sweep = BinaryThresholdSweep(thresholds=thresholds, device=device)
    mode = str(inference_mode or "direct").lower().replace("-", "_")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for x, y in progress_iter(dataloader, desc=description, unit="batch", colour="green"):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0).to(device)
            y = y.to(device)
            if mode == "sliding_window":
                for sample_index in range(x.shape[0]):
                    logits = sliding_window_logits(model,
                                                   x[sample_index:sample_index + 1],
                                                   window_size=int(window_size),
                                                   overlap=int(window_overlap),
                                                   window_batch_size=int(window_batch_size))
                    sweep.update(y[sample_index:sample_index + 1], logits)
            else:
                logits = _main_logits(model(x))
                sweep.update(y, logits)
    if was_training:
        model.train()
    results = sweep.compute()
    return sweep.best(threshold_metric), results


def _threshold_results_to_dicts(results: Sequence[ThresholdResult]) -> List[Dict[str, float]]:
    return [dict(threshold=r.threshold,
                 f1=r.f1,
                 iou=r.iou,
                 precision=r.precision,
                 recall=r.recall,
                 mcc=r.mcc,
                 empty_fp=r.empty_tile_fp_rate,
                 nonempty_recall=r.nonempty_tile_recall) for r in results]


def _best_to_dict(best: ThresholdResult) -> Dict[str, float]:
    return dict(threshold=best.threshold,
                f1=best.f1,
                iou=best.iou,
                precision=best.precision,
                recall=best.recall,
                mcc=best.mcc,
                empty_fp=best.empty_tile_fp_rate,
                nonempty_recall=best.nonempty_tile_recall)


def _checkpoint_model_state(payload: Any) -> Any:
    """Return a model state dict from either a full CL checkpoint or a bare .pth state dict."""
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def _load_cl_checkpoint(path: Path, model: nn.Module, optimizer: Any = None, scheduler: Any = None, device: torch.device | None = None) -> Dict[str, Any]:
    """Load a CL checkpoint or bare model weights and return resumable metadata."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CL resume checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(_checkpoint_model_state(payload), strict=True)
    metadata: Dict[str, Any] = {"source": str(checkpoint_path), "kind": "model_state"}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata["kind"] = "full_checkpoint"
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        for key in ["best_score", "global_epoch", "replay_indices", "validation_history", "detailed_matrix",
                    "completed_task_id", "completed_epoch_in_task"]:
            if key in payload:
                metadata[key] = payload[key]
    return metadata


def _save_cl_checkpoint(path: Path,
                        model: nn.Module,
                        optimizer: Any,
                        scheduler: Any,
                        strategy: str,
                        best_score: float,
                        global_epoch: int,
                        replay_buffer: ReplayBuffer,
                        validation_history: Sequence[Dict[str, Any]],
                        detailed_matrix: Sequence[Dict[str, Any]],
                        completed_task_id: int,
                        completed_epoch_in_task: int) -> None:
    """Persist a resumable CL checkpoint after each completed epoch/task."""
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "strategy": strategy,
        "best_score": float(best_score),
        "global_epoch": int(global_epoch),
        "replay_indices": list(replay_buffer.indices),
        "validation_history": list(validation_history),
        "detailed_matrix": list(detailed_matrix),
        "completed_task_id": int(completed_task_id),
        "completed_epoch_in_task": int(completed_epoch_in_task),
    }
    torch.save(checkpoint, path)


def _next_position_from_checkpoint(metadata: Dict[str, Any], epochs_per_task: int) -> Tuple[int, int]:
    """Return one-based task id and one-based epoch-in-task to run next."""
    completed_task_id = int(metadata.get("completed_task_id", 0) or 0)
    completed_epoch = int(metadata.get("completed_epoch_in_task", 0) or 0)
    if completed_task_id <= 0:
        return 1, 1
    if completed_epoch >= int(epochs_per_task):
        return completed_task_id + 1, 1
    return completed_task_id, completed_epoch + 1


def _parse_ensemble_member(text: str) -> Tuple[str, str, str]:
    """Parse decoder:encoder[:label] member spec for CL ensembles."""
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid ensemble member '{text}'. Use decoder:encoder or decoder:encoder:label, e.g. unet:resnet50.")
    decoder, encoder = parts[0], parts[1]
    label = parts[2] if len(parts) >= 3 and parts[2] else f"{decoder}_{encoder}"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return decoder, encoder, label


def _load_cl_member_config(path: Path) -> TrainConfig:
    import yaml
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = yaml.load(text, Loader=yaml.FullLoader)
    return TrainConfig(**(data or {}))


def _summarise_cl_matrix(detailed_matrix: Sequence[Dict[str, Any]], metric: str = "f1") -> Dict[str, Any]:
    """Summarise task retention, plasticity and forgetting from the detailed CL matrix."""
    by_task: Dict[int, List[Tuple[int, float]]] = {}
    diagonal: List[float] = []
    for row in detailed_matrix:
        best = row.get("best") or {}
        if metric not in best:
            continue
        after_task = int(row.get("after_task", 0))
        eval_task = int(row.get("eval_task", 0))
        value = float(best[metric])
        by_task.setdefault(eval_task, []).append((after_task, value))
        if after_task == eval_task:
            diagonal.append(value)
    final_values: List[float] = []
    forgetting_values: List[float] = []
    per_task: List[Dict[str, float]] = []
    for eval_task, values in sorted(by_task.items()):
        values = sorted(values, key=lambda item: item[0])
        final_after, final_value = values[-1]
        best_seen = max(value for _, value in values)
        forgetting = max(0.0, best_seen - final_value)
        final_values.append(final_value)
        forgetting_values.append(forgetting)
        per_task.append({"eval_task": eval_task,
                         "final_after_task": final_after,
                         f"final_{metric}": final_value,
                         f"best_seen_{metric}": best_seen,
                         "forgetting": forgetting})
    return {"metric": metric,
            "mean_final_task_score": float(np.mean(final_values)) if final_values else float("nan"),
            "mean_plasticity": float(np.mean(diagonal)) if diagonal else float("nan"),
            "mean_forgetting": float(np.mean(forgetting_values)) if forgetting_values else float("nan"),
            "max_forgetting": float(np.max(forgetting_values)) if forgetting_values else float("nan"),
            "per_task": per_task}

def run_single_strategy(config: TrainConfig,
                        strategy: str,
                        tasks: Sequence[CLTask],
                        train_dataset: Any,
                        scoring_dataset: Any,
                        mixed_val_dataset: Any,
                        eval_dataset: Any,
                        output_root: Path,
                        epochs_per_task: int,
                        replay_buffer_size: int,
                        replay_batch_size: int,
                        uncertainty_subset_fraction: float,
                        eval_split: str,
                        eval_inference_mode: str,
                        window_size: int,
                        window_overlap: int,
                        window_batch_size: int,
                        run_label: Optional[str] = None,
                        replay_mode: str = "separate",
                        resume: bool = False,
                        resume_from: Optional[Path] = None,
                        resume_start_task: Optional[int] = None,
                        resume_start_epoch: Optional[int] = None) -> Dict[str, Any]:
    """Train one CL replay strategy and save its artefacts."""
    strategy = str(strategy).lower().replace("-", "_")
    if strategy not in DEFAULT_CL_STRATEGIES:
        raise ValueError(f"Unsupported CL strategy '{strategy}'. Choose from {DEFAULT_CL_STRATEGIES}")

    strategy_config = copy.deepcopy(config)
    base_name = strategy_config.name or "continual_learning"
    strategy_suffix = strategy if not run_label else f"{strategy}_{run_label}"
    strategy_config.name = f"{base_name}_{strategy_suffix}"
    strategy_config.output_folder = str(output_root)
    strategy_config.trainer.max_epochs = int(epochs_per_task) * len(tasks)

    exp_id, out_folder, model_folder, logs_folder = init_experiment(strategy_config, log_name="output.log")
    store_config(strategy_config, out_folder / "config.yaml")
    LOG.info("CL strategy started: %s", strategy)
    LOG.info("Output folder: %s", out_folder)
    replay_mode = str(replay_mode or "separate").lower().replace("-", "_")
    if replay_mode not in {"separate", "concat"}:
        raise ValueError("replay_mode must be 'separate' or 'concat'")
    LOG.info("Replay buffer: size=%d | replay_batch=%d | uncertainty_subset_fraction=%.3f | replay_mode=%s",
             int(replay_buffer_size), int(replay_batch_size), float(uncertainty_subset_fraction), replay_mode)

    device = torch.device("cpu" if strategy_config.trainer.cpu or not torch.cuda.is_available() else "cuda")
    if resume or resume_from is not None:
        # A full CL checkpoint or model state already contains trained weights; avoid fragile hub downloads on resume.
        strategy_config.model.pretrained = False
    from floods.prepare import prepare_model
    model = prepare_model(config=strategy_config, num_classes=1).to(device)
    params = [{"params": model.encoder_params(), "lr": strategy_config.optimizer.encoder_lr},
              {"params": model.decoder_params(), "lr": strategy_config.optimizer.decoder_lr}]
    optimizer = strategy_config.optimizer.instantiate(params)
    scheduler = strategy_config.scheduler.instantiate(optimizer)
    loss_weights = _loss_weights(strategy_config, train_dataset, device)
    criterion = strategy_config.loss.instantiate(ignore_index=255, weight=loss_weights).to(device)
    use_amp = bool(strategy_config.trainer.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_amp)
    from floods.logging.tensorboard import TensorBoardLogger
    logger = TensorBoardLogger(log_folder=logs_folder, comment=strategy_config.comment)

    val_loader = DataLoader(mixed_val_dataset,
                            batch_size=strategy_config.trainer.batch_size,
                            shuffle=False,
                            num_workers=strategy_config.trainer.num_workers,
                            worker_init_fn=seed_worker)
    eval_collate = pad_segmentation_batch if eval_split == "test" or eval_inference_mode == "sliding_window" else None
    replay_buffer = ReplayBuffer(max_size=replay_buffer_size, seed=strategy_config.seed)

    best_score = -float("inf")
    best_payload: Optional[Dict[str, Any]] = None
    validation_history: List[Dict[str, Any]] = []
    detailed_matrix: List[Dict[str, Any]] = []
    global_epoch = 0
    start_task_id = 1
    start_epoch_in_task = 1

    resume_path: Optional[Path] = Path(resume_from) if resume_from else None
    if resume and resume_path is None:
        candidate = model_folder / "last.ckpt"
        if candidate.exists():
            resume_path = candidate
        else:
            raise FileNotFoundError(
                f"--resume was requested but no resumable CL checkpoint was found at {candidate}. "
                "Use --resume-from PATH to a CL last.ckpt/model_best*.pth, or restart with a fresh --run-id."
            )
    if resume_path is not None:
        metadata = _load_cl_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler, device=device)
        best_score = float(metadata.get("best_score", best_score))
        global_epoch = int(metadata.get("global_epoch", global_epoch) or 0)
        validation_history = list(metadata.get("validation_history", validation_history) or [])
        detailed_matrix = list(metadata.get("detailed_matrix", detailed_matrix) or [])
        replay_buffer.indices = [int(i) for i in metadata.get("replay_indices", [])]
        replay_buffer.seen = max(len(replay_buffer.indices), int(metadata.get("replay_seen", len(replay_buffer.indices)) or len(replay_buffer.indices)))
        inferred_task, inferred_epoch = _next_position_from_checkpoint(metadata, int(epochs_per_task))
        start_task_id = int(resume_start_task or inferred_task)
        start_epoch_in_task = int(resume_start_epoch or inferred_epoch)
        if metadata.get("kind") == "model_state" and start_task_id > 1 and not replay_buffer.indices:
            # A bare best .pth does not contain replay state. Rebuild replay from completed prior tasks.
            for prior_task in tasks[:start_task_id - 1]:
                replay_buffer.add_many(prior_task.train_indices)
            global_epoch = (start_task_id - 1) * int(epochs_per_task) + max(0, start_epoch_in_task - 1)
        LOG.info("Resumed CL %s from %s | next task=%d next epoch-in-task=%d | replay=%d/%d",
                 strategy, resume_path, start_task_id, start_epoch_in_task, len(replay_buffer), replay_buffer.max_size)

    start_time = time.time()
    score_key = str(strategy_config.trainer.threshold_metric or "f1")
    thresholds = strategy_config.trainer.thresholds

    for task in tasks:
        if task.task_id < start_task_id:
            LOG.info("Skipping already-completed CL task %d/%d (%s) due to resume.", task.task_id, len(tasks), task.name)
            continue
        first_epoch = start_epoch_in_task if task.task_id == start_task_id else 1
        if first_epoch > int(epochs_per_task):
            LOG.info("Skipping already-completed CL task %d/%d (%s) due to resume.", task.task_id, len(tasks), task.name)
            continue
        if not task.train_indices:
            LOG.warning("CL task %d (%s) has no training samples; skipping training for this task.", task.task_id, task.name)
        else:
            task_loader = DataLoader(Subset(train_dataset, task.train_indices),
                                     batch_size=strategy_config.trainer.batch_size,
                                     shuffle=True,
                                     num_workers=strategy_config.trainer.num_workers,
                                     worker_init_fn=seed_worker,
                                     drop_last=False)
            LOG.info("CL task %d/%d (%s): train=%d | eval=%d | replay_before=%d",
                     task.task_id, len(tasks), task.name, len(task.train_indices), len(task.eval_indices), len(replay_buffer))
            for local_epoch in range(first_epoch - 1, int(epochs_per_task)):
                global_epoch += 1
                model.train()
                losses: List[float] = []
                task_bar = progress_iter(task_loader,
                                desc=f"CL {strategy} task {task.task_id}/{len(tasks)} epoch {local_epoch + 1}/{epochs_per_task}",
                                unit="batch",
                                colour="blue",
                                disable=not strategy_config.trainer.progress_bar)
                for batch in task_bar:
                    x, y = batch[:2]
                    replay_indices = sample_replay_indices(buffer=replay_buffer,
                                                           strategy=strategy,
                                                           model=model,
                                                           scoring_dataset=scoring_dataset,
                                                           replay_batch_size=replay_batch_size,
                                                           uncertainty_subset_fraction=uncertainty_subset_fraction,
                                                           device=device,
                                                           score_batch_size=strategy_config.trainer.batch_size,
                                                           num_workers=0)
                    rx, ry = collate_replay_samples(train_dataset, replay_indices, device=device)
                    x = x.float().to(device)
                    y = y.long().to(device)
                    x = torch.nan_to_num(x, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        if rx is not None and ry is not None and replay_mode == "concat":
                            # Concatenated replay appends replay samples to the current-task batch.
                            # This can OOM on 512x512 ResNet models and is retained only for exact comparability.
                            combined_x = torch.cat([x, torch.nan_to_num(rx.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)], dim=0)
                            combined_y = torch.cat([y, ry.long()], dim=0)
                            out = model(combined_x)
                            loss = criterion(out, combined_y)
                        else:
                            out = model(x)
                            current_loss = criterion(out, y)
                            if rx is not None and ry is not None:
                                replay_x = torch.nan_to_num(rx.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
                                replay_loss = criterion(model(replay_x), ry.long())
                                n_current = max(1, int(x.shape[0]))
                                n_replay = max(1, int(replay_x.shape[0]))
                                loss = (current_loss * n_current + replay_loss * n_replay) / float(n_current + n_replay)
                            else:
                                loss = current_loss
                    if not torch.isfinite(loss.detach()).all():
                        LOG.warning("Skipping non-finite CL loss at task %d epoch %d.", task.task_id, local_epoch + 1)
                        continue
                    scaler.scale(loss).backward()
                    if strategy_config.trainer.grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(strategy_config.trainer.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(float(loss.detach().cpu()))
                    task_bar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}"})
                try:
                    scheduler.step()
                except TypeError:
                    scheduler.step(float(np.mean(losses)) if losses else 0.0)

                avg_loss = float(np.mean(losses)) if losses else float("nan")
                best, sweep_results = evaluate_logits(model=model,
                                                       dataloader=val_loader,
                                                       device=device,
                                                       thresholds=thresholds,
                                                       threshold_metric=score_key,
                                                       inference_mode="direct",
                                                       description=f"CL {strategy} mixed val")
                score = float(getattr(best, score_key))
                row = {"task": task.task_id,
                       "task_name": task.name,
                       "epoch_in_task": local_epoch + 1,
                       "global_epoch": global_epoch,
                       "train_loss": avg_loss,
                       "best": _best_to_dict(best)}
                validation_history.append(row)
                logger.log_scalar("cl/train_loss", avg_loss)
                logger.log_scalar(f"cl/val_best_{score_key}", score)
                LOG.info("CL %s task %d epoch %d finished. train loss: %.4f | mixed-val best %s: %.4f at threshold %.2f",
                         strategy, task.task_id, local_epoch + 1, avg_loss, score_key, score, best.threshold)
                if score > best_score:
                    best_score = score
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    best_payload = {"model_state_dict": best_state,
                                    "strategy": strategy,
                                    "best_score": best_score,
                                    "best_threshold": best.threshold,
                                    "task_id": task.task_id,
                                    "epoch_in_task": local_epoch + 1,
                                    "global_epoch": global_epoch,
                                    "validation_best": _best_to_dict(best),
                                    "config": config_to_plain_dict(strategy_config)}
                    best_path = model_folder / f"model_best_{score_key}-{best_score:.4f}.pth"
                    torch.save(best_state, best_path)
                    LOG.info("CL %s validation improved to %.4f. Saved %s", strategy, best_score, best_path)
                _save_cl_checkpoint(model_folder / "last.ckpt",
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    strategy=strategy,
                                    best_score=best_score,
                                    global_epoch=global_epoch,
                                    replay_buffer=replay_buffer,
                                    validation_history=validation_history,
                                    detailed_matrix=detailed_matrix,
                                    completed_task_id=task.task_id,
                                    completed_epoch_in_task=local_epoch + 1)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        replay_buffer.add_many(task.train_indices)
        LOG.info("Replay buffer after task %d: %d/%d samples", task.task_id, len(replay_buffer), replay_buffer.max_size)

        # Detailed CL evaluation: evaluate the current model on each seen task-specific eval subset.
        for past_task in tasks[:task.task_id]:
            if not past_task.eval_indices:
                detailed_matrix.append({"after_task": task.task_id,
                                        "eval_task": past_task.task_id,
                                        "eval_task_name": past_task.name,
                                        "samples": 0,
                                        "best": None})
                continue
            eval_loader = DataLoader(Subset(eval_dataset, past_task.eval_indices),
                                     batch_size=1 if eval_inference_mode == "sliding_window" else strategy_config.trainer.batch_size,
                                     shuffle=False,
                                     num_workers=strategy_config.trainer.num_workers,
                                     worker_init_fn=seed_worker,
                                     collate_fn=eval_collate)
            best_eval, eval_sweep = evaluate_logits(model=model,
                                                     dataloader=eval_loader,
                                                     device=device,
                                                     thresholds=thresholds,
                                                     threshold_metric=score_key,
                                                     inference_mode=eval_inference_mode,
                                                     window_size=window_size,
                                                     window_overlap=window_overlap,
                                                     window_batch_size=window_batch_size,
                                                     description=f"CL {strategy} eval T{past_task.task_id}")
            detailed_matrix.append({"after_task": task.task_id,
                                    "eval_task": past_task.task_id,
                                    "eval_task_name": past_task.name,
                                    "samples": len(past_task.eval_indices),
                                    "best": _best_to_dict(best_eval),
                                    "sweep": _threshold_results_to_dicts(eval_sweep)})
            LOG.info("CL %s after task %d -> eval task %d (%s): best %s %.4f at threshold %.2f",
                     strategy, task.task_id, past_task.task_id, past_task.name, score_key, getattr(best_eval, score_key), best_eval.threshold)
        _save_cl_checkpoint(model_folder / "last.ckpt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            strategy=strategy,
                            best_score=best_score,
                            global_epoch=global_epoch,
                            replay_buffer=replay_buffer,
                            validation_history=validation_history,
                            detailed_matrix=detailed_matrix,
                            completed_task_id=task.task_id,
                            completed_epoch_in_task=int(epochs_per_task))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    last_checkpoint = {"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                       "optimizer_state_dict": optimizer.state_dict(),
                       "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
                       "strategy": strategy,
                       "best_score": best_score,
                       "global_epoch": global_epoch,
                       "replay_indices": replay_buffer.indices,
                       "validation_history": validation_history,
                       "detailed_matrix": detailed_matrix,
                       "completed_task_id": len(tasks),
                       "completed_epoch_in_task": int(epochs_per_task)}
    torch.save(last_checkpoint, model_folder / "last.ckpt")
    if best_payload is not None:
        # Keep a stable alias for downstream evaluation commands.
        torch.save(best_payload["model_state_dict"], model_folder / f"{strategy}_best.pth")

    result = {"strategy": strategy,
              "run_label": run_label,
              "run_id": exp_id,
              "output_folder": str(out_folder),
              "best_checkpoint": str(model_folder / f"{strategy}_best.pth") if best_payload is not None else None,
              "best_score": best_score,
              "metric": score_key,
              "elapsed_seconds": elapsed,
              "tasks": [{"task_id": t.task_id,
                         "name": t.name,
                         "start_year": t.start_year,
                         "end_year": t.end_year,
                         "train_samples": len(t.train_indices),
                         "eval_samples": len(t.eval_indices)} for t in tasks],
              "validation_history": validation_history,
              "detailed_matrix": detailed_matrix,
              "cl_summary": _summarise_cl_matrix(detailed_matrix, metric=score_key)}
    with open(out_folder / "cl_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    LOG.info("CL strategy completed: %s | best %s %.4f | time %.1fs", strategy, score_key, best_score, elapsed)
    return result


def continual_train(config: TrainConfig,
                    activations_json_path: Path,
                    strategies: Sequence[str] = DEFAULT_CL_STRATEGIES,
                    task_year_ranges: Sequence[str] = DEFAULT_TASK_YEAR_RANGES,
                    epochs_per_task: int = 5,
                    replay_buffer_size: int = 100,
                    replay_batch_size: int = 16,
                    uncertainty_subset_fraction: float = 1.0,
                    eval_split: str = "val",
                    eval_inference_mode: str = "direct",
                    window_size: int = 512,
                    window_overlap: int = 128,
                    window_batch_size: int = 1,
                    model_mode: str = "single",
                    ensemble_members: Optional[Sequence[str]] = None,
                    ensemble_method: str = "mean_logit",
                    replay_mode: str = "separate",
                    resume: bool = False,
                    resume_from: Optional[Path] = None,
                    resume_start_task: Optional[int] = None,
                    resume_start_epoch: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run rehearsal-based continual learning for one or more replay strategies."""
    if int(epochs_per_task) <= 0:
        raise ValueError("epochs_per_task must be positive")
    if int(replay_batch_size) < 0:
        raise ValueError("replay_batch_size must be >= 0")
    replay_mode = str(replay_mode or "separate").lower().replace("-", "_")
    if replay_mode not in {"separate", "concat"}:
        raise ValueError("replay_mode must be 'separate' or 'concat'")
    eval_split = str(eval_split or "val").lower()
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")
    eval_inference_mode = str(eval_inference_mode or "direct").lower().replace("-", "_")
    if eval_inference_mode not in {"direct", "sliding_window"}:
        raise ValueError("eval_inference_mode must be direct or sliding_window")

    seed_everything(config.seed, deterministic=True)
    use_rgb = (config.data.in_channels - int(config.data.include_dem)) == 3
    from floods.prepare import prepare_datasets
    train_dataset, valid_dataset = prepare_datasets(config=config, use_rgb=use_rgb)
    normalization = valid_dataset.normalization
    scoring_dataset = clone_train_as_scoring_dataset(config, train_dataset, use_rgb=use_rgb, normalization=normalization)
    eval_dataset = valid_dataset if eval_split == "val" else prepare_eval_split_dataset(config, split="test", use_rgb=use_rgb, normalization=normalization)
    event_years = load_event_years(Path(activations_json_path))
    year_ranges = parse_year_ranges(task_year_ranges)
    tasks = build_cl_tasks(train_dataset=train_dataset, eval_dataset=eval_dataset, event_years=event_years, year_ranges=year_ranges)

    LOG.info("Continual-learning tasks:")
    for task in tasks:
        LOG.info("  Task %d (%s): train=%d | %s=%d", task.task_id, task.name, len(task.train_indices), eval_split, len(task.eval_indices))
    if not any(t.train_indices for t in tasks):
        raise ValueError("No training samples matched the requested task year ranges. Check --activations-json-path and --task-year-ranges.")

    output_root = Path(config.output_folder)
    results: List[Dict[str, Any]] = []
    model_mode = str(model_mode or "single").lower()
    for strategy in strategies:
        if model_mode == "ensemble":
            member_results: List[Dict[str, Any]] = []
            for member_text in (ensemble_members or ["unet:resnet50", "deeplabv3p:resnet50"]):
                decoder, encoder, label = _parse_ensemble_member(member_text)
                member_config = copy.deepcopy(config)
                member_config.model.decoder = decoder
                member_config.model.encoder = encoder
                member_config.name = config.name or "continual_learning"
                LOG.info("CL ensemble member started: strategy=%s member=%s decoder=%s encoder=%s", strategy, label, decoder, encoder)
                member_results.append(run_single_strategy(config=member_config,
                                                          strategy=strategy,
                                                          tasks=tasks,
                                                          train_dataset=train_dataset,
                                                          scoring_dataset=scoring_dataset,
                                                          mixed_val_dataset=valid_dataset,
                                                          eval_dataset=eval_dataset,
                                                          output_root=output_root,
                                                          epochs_per_task=epochs_per_task,
                                                          replay_buffer_size=replay_buffer_size,
                                                          replay_batch_size=replay_batch_size,
                                                          uncertainty_subset_fraction=uncertainty_subset_fraction,
                                                          eval_split=eval_split,
                                                          eval_inference_mode=eval_inference_mode,
                                                          window_size=window_size,
                                                          window_overlap=window_overlap,
                                                          window_batch_size=window_batch_size,
                                                          run_label=label,
                                                          replay_mode=replay_mode,
                                                          resume=resume,
                                                          resume_from=resume_from,
                                                          resume_start_task=resume_start_task,
                                                          resume_start_epoch=resume_start_epoch))
            results.append({"strategy": strategy,
                            "model_mode": "ensemble",
                            "ensemble_method": ensemble_method,
                            "members": member_results})
        else:
            result = run_single_strategy(config=config,
                                         strategy=strategy,
                                         tasks=tasks,
                                         train_dataset=train_dataset,
                                         scoring_dataset=scoring_dataset,
                                         mixed_val_dataset=valid_dataset,
                                         eval_dataset=eval_dataset,
                                         output_root=output_root,
                                         epochs_per_task=epochs_per_task,
                                         replay_buffer_size=replay_buffer_size,
                                         replay_batch_size=replay_batch_size,
                                         uncertainty_subset_fraction=uncertainty_subset_fraction,
                                         eval_split=eval_split,
                                         eval_inference_mode=eval_inference_mode,
                                         window_size=window_size,
                                         window_overlap=window_overlap,
                                         window_batch_size=window_batch_size,
                                         replay_mode=replay_mode,
                                         resume=resume,
                                         resume_from=resume_from,
                                         resume_start_task=resume_start_task,
                                         resume_start_epoch=resume_start_epoch)
            results.append(result)
    return results
