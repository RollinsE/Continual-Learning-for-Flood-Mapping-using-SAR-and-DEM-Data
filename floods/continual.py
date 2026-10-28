"""Rehearsal-based continual learning over chronological MMFlood tasks.

Protocol (see docs/CONTINUAL_LEARNING.md):

* Tasks are year ranges of EMSR event start dates, trained in order.
* Strategies:
    - ``finetune``          sequential training, no replay (lower bound / forgetting baseline)
    - ``random``            reservoir replay, uniform sampling
    - ``least_confidence``  replay sampled in proportion to mean pixel uncertainty 1 - max(p, 1 - p)
    - ``entropy``           replay sampled in proportion to mean pixel binary entropy
    - ``hybrid``            half the replay batch uncertainty-weighted (entropy), half uniform
    - ``joint``             cumulative training on every task seen so far (upper bound)
  ``margin`` is also accepted; for a single sigmoid output it equals 2 x least_confidence
  and therefore selects the same samples.
* Uncertainty scores are computed once per epoch on non-augmented replay tiles, over
  valid (non-ignore) pixels only.
* After each task the model is evaluated on every task seen so far. One decision threshold
  per evaluation row is chosen on the validation tiles of the seen tasks (or fixed with
  ``--cl-threshold``) and applied to every cell, so test cells never tune their own threshold.
* No checkpoint selection across tasks: the model at the end of each task is saved and
  evaluated (``after_task_<t>.pth``); validation during training only uses seen tasks.
"""
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

from floods.config import TrainConfig
from floods.eval_collate import pad_segmentation_batch
from floods.evaluation import BinaryThresholdSweep, ThresholdResult
from floods.sliding_window import probability_to_logit, sliding_window_logits
from floods.utils.common import config_to_plain_dict, get_logger, init_experiment, store_config
from floods.utils.console import progress_iter
from floods.utils.ml import seed_everything, seed_worker

LOG = get_logger(__name__)

DEFAULT_CL_STRATEGIES = ("finetune", "random", "least_confidence", "entropy", "hybrid", "joint")
SUPPORTED_CL_STRATEGIES = DEFAULT_CL_STRATEGIES + ("margin",)
UNCERTAINTY_STRATEGIES = {"least_confidence", "entropy", "margin", "hybrid"}
NO_REPLAY_STRATEGIES = {"finetune", "joint"}
DEFAULT_TASK_YEAR_RANGES = ("2014-2017", "2018-2019", "2020-2021")
CL_RESULTS_SCHEMA_VERSION = 2


@dataclass
class CLTask:
    """Container for one continual-learning task (a year range or a group of regions)."""

    task_id: int
    name: str
    start_year: Optional[int]
    end_year: Optional[int]
    train_indices: List[int]
    eval_indices: List[int]
    val_indices: Optional[List[int]] = None
    regions: Optional[Tuple[str, ...]] = None


# --------------------------------------------------------------------------- replay buffer


class ReplayBuffer:
    """Reservoir replay buffer storing indices into the training dataset."""

    def __init__(self, max_size: int, seed: int = 42) -> None:
        self.max_size = int(max_size)
        if self.max_size < 0:
            raise ValueError("replay_buffer_size must be >= 0")
        self.indices: List[int] = []
        self.seen = 0
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def add_many(self, indices: Iterable[int]) -> None:
        for index in indices:
            self.add(int(index))

    def add(self, index: int) -> None:
        self.seen += 1
        if self.max_size == 0:
            return
        if len(self.indices) < self.max_size:
            self.indices.append(index)
            return
        replacement_position = self.rng.randint(0, self.seen - 1)
        if replacement_position < self.max_size:
            self.indices[replacement_position] = index

    def random_sample(self, count: int, exclude: Sequence[int] = ()) -> List[int]:
        pool = [i for i in self.indices if i not in set(exclude)] if exclude else list(self.indices)
        if not pool or count <= 0:
            return []
        return self.rng.sample(pool, min(int(count), len(pool)))

    def state_dict(self) -> Dict[str, Any]:
        return {"indices": list(self.indices), "seen": int(self.seen), "max_size": int(self.max_size),
                "rng_state": self.rng.getstate(), "np_rng_state": self.np_rng.bit_generator.state}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.indices = [int(i) for i in state.get("indices", [])]
        self.seen = max(len(self.indices), int(state.get("seen", len(self.indices)) or len(self.indices)))
        if state.get("rng_state") is not None:
            rng_state = state["rng_state"]
            if isinstance(rng_state, list):
                rng_state = (rng_state[0], tuple(rng_state[1]), rng_state[2])
            self.rng.setstate(rng_state)
        if state.get("np_rng_state") is not None:
            self.np_rng.bit_generator.state = state["np_rng_state"]


# --------------------------------------------------------------------------- tasks


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
    for (a_start, a_end), (b_start, b_end) in zip(sorted(ranges), sorted(ranges)[1:]):
        if b_start <= a_end:
            raise ValueError(f"CL task year ranges overlap: {a_start}-{a_end} and {b_start}-{b_end}")
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
        year = event_years.get(event_id_from_path(image_path))
        if year is not None and start_year <= year <= end_year:
            indices.append(idx)
    return indices


def _log_unassigned(dataset: Any, event_years: Dict[str, int], year_ranges: Sequence[Tuple[int, int]], label: str) -> None:
    files = _dataset_file_list(dataset)
    unassigned = set()
    count = 0
    for image_path in files:
        event = event_id_from_path(image_path)
        year = event_years.get(event)
        if year is None or not any(start <= year <= end for start, end in year_ranges):
            unassigned.add(event)
            count += 1
    if count:
        LOG.warning("%s: %d/%d tiles from %d events fall outside every CL task (or have no start year) and are not used: %s",
                    label, count, len(files), len(unassigned), ", ".join(sorted(unassigned)[:15]) + (" ..." if len(unassigned) > 15 else ""))


def build_cl_tasks(train_dataset: Any,
                   eval_dataset: Any,
                   event_years: Dict[str, int],
                   year_ranges: Sequence[Tuple[int, int]],
                   val_dataset: Any = None) -> List[CLTask]:
    """Create chronological tasks for train, evaluation and (optionally) validation datasets."""
    tasks: List[CLTask] = []
    for task_number, (start_year, end_year) in enumerate(year_ranges, start=1):
        name = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
        tasks.append(CLTask(task_id=task_number,
                            name=name,
                            start_year=start_year,
                            end_year=end_year,
                            train_indices=indices_for_year_range(train_dataset, event_years, start_year, end_year),
                            eval_indices=indices_for_year_range(eval_dataset, event_years, start_year, end_year),
                            val_indices=(indices_for_year_range(val_dataset, event_years, start_year, end_year)
                                         if val_dataset is not None else None)))
    return tasks


def indices_for_regions(dataset: Any, event_regions: Dict[str, str], regions: Sequence[str]) -> List[int]:
    """Return dataset indices whose event lies in one of ``regions``."""
    wanted = set(regions)
    return [idx for idx, image_path in enumerate(_dataset_file_list(dataset))
            if event_regions.get(event_id_from_path(image_path)) in wanted]


def build_region_tasks(train_dataset: Any,
                       eval_dataset: Any,
                       event_regions: Dict[str, str],
                       region_tasks: Sequence[Tuple[str, Sequence[str]]],
                       val_dataset: Any = None) -> List[CLTask]:
    """Create geographic tasks (e.g. europe -> asia -> americas+africa+oceania)."""
    tasks: List[CLTask] = []
    for task_number, (name, regions) in enumerate(region_tasks, start=1):
        tasks.append(CLTask(task_id=task_number, name=name, start_year=None, end_year=None,
                            train_indices=indices_for_regions(train_dataset, event_regions, regions),
                            eval_indices=indices_for_regions(eval_dataset, event_regions, regions),
                            val_indices=(indices_for_regions(val_dataset, event_regions, regions)
                                         if val_dataset is not None else None),
                            regions=tuple(regions)))
    return tasks


def _task_events(dataset: Any, indices: Sequence[int]) -> List[str]:
    files = _dataset_file_list(dataset)
    return sorted({event_id_from_path(files[i]) for i in indices})


def _log_region_unassigned(dataset: Any, event_regions: Dict[str, str], region_tasks: Sequence[Tuple[str, Sequence[str]]],
                           event_countries: Dict[str, str], label: str) -> None:
    used = {r for _, regions in region_tasks for r in regions}
    missing: Dict[str, int] = {}
    count = 0
    files = _dataset_file_list(dataset)
    for image_path in files:
        event = event_id_from_path(image_path)
        if event_regions.get(event) not in used:
            missing[event] = missing.get(event, 0) + 1
            count += 1
    if count:
        detail = ", ".join(f"{e} ({event_countries.get(e, '?')}, {event_regions.get(e, 'no region')})"
                           for e in sorted(missing)[:15])
        LOG.warning("%s: %d/%d tiles from %d events are outside every region task and are not used: %s%s",
                    label, count, len(files), len(missing), detail, " ..." if len(missing) > 15 else "")


# --------------------------------------------------------------------------- datasets


def make_scoring_dataset(train_dataset: Any) -> Any:
    """Return a non-augmented view of the training dataset for uncertainty scoring.

    A shallow copy keeps the same file lists, modalities and normalisation, but drops
    augmentation, crop supervision and modality dropout so scores are deterministic.
    """
    scoring = copy.copy(train_dataset)
    for attr in ("transform_base", "transform_sar", "transform_dem",
                 "sparse_crop_supervision", "hard_negative_crop_supervision", "hard_positive_crop_supervision"):
        if hasattr(scoring, attr):
            setattr(scoring, attr, None)
    if hasattr(scoring, "modality_dropout_prob"):
        scoring.modality_dropout_prob = 0.0
        scoring.modality_dropout_indices = tuple()
    return scoring


def collate_replay_samples(dataset: Any, indices: Sequence[int], device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load replay samples by index (with training augmentation) and stack them."""
    if not indices:
        return None, None
    images: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for index in indices:
        sample = dataset[int(index)]
        image, label = sample[:2]
        images.append(torch.as_tensor(image).float())
        labels.append(torch.as_tensor(label).long())
    return torch.stack(images).to(device), torch.stack(labels).to(device)


def _clean_input(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)


# --------------------------------------------------------------------------- replay selection


def _main_logits(output: Any) -> torch.Tensor:
    return BinaryThresholdSweep._main_prediction(output)


def pixel_uncertainty(prob: torch.Tensor, strategy: str) -> torch.Tensor:
    """Per-pixel uncertainty for a binary sigmoid output (higher = more uncertain)."""
    strategy = str(strategy).lower()
    if strategy in {"entropy", "hybrid"}:
        p = prob.clamp(1e-6, 1.0 - 1e-6)
        return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))
    if strategy == "margin":
        return 1.0 - torch.abs(2.0 * prob - 1.0)
    if strategy == "least_confidence":
        return 1.0 - torch.maximum(prob, 1.0 - prob)
    raise ValueError(f"Strategy '{strategy}' has no uncertainty score")


def score_uncertainty(model: nn.Module,
                      scoring_dataset: Any,
                      candidate_indices: Sequence[int],
                      strategy: str,
                      device: torch.device,
                      batch_size: int,
                      ignore_index: int = 255) -> Dict[int, float]:
    """Mean pixel uncertainty per candidate tile, over valid (non-ignore) pixels only."""
    if not candidate_indices:
        return {}
    loader = DataLoader(Subset(scoring_dataset, list(candidate_indices)),
                        batch_size=max(1, int(batch_size)), shuffle=False, num_workers=0)
    scores: Dict[int, float] = {}
    cursor = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[:2]
            logits = _main_logits(model(_clean_input(x).to(device)))
            prob = torch.sigmoid(logits.float())
            if prob.ndim == 4 and prob.shape[1] == 1:
                prob = prob[:, 0]
            valid = (y.to(device) != ignore_index)
            if valid.ndim == 4:
                valid = valid[:, 0]
            uncertainty = pixel_uncertainty(prob, strategy) * valid
            counts = valid.flatten(1).sum(dim=1).clamp_min(1)
            batch_scores = uncertainty.flatten(1).sum(dim=1) / counts
            batch_scores = torch.where(valid.flatten(1).any(dim=1), batch_scores, torch.zeros_like(batch_scores))
            for value in batch_scores.detach().cpu().tolist():
                scores[int(candidate_indices[cursor])] = float(value)
                cursor += 1
    if was_training:
        model.train()
    return scores


def _weighted_sample(candidates: Sequence[int], scores: Dict[int, float], count: int,
                     rng: np.random.Generator, mode: str = "proportional") -> List[int]:
    candidates = list(candidates)
    count = min(int(count), len(candidates))
    if count <= 0:
        return []
    values = np.asarray([max(float(scores.get(i, 0.0)), 0.0) for i in candidates], dtype=np.float64)
    if mode == "topk":
        order = np.argsort(-values, kind="stable")[:count]
        return [candidates[i] for i in order]
    weights = values + 1e-6
    weights = weights / weights.sum()
    chosen = rng.choice(len(candidates), size=count, replace=False, p=weights)
    return [candidates[i] for i in chosen]


def sample_replay_indices(buffer: ReplayBuffer,
                          strategy: str,
                          scores: Optional[Dict[int, float]],
                          replay_batch_size: int,
                          sampling: str = "proportional") -> List[int]:
    """Draw one replay batch from the buffer for the given strategy."""
    strategy = str(strategy).lower()
    if strategy in NO_REPLAY_STRATEGIES or len(buffer) == 0 or replay_batch_size <= 0:
        return []
    if strategy == "random" or not scores:
        return buffer.random_sample(replay_batch_size)
    candidates = [i for i in buffer.indices if i in scores]
    if strategy == "hybrid":
        n_uncertain = (int(replay_batch_size) + 1) // 2
        picked = _weighted_sample(candidates, scores, n_uncertain, buffer.np_rng, mode=sampling)
        return picked + buffer.random_sample(int(replay_batch_size) - len(picked), exclude=picked)
    return _weighted_sample(candidates, scores, replay_batch_size, buffer.np_rng, mode=sampling)


# --------------------------------------------------------------------------- evaluation


def _forward_logits(models: Sequence[nn.Module], x: torch.Tensor, mode: str, window_size: int,
                    window_overlap: int, window_batch_size: int, ensemble_method: str = "mean_logit") -> torch.Tensor:
    """Return B x H x W logits for one model or a mean ensemble."""
    outputs = []
    for model in models:
        if mode == "sliding_window":
            per_image = [sliding_window_logits(model, x[i:i + 1], window_size=int(window_size), overlap=int(window_overlap),
                                               window_batch_size=int(window_batch_size)) for i in range(x.shape[0])]
            logits = torch.cat(per_image, dim=0)
        else:
            logits = _main_logits(model(x))
            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = logits[:, 0]
        outputs.append(logits.float())
    if len(outputs) == 1:
        return outputs[0]
    if ensemble_method == "mean_prob":
        return probability_to_logit(torch.stack([torch.sigmoid(o) for o in outputs]).mean(dim=0))
    return torch.stack(outputs).mean(dim=0)


def evaluate_logits(model: nn.Module | Sequence[nn.Module],
                    dataloader: DataLoader,
                    device: torch.device,
                    thresholds: Sequence[float],
                    threshold_metric: str,
                    inference_mode: str = "direct",
                    window_size: int = 512,
                    window_overlap: int = 128,
                    window_batch_size: int = 1,
                    description: str = "Evaluate",
                    ensemble_method: str = "mean_logit") -> Tuple[ThresholdResult, List[ThresholdResult]]:
    """Evaluate a model (or ensemble) with global pixel metrics over a threshold sweep."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    sweep = BinaryThresholdSweep(thresholds=thresholds, device=device)
    mode = str(inference_mode or "direct").lower().replace("-", "_")
    states = [m.training for m in models]
    for m in models:
        m.eval()
    with torch.no_grad():
        for batch in progress_iter(dataloader, desc=description, unit="batch", colour="green"):
            x, y = batch[:2]
            logits = _forward_logits(models, _clean_input(x).to(device), mode, window_size, window_overlap,
                                     window_batch_size, ensemble_method)
            sweep.update(y.to(device), logits)
    for m, state in zip(models, states):
        m.train(state)
    results = sweep.compute()
    return sweep.best(threshold_metric), results


def result_at_threshold(results: Sequence[ThresholdResult], threshold: float) -> ThresholdResult:
    for result in results:
        if abs(result.threshold - float(threshold)) < 1e-6:
            return result
    raise ValueError(f"Threshold {threshold} is not part of the evaluated sweep")


def _metrics_dict(result: ThresholdResult) -> Dict[str, float]:
    return dict(threshold=result.threshold, f1=result.f1, iou=result.iou, precision=result.precision,
                recall=result.recall, mcc=result.mcc, empty_fp=result.empty_tile_fp_rate,
                nonempty_recall=result.nonempty_tile_recall)


def _threshold_results_to_dicts(results: Sequence[ThresholdResult]) -> List[Dict[str, float]]:
    return [_metrics_dict(r) for r in results]


def _summarise_cl_matrix(detailed_matrix: Sequence[Dict[str, Any]], metric: str = "f1") -> Dict[str, Any]:
    """Average accuracy, plasticity, forgetting and backward transfer from the task matrix.

    All values use the per-row validation-selected threshold (``metrics``), never the
    per-cell oracle threshold. Forgetting and backward transfer exclude the final task,
    which cannot have been forgotten yet.
    """
    by_task: Dict[int, List[Tuple[int, float]]] = {}
    diagonal: Dict[int, float] = {}
    last_after = 0
    for row in detailed_matrix:
        metrics = row.get("metrics") or {}
        if metric not in metrics:
            continue
        after_task = int(row.get("after_task", 0))
        eval_task = int(row.get("eval_task", 0))
        value = float(metrics[metric])
        last_after = max(last_after, after_task)
        by_task.setdefault(eval_task, []).append((after_task, value))
        if after_task == eval_task:
            diagonal[eval_task] = value
    final_values: List[float] = []
    forgetting_values: List[float] = []
    bwt_values: List[float] = []
    per_task: List[Dict[str, float]] = []
    for eval_task, values in sorted(by_task.items()):
        values = sorted(values, key=lambda item: item[0])
        final_after, final_value = values[-1]
        best_seen = max(value for _, value in values)
        final_values.append(final_value)
        entry = {"eval_task": eval_task, "final_after_task": final_after,
                 f"final_{metric}": final_value, f"best_seen_{metric}": best_seen,
                 f"just_learned_{metric}": diagonal.get(eval_task, float("nan"))}
        if eval_task < last_after:
            forgetting_values.append(max(0.0, best_seen - final_value))
            if eval_task in diagonal:
                bwt_values.append(final_value - diagonal[eval_task])
            entry["forgetting"] = max(0.0, best_seen - final_value)
        per_task.append(entry)

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {"metric": metric,
            "threshold_policy": "per evaluation row; selected on validation tiles of seen tasks unless fixed",
            "mean_final_task_score": _mean(final_values),
            "mean_plasticity": _mean(list(diagonal.values())),
            "mean_forgetting": _mean(forgetting_values),
            "max_forgetting": float(np.max(forgetting_values)) if forgetting_values else float("nan"),
            "backward_transfer": _mean(bwt_values),
            "per_task": per_task}


def _loader(dataset: Any, indices: Sequence[int], batch_size: int, num_workers: int, collate=None, shuffle: bool = False) -> DataLoader:
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=seed_worker, collate_fn=collate, drop_last=False)


def _replay_loader(dataset: Any, replay_batches: Sequence[Sequence[int]], num_workers: int) -> DataLoader:
    """Load pre-drawn replay batches in background workers, in order (one batch per current-task step)."""
    return DataLoader(dataset, batch_sampler=[list(map(int, b)) for b in replay_batches],
                      num_workers=num_workers, worker_init_fn=seed_worker)


def _seen(tasks: Sequence[CLTask], upto_task_id: int, attr: str) -> List[int]:
    indices: List[int] = []
    for task in tasks:
        if task.task_id <= upto_task_id:
            indices.extend(getattr(task, attr) or [])
    return indices


def evaluate_after_task(models: Sequence[nn.Module],
                        task_id: int,
                        tasks: Sequence[CLTask],
                        val_dataset: Any,
                        eval_dataset: Any,
                        device: torch.device,
                        thresholds: Sequence[float],
                        score_key: str,
                        batch_size: int,
                        num_workers: int,
                        eval_split: str,
                        eval_inference_mode: str,
                        window_size: int,
                        window_overlap: int,
                        window_batch_size: int,
                        fixed_threshold: Optional[float],
                        label: str,
                        ensemble_method: str = "mean_logit") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Choose one threshold for this row and evaluate every task seen so far."""
    seen_val = _seen(tasks, task_id, "val_indices")
    selection: Dict[str, Any] = {"after_task": task_id}
    if fixed_threshold is not None:
        threshold = float(fixed_threshold)
        selection.update(source="fixed", threshold=threshold)
    elif seen_val:
        best_val, val_sweep = evaluate_logits(models, _loader(val_dataset, seen_val, batch_size, num_workers),
                                              device, thresholds, score_key, description=f"{label} val T1-{task_id}",
                                              ensemble_method=ensemble_method)
        threshold = float(best_val.threshold)
        selection.update(source="val_seen_tasks", threshold=threshold, val_tiles=len(seen_val), val_best=_metrics_dict(best_val))
    else:
        threshold = 0.5
        selection.update(source="default_0.5_no_val_tiles", threshold=threshold)
        LOG.warning("No validation tiles for tasks 1-%d; using threshold 0.5 for this evaluation row.", task_id)

    eval_collate = pad_segmentation_batch if eval_split == "test" or eval_inference_mode == "sliding_window" else None
    eval_batch = 1 if eval_inference_mode == "sliding_window" else batch_size
    rows: List[Dict[str, Any]] = []
    for past_task in tasks[:task_id]:
        base = {"after_task": task_id, "eval_task": past_task.task_id, "eval_task_name": past_task.name,
                "eval_split": eval_split, "threshold": threshold, "threshold_source": selection["source"]}
        if not past_task.eval_indices:
            rows.append({**base, "samples": 0, "metrics": None, "best": None, "oracle_best": None})
            continue
        oracle, sweep = evaluate_logits(models,
                                        _loader(eval_dataset, past_task.eval_indices, eval_batch, num_workers, collate=eval_collate),
                                        device, thresholds, score_key, inference_mode=eval_inference_mode,
                                        window_size=window_size, window_overlap=window_overlap,
                                        window_batch_size=window_batch_size,
                                        description=f"{label} {eval_split} T{past_task.task_id}",
                                        ensemble_method=ensemble_method)
        chosen = _metrics_dict(result_at_threshold(sweep, threshold))
        rows.append({**base, "samples": len(past_task.eval_indices), "metrics": chosen,
                     # "best" holds the same selected-threshold metrics, for readers that expect that key.
                     "best": chosen,
                     # Diagnostic only: threshold tuned on this cell itself (optimistic).
                     "oracle_best": _metrics_dict(oracle),
                     "sweep": _threshold_results_to_dicts(sweep)})
        LOG.info("%s after task %d -> %s task %d (%s): %s %.4f at threshold %.2f (%s)",
                 label, task_id, eval_split, past_task.task_id, past_task.name, score_key,
                 chosen[score_key], threshold, selection["source"])
    return rows, selection


# --------------------------------------------------------------------------- checkpoints


def _rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        # Checkpoints are loaded with map_location=<device>, which moves these
        # ByteTensors to the GPU; the setters only accept CPU uint8 tensors.
        torch.set_rng_state(torch.as_tensor(state["torch"]).cpu().to(torch.uint8))
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([torch.as_tensor(s).cpu().to(torch.uint8) for s in state["cuda"]])
    except Exception as exc:  # pragma: no cover - best effort
        LOG.warning("Could not restore RNG state from checkpoint: %s", exc)


def _checkpoint_model_state(payload: Any) -> Any:
    """Return a model state dict from either a full CL checkpoint or a bare .pth state dict."""
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def _torch_load(path: Path, device: torch.device | str = "cpu") -> Any:
    """Load a checkpoint written by this package (contains RNG/buffer state, not only tensors)."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location=device)


def _load_cl_checkpoint(path: Path, model: nn.Module, optimizer: Any = None, scheduler: Any = None,
                        scaler: Any = None, device: torch.device | None = None) -> Dict[str, Any]:
    """Load a CL checkpoint or bare model weights and return resumable metadata."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CL resume checkpoint not found: {checkpoint_path}")
    payload = _torch_load(checkpoint_path, device or "cpu")
    model.load_state_dict(_checkpoint_model_state(payload), strict=True)
    metadata: Dict[str, Any] = {"source": str(checkpoint_path), "kind": "model_state"}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata["kind"] = "full_checkpoint"
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        for key in ["global_epoch", "replay_buffer", "replay_indices", "validation_history", "detailed_matrix",
                    "threshold_selection", "completed_task_id", "completed_epoch_in_task", "task_end_done",
                    "rng_state", "strategy", "protocol_version"]:
            if key in payload:
                metadata[key] = payload[key]
    return metadata


def _save_cl_checkpoint(path: Path, *, model: nn.Module, optimizer: Any, scheduler: Any, scaler: Any,
                        strategy: str, global_epoch: int, replay_buffer: ReplayBuffer,
                        validation_history: Sequence[Dict[str, Any]], detailed_matrix: Sequence[Dict[str, Any]],
                        threshold_selection: Sequence[Dict[str, Any]], completed_task_id: int,
                        completed_epoch_in_task: int, task_end_done: bool) -> None:
    """Persist a resumable CL checkpoint (written to a temporary file first)."""
    checkpoint = {
        "protocol_version": CL_RESULTS_SCHEMA_VERSION,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "strategy": strategy,
        "global_epoch": int(global_epoch),
        "replay_buffer": replay_buffer.state_dict(),
        "replay_indices": list(replay_buffer.indices),
        "validation_history": list(validation_history),
        "detailed_matrix": list(detailed_matrix),
        "threshold_selection": list(threshold_selection),
        "completed_task_id": int(completed_task_id),
        "completed_epoch_in_task": int(completed_epoch_in_task),
        "task_end_done": bool(task_end_done),
        "rng_state": _rng_state(),
    }
    tmp_path = Path(str(path) + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def _next_position_from_checkpoint(metadata: Dict[str, Any], epochs_per_task: int) -> Tuple[int, int]:
    """Return the one-based task id and epoch-in-task to run next.

    An epoch value of ``epochs_per_task + 1`` means "training for this task is done,
    run only its end-of-task evaluation".
    """
    completed_task_id = int(metadata.get("completed_task_id", 0) or 0)
    completed_epoch = int(metadata.get("completed_epoch_in_task", 0) or 0)
    if completed_task_id <= 0:
        return 1, 1
    if completed_epoch >= int(epochs_per_task):
        if metadata.get("task_end_done", True):
            return completed_task_id + 1, 1
        return completed_task_id, int(epochs_per_task) + 1
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


def _normalise_strategy(strategy: str) -> str:
    strategy = str(strategy).lower().replace("-", "_")
    if strategy not in SUPPORTED_CL_STRATEGIES:
        raise ValueError(f"Unsupported CL strategy '{strategy}'. Choose from {SUPPORTED_CL_STRATEGIES}")
    if strategy == "margin":
        LOG.warning("CL strategy 'margin' equals 2 x least_confidence for a sigmoid output and selects the same replay "
                    "samples. It is kept only so older commands still run; use least_confidence instead.")
    return strategy


# --------------------------------------------------------------------------- training


def run_single_strategy(config: TrainConfig,
                        strategy: str,
                        tasks: Sequence[CLTask],
                        train_dataset: Any,
                        scoring_dataset: Any,
                        val_dataset: Any,
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
                        uncertainty_sampling: str = "proportional",
                        fixed_threshold: Optional[float] = None,
                        resume: bool = False,
                        resume_from: Optional[Path] = None,
                        resume_start_task: Optional[int] = None,
                        resume_start_epoch: Optional[int] = None) -> Dict[str, Any]:
    """Train one CL strategy over all tasks and save its artefacts."""
    strategy = _normalise_strategy(strategy)
    strategy_config = copy.deepcopy(config)
    base_name = strategy_config.name or "continual_learning"
    strategy_suffix = strategy if not run_label else f"{strategy}_{run_label}"
    strategy_config.name = f"{base_name}_{strategy_suffix}"
    strategy_config.output_folder = str(output_root)
    strategy_config.trainer.max_epochs = int(epochs_per_task) * len(tasks)

    exp_id, out_folder, model_folder, logs_folder = init_experiment(strategy_config, log_name="output.log")
    store_config(strategy_config, out_folder / "config.yaml")
    replay_mode = str(replay_mode or "separate").lower().replace("-", "_")
    if replay_mode not in {"separate", "concat"}:
        raise ValueError("replay_mode must be 'separate' or 'concat'")
    uses_replay = strategy not in NO_REPLAY_STRATEGIES
    LOG.info("CL strategy started: %s | output=%s", strategy, out_folder)
    LOG.info("Replay: %s | buffer=%d | replay_batch=%d | subset_fraction=%.3f | sampling=%s | mode=%s",
             "on" if uses_replay else "off", int(replay_buffer_size), int(replay_batch_size),
             float(uncertainty_subset_fraction), uncertainty_sampling, replay_mode)

    device = torch.device("cpu" if strategy_config.trainer.cpu or not torch.cuda.is_available() else "cuda")
    if resume_from is not None or (resume and (model_folder / "last.ckpt").exists()):
        # Trained weights come from the checkpoint; avoid hub downloads on resume.
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

    batch_size = int(strategy_config.trainer.batch_size)
    num_workers = int(strategy_config.trainer.num_workers)
    score_key = str(strategy_config.trainer.threshold_metric or "f1")
    thresholds = sorted(set(float(t) for t in (strategy_config.trainer.thresholds or [0.5])))
    if fixed_threshold is not None and all(abs(t - float(fixed_threshold)) > 1e-6 for t in thresholds):
        thresholds = sorted(thresholds + [float(fixed_threshold)])

    replay_buffer = ReplayBuffer(max_size=replay_buffer_size if uses_replay else 0, seed=strategy_config.seed)
    validation_history: List[Dict[str, Any]] = []
    detailed_matrix: List[Dict[str, Any]] = []
    threshold_selection: List[Dict[str, Any]] = []
    global_epoch = 0
    start_task_id, start_epoch_in_task = 1, 1

    resume_path: Optional[Path] = Path(resume_from) if resume_from else None
    if resume and resume_path is None:
        candidate = model_folder / "last.ckpt"
        if candidate.exists():
            resume_path = candidate
        else:
            # Strategies that had not started when the run was interrupted start fresh.
            LOG.info("No CL checkpoint for %s yet (%s); starting this strategy from the beginning.", strategy, candidate)
    if resume_path is not None:
        metadata = _load_cl_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler,
                                       scaler=scaler, device=device)
        if metadata.get("strategy") not in (None, strategy):
            raise ValueError(f"Checkpoint {resume_path} belongs to strategy '{metadata.get('strategy')}', not '{strategy}'.")
        global_epoch = int(metadata.get("global_epoch", 0) or 0)
        validation_history = list(metadata.get("validation_history") or [])
        detailed_matrix = list(metadata.get("detailed_matrix") or [])
        threshold_selection = list(metadata.get("threshold_selection") or [])
        if metadata.get("replay_buffer"):
            replay_buffer.load_state_dict(metadata["replay_buffer"])
        elif metadata.get("replay_indices"):
            replay_buffer.indices = [int(i) for i in metadata["replay_indices"]]
            replay_buffer.seen = max(len(replay_buffer.indices), sum(len(t.train_indices) for t in tasks
                                                                     if t.task_id <= int(metadata.get("completed_task_id", 0) or 0)))
        _restore_rng_state(metadata.get("rng_state"))
        inferred_task, inferred_epoch = _next_position_from_checkpoint(metadata, int(epochs_per_task))
        start_task_id = int(resume_start_task or inferred_task)
        start_epoch_in_task = int(resume_start_epoch or inferred_epoch)
        if metadata.get("kind") == "model_state":
            if resume_start_task is None:
                raise ValueError("Resuming from a bare model .pth needs --resume-start-task (it holds no progress information).")
            if uses_replay and start_task_id > 1 and not replay_buffer.indices:
                for prior_task in tasks[:start_task_id - 1]:
                    replay_buffer.add_many(prior_task.train_indices)
            global_epoch = (start_task_id - 1) * int(epochs_per_task) + max(0, start_epoch_in_task - 1)
        LOG.info("Resumed CL %s from %s | next task=%d epoch=%d | replay=%d/%d",
                 strategy, resume_path, start_task_id, start_epoch_in_task, len(replay_buffer), replay_buffer.max_size)

    start_time = time.time()
    label = f"CL {strategy}" + (f" [{run_label}]" if run_label else "")

    def save_progress(task_id: int, epoch_in_task: int, task_end_done: bool) -> None:
        _save_cl_checkpoint(model_folder / "last.ckpt", model=model, optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, strategy=strategy, global_epoch=global_epoch, replay_buffer=replay_buffer,
                            validation_history=validation_history, detailed_matrix=detailed_matrix,
                            threshold_selection=threshold_selection, completed_task_id=task_id,
                            completed_epoch_in_task=epoch_in_task, task_end_done=task_end_done)

    for task in tasks:
        if task.task_id < start_task_id:
            continue
        first_epoch = start_epoch_in_task if task.task_id == start_task_id else 1
        train_indices = _seen(tasks, task.task_id, "train_indices") if strategy == "joint" else list(task.train_indices)
        seen_val = _seen(tasks, task.task_id, "val_indices")
        if not train_indices:
            LOG.warning("%s task %d (%s) has no training samples; skipping training for this task.", label, task.task_id, task.name)
        elif first_epoch <= int(epochs_per_task):
            task_loader = _loader(train_dataset, train_indices, batch_size, num_workers, shuffle=True)
            LOG.info("%s task %d/%d (%s): train=%d%s | val_seen=%d | %s=%d | replay=%d",
                     label, task.task_id, len(tasks), task.name, len(train_indices),
                     " (cumulative)" if strategy == "joint" else "", len(seen_val), eval_split,
                     len(task.eval_indices), len(replay_buffer))
            for local_epoch in range(first_epoch - 1, int(epochs_per_task)):
                global_epoch += 1
                scores: Optional[Dict[int, float]] = None
                if strategy in UNCERTAINTY_STRATEGIES and len(replay_buffer) > 0:
                    candidates = list(replay_buffer.indices)
                    fraction = float(uncertainty_subset_fraction)
                    if 0.0 < fraction < 1.0 and len(candidates) > replay_batch_size:
                        count = max(int(replay_batch_size), int(math.ceil(len(candidates) * fraction)))
                        candidates = replay_buffer.rng.sample(candidates, min(count, len(candidates)))
                    scores = score_uncertainty(model, scoring_dataset, candidates, strategy, device, batch_size)
                    if scores:
                        values = np.asarray(list(scores.values()))
                        LOG.info("%s epoch %d replay uncertainty: n=%d mean=%.4f max=%.4f",
                                 label, local_epoch + 1, len(values), float(values.mean()), float(values.max()))
                model.train()
                losses: List[float] = []
                bar = progress_iter(task_loader, desc=f"{label} task {task.task_id}/{len(tasks)} epoch {local_epoch + 1}/{epochs_per_task}",
                                    unit="batch", colour="blue", disable=not strategy_config.trainer.progress_bar)
                # Draw every replay batch for this epoch up front (same RNG sequence as drawing per step),
                # then load them with background workers instead of synchronously in the training loop.
                replay_batches = [sample_replay_indices(replay_buffer, strategy, scores, replay_batch_size, uncertainty_sampling)
                                  for _ in range(len(task_loader))]
                replay_iter = None
                if replay_batches and all(replay_batches):
                    replay_iter = iter(_replay_loader(train_dataset, replay_batches, max(1, int(num_workers) // 2)))
                for step, batch in enumerate(bar):
                    x, y = batch[:2]
                    if replay_iter is not None:
                        replay_batch = next(replay_iter)
                        rx = torch.as_tensor(replay_batch[0]).float().to(device)
                        ry = torch.as_tensor(replay_batch[1]).long().to(device)
                    else:
                        rx, ry = collate_replay_samples(train_dataset, replay_batches[step] if replay_batches else [],
                                                        device=device)
                    x = _clean_input(x).to(device)
                    y = y.long().to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        if rx is not None and replay_mode == "concat":
                            out = model(torch.cat([x, _clean_input(rx)], dim=0))
                            loss = criterion(out, torch.cat([y, ry.long()], dim=0))
                        else:
                            loss = criterion(model(x), y)
                            if rx is not None:
                                replay_loss = criterion(model(_clean_input(rx)), ry.long())
                                n_current, n_replay = int(x.shape[0]), int(rx.shape[0])
                                loss = (loss * n_current + replay_loss * n_replay) / float(n_current + n_replay)
                    if not torch.isfinite(loss.detach()).all():
                        LOG.warning("Skipping non-finite CL loss at task %d epoch %d.", task.task_id, local_epoch + 1)
                        continue
                    scaler.scale(loss).backward()
                    if strategy_config.trainer.grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(strategy_config.trainer.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    loss_value = float(loss.detach().cpu())
                    losses.append(loss_value)
                    bar.set_postfix({"loss": f"{loss_value:.4f}"})
                try:
                    scheduler.step()
                except TypeError:
                    scheduler.step(float(np.mean(losses)) if losses else 0.0)

                avg_loss = float(np.mean(losses)) if losses else float("nan")
                row: Dict[str, Any] = {"task": task.task_id, "task_name": task.name, "epoch_in_task": local_epoch + 1,
                                       "global_epoch": global_epoch, "train_loss": avg_loss,
                                       "replay_scores_mean": float(np.mean(list(scores.values()))) if scores else None}
                if seen_val:
                    best, _ = evaluate_logits(model, _loader(val_dataset, seen_val, batch_size, num_workers), device,
                                              thresholds, score_key, description=f"{label} val seen tasks")
                    row["val_seen_best"] = _metrics_dict(best)
                    logger.log_scalar(f"cl/val_seen_best_{score_key}", float(getattr(best, score_key)))
                    LOG.info("%s task %d epoch %d | loss %.4f | val (tasks 1-%d) best %s %.4f at %.2f",
                             label, task.task_id, local_epoch + 1, avg_loss, task.task_id, score_key,
                             getattr(best, score_key), best.threshold)
                validation_history.append(row)
                logger.log_scalar("cl/train_loss", avg_loss)
                save_progress(task.task_id, local_epoch + 1, task_end_done=False)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # ---- end of task: buffer update, checkpoint, evaluation matrix row
        if uses_replay:
            replay_buffer.add_many(task.train_indices)
            LOG.info("Replay buffer after task %d: %d/%d samples (seen=%d)", task.task_id, len(replay_buffer),
                     replay_buffer.max_size, replay_buffer.seen)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, model_folder / f"after_task_{task.task_id}.pth")
        detailed_matrix = [r for r in detailed_matrix if int(r.get("after_task", 0)) != task.task_id]
        threshold_selection = [r for r in threshold_selection if int(r.get("after_task", 0)) != task.task_id]
        rows, selection = evaluate_after_task([model], task.task_id, tasks, val_dataset, eval_dataset, device, thresholds,
                                              score_key, batch_size, num_workers, eval_split, eval_inference_mode,
                                              window_size, window_overlap, window_batch_size, fixed_threshold, label)
        detailed_matrix.extend(rows)
        threshold_selection.append(selection)
        save_progress(task.task_id, int(epochs_per_task), task_end_done=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(final_state, model_folder / f"{strategy}_final.pth")
    # Also saved under the *_best name used by other commands. There is no cross-task
    # model selection, so this is the model after the final task.
    torch.save(final_state, model_folder / f"{strategy}_best.pth")

    result = {"schema_version": CL_RESULTS_SCHEMA_VERSION,
              "strategy": strategy,
              "run_label": run_label,
              "run_id": exp_id,
              "output_folder": str(out_folder),
              "final_checkpoint": str(model_folder / f"{strategy}_final.pth"),
              "task_checkpoints": {str(t.task_id): str(model_folder / f"after_task_{t.task_id}.pth") for t in tasks},
              "metric": score_key,
              "elapsed_seconds": elapsed,
              "protocol": {"uses_replay": uses_replay, "cumulative_training": strategy == "joint",
                           "replay_buffer_size": int(replay_buffer_size), "replay_batch_size": int(replay_batch_size),
                           "replay_mode": replay_mode, "uncertainty_sampling": uncertainty_sampling,
                           "uncertainty_subset_fraction": float(uncertainty_subset_fraction),
                           "epochs_per_task": int(epochs_per_task), "eval_split": eval_split,
                           "eval_inference_mode": eval_inference_mode,
                           "threshold": ("fixed" if fixed_threshold is not None else "selected on validation tiles of seen tasks after each task"),
                           "fixed_threshold": fixed_threshold, "thresholds": thresholds,
                           "model_selection": "none across tasks; model at the end of each task"},
              "tasks": [{"task_id": t.task_id, "name": t.name, "start_year": t.start_year, "end_year": t.end_year,
                         "regions": list(t.regions) if t.regions else None,
                         "train_samples": len(t.train_indices), "val_samples": len(t.val_indices or []),
                         "eval_samples": len(t.eval_indices)} for t in tasks],
              "threshold_selection": threshold_selection,
              "validation_history": validation_history,
              "detailed_matrix": detailed_matrix,
              "cl_summary": _summarise_cl_matrix(detailed_matrix, metric=score_key),
              "config": config_to_plain_dict(strategy_config)}
    with open(out_folder / "cl_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    summary = result["cl_summary"]
    LOG.info("CL %s completed in %.1fs | final mean %s %.4f | forgetting %.4f | BWT %.4f", strategy, elapsed, score_key,
             summary["mean_final_task_score"], summary["mean_forgetting"], summary["backward_transfer"])
    return result


def evaluate_cl_ensemble(member_results: Sequence[Dict[str, Any]],
                         member_configs: Sequence[TrainConfig],
                         strategy: str,
                         tasks: Sequence[CLTask],
                         val_dataset: Any,
                         eval_dataset: Any,
                         output_root: Path,
                         base_name: str,
                         ensemble_method: str,
                         eval_split: str,
                         eval_inference_mode: str,
                         window_size: int,
                         window_overlap: int,
                         window_batch_size: int,
                         fixed_threshold: Optional[float]) -> Dict[str, Any]:
    """Evaluate trained CL members together, task by task, with the same threshold protocol."""
    from floods.prepare import prepare_model

    config = member_configs[0]
    device = torch.device("cpu" if config.trainer.cpu or not torch.cuda.is_available() else "cuda")
    score_key = str(config.trainer.threshold_metric or "f1")
    thresholds = sorted(set(float(t) for t in (config.trainer.thresholds or [0.5])))
    if fixed_threshold is not None and all(abs(t - float(fixed_threshold)) > 1e-6 for t in thresholds):
        thresholds = sorted(thresholds + [float(fixed_threshold)])
    out_folder = Path(output_root) / f"{base_name}_{strategy}_ensemble"
    out_folder.mkdir(parents=True, exist_ok=True)
    models = []
    for cfg in member_configs:
        cfg = copy.deepcopy(cfg)
        cfg.model.pretrained = False
        models.append(prepare_model(config=cfg, num_classes=1).to(device))
    detailed_matrix: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []
    label = f"CL {strategy} ensemble"
    for task in tasks:
        for model, member in zip(models, member_results):
            state = _torch_load(Path(member["task_checkpoints"][str(task.task_id)]), device)
            model.load_state_dict(_checkpoint_model_state(state), strict=True)
        rows, selection = evaluate_after_task(models, task.task_id, tasks, val_dataset, eval_dataset, device, thresholds,
                                              score_key, int(config.trainer.batch_size), int(config.trainer.num_workers),
                                              eval_split, eval_inference_mode, window_size, window_overlap,
                                              window_batch_size, fixed_threshold, label, ensemble_method=ensemble_method)
        detailed_matrix.extend(rows)
        selections.append(selection)
    result = {"schema_version": CL_RESULTS_SCHEMA_VERSION, "strategy": strategy, "model_mode": "ensemble",
              "ensemble_method": ensemble_method, "metric": score_key,
              "members": [{"run_label": m.get("run_label"), "output_folder": m.get("output_folder")} for m in member_results],
              "threshold_selection": selections, "detailed_matrix": detailed_matrix,
              "cl_summary": _summarise_cl_matrix(detailed_matrix, metric=score_key)}
    with open(out_folder / "cl_ensemble_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    LOG.info("CL %s ensemble | final mean %s %.4f | forgetting %.4f", strategy, score_key,
             result["cl_summary"]["mean_final_task_score"], result["cl_summary"]["mean_forgetting"])
    return result


def continual_train(config: TrainConfig,
                    activations_json_path: Path,
                    strategies: Sequence[str] = DEFAULT_CL_STRATEGIES,
                    task_year_ranges: Sequence[str] = DEFAULT_TASK_YEAR_RANGES,
                    task_split: str = "year",
                    task_regions: Optional[Sequence[str]] = None,
                    dry_run: bool = False,
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
                    uncertainty_sampling: str = "proportional",
                    fixed_threshold: Optional[float] = None,
                    resume: bool = False,
                    resume_from: Optional[Path] = None,
                    resume_start_task: Optional[int] = None,
                    resume_start_epoch: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run continual-learning experiments for one or more strategies."""
    if int(epochs_per_task) <= 0:
        raise ValueError("epochs_per_task must be positive")
    if int(replay_batch_size) < 0:
        raise ValueError("replay_batch_size must be >= 0")
    replay_mode = str(replay_mode or "separate").lower().replace("-", "_")
    if replay_mode not in {"separate", "concat"}:
        raise ValueError("replay_mode must be 'separate' or 'concat'")
    uncertainty_sampling = str(uncertainty_sampling or "proportional").lower()
    if uncertainty_sampling not in {"proportional", "topk"}:
        raise ValueError("uncertainty_sampling must be 'proportional' or 'topk'")
    eval_split = str(eval_split or "val").lower()
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")
    eval_inference_mode = str(eval_inference_mode or "direct").lower().replace("-", "_")
    if eval_inference_mode not in {"direct", "sliding_window"}:
        raise ValueError("eval_inference_mode must be direct or sliding_window")
    if eval_split == "test" and eval_inference_mode == "direct":
        LOG.warning("Test scenes are full rasters; direct inference pads them per batch. "
                    "--cl-eval-inference-mode sliding_window matches the main evaluation protocol.")
    if fixed_threshold is not None and not 0.0 < float(fixed_threshold) < 1.0:
        raise ValueError("--cl-threshold must be between 0 and 1")
    strategies = [_normalise_strategy(s) for s in strategies]
    model_mode = str(model_mode or "single").lower()
    if resume_from is not None and (len(strategies) > 1 or model_mode == "ensemble"):
        raise ValueError("--resume-from points at one run's checkpoint, so it can only be used with a single strategy "
                         "in single-model mode. Use --resume to continue every strategy/member from its own folder.")

    seed_everything(config.seed, deterministic=True)
    from floods.prepare import prepare_datasets, prepare_evaluation_dataset
    from floods.modalities import resolve_input_modalities
    configured = getattr(config.data, "input_modalities", None)
    implicit_rgb = (not configured) and (int(config.data.in_channels) - int(config.data.include_dem)) == 3
    modalities = resolve_input_modalities(configured, in_channels=config.data.in_channels,
                                          include_dem=config.data.include_dem, use_rgb=implicit_rgb)
    use_rgb = modalities[:3] == ["r", "g", "b"]
    train_dataset, val_dataset = prepare_datasets(config=config, use_rgb=use_rgb)
    scoring_dataset = make_scoring_dataset(train_dataset)
    if eval_split == "val":
        eval_dataset = val_dataset
    else:
        eval_config = copy.deepcopy(config)
        eval_dataset, _, _ = prepare_evaluation_dataset(eval_config, split="test")
    task_split = str(task_split or "year").lower()
    event_countries: Dict[str, str] = {}
    if task_split == "region":
        from floods.cl_regions import load_event_regions, parse_region_tasks
        event_regions, event_countries = load_event_regions(Path(activations_json_path))
        region_tasks = parse_region_tasks(task_regions or ["europe", "asia", "americas+africa+oceania"])
        tasks = build_region_tasks(train_dataset=train_dataset, eval_dataset=eval_dataset, event_regions=event_regions,
                                   region_tasks=region_tasks, val_dataset=val_dataset)
        _log_region_unassigned(train_dataset, event_regions, region_tasks, event_countries, "train")
        _log_region_unassigned(val_dataset, event_regions, region_tasks, event_countries, "val")
        if eval_split == "test":
            _log_region_unassigned(eval_dataset, event_regions, region_tasks, event_countries, "test")
    elif task_split == "year":
        event_years = load_event_years(Path(activations_json_path))
        year_ranges = parse_year_ranges(task_year_ranges)
        tasks = build_cl_tasks(train_dataset=train_dataset, eval_dataset=eval_dataset, event_years=event_years,
                               year_ranges=year_ranges, val_dataset=val_dataset)
        _log_unassigned(train_dataset, event_years, year_ranges, "train")
        _log_unassigned(val_dataset, event_years, year_ranges, "val")
        if eval_split == "test":
            _log_unassigned(eval_dataset, event_years, year_ranges, "test")
    else:
        raise ValueError("task_split must be 'year' or 'region'")

    LOG.info("Continual-learning tasks (split=%s, modalities=%s):", task_split, modalities)
    for task in tasks:
        n_train_ev = len(_task_events(train_dataset, task.train_indices))
        n_val_ev = len(_task_events(val_dataset, task.val_indices or []))
        n_eval_ev = len(_task_events(eval_dataset, task.eval_indices))
        LOG.info("  Task %d (%s): train=%d tiles/%d events | val=%d/%d | %s=%d/%d", task.task_id, task.name,
                 len(task.train_indices), n_train_ev, len(task.val_indices or []), n_val_ev,
                 eval_split, len(task.eval_indices), n_eval_ev)
        if task_split == "region":
            countries = sorted({event_countries.get(e, "?") for e in _task_events(train_dataset, task.train_indices)
                                + _task_events(eval_dataset, task.eval_indices)})
            LOG.info("    countries: %s", ", ".join(countries))
        if not task.eval_indices:
            LOG.warning("  Task %d (%s) has no %s samples, so its forgetting cannot be measured; merge it with another "
                        "region.", task.task_id, task.name, eval_split)
        elif n_eval_ev < 3:
            LOG.warning("  Task %d (%s) has only %d %s event(s); its scores will be noisy.", task.task_id, task.name,
                        n_eval_ev, eval_split)
        if not task.val_indices and fixed_threshold is None:
            LOG.warning("  Task %d (%s) has no validation tiles; pass --cl-threshold.", task.task_id, task.name)
    if not any(t.train_indices for t in tasks):
        raise ValueError("No training samples matched the requested CL tasks. Check --activations-json-path and the task options.")
    if dry_run:
        LOG.info("Dry run: task summary only, no training.")
        return []

    output_root = Path(config.output_folder)
    common = dict(tasks=tasks, train_dataset=train_dataset, scoring_dataset=scoring_dataset, val_dataset=val_dataset,
                  eval_dataset=eval_dataset, output_root=output_root, epochs_per_task=epochs_per_task,
                  replay_buffer_size=replay_buffer_size, replay_batch_size=replay_batch_size,
                  uncertainty_subset_fraction=uncertainty_subset_fraction, eval_split=eval_split,
                  eval_inference_mode=eval_inference_mode, window_size=window_size, window_overlap=window_overlap,
                  window_batch_size=window_batch_size, replay_mode=replay_mode, uncertainty_sampling=uncertainty_sampling,
                  fixed_threshold=fixed_threshold, resume=resume, resume_from=resume_from,
                  resume_start_task=resume_start_task, resume_start_epoch=resume_start_epoch)
    results: List[Dict[str, Any]] = []
    for strategy in strategies:
        # Every strategy starts from the same seeded state, so differences come from the strategy.
        seed_everything(config.seed, deterministic=True)
        from floods.utils.ml import reseed_dataset_augmentations
        reseed_dataset_augmentations(train_dataset, int(config.seed))
        if model_mode == "ensemble":
            member_results: List[Dict[str, Any]] = []
            member_configs: List[TrainConfig] = []
            for member_text in (ensemble_members or ["unet:resnet50", "deeplabv3p:resnet50"]):
                decoder, encoder, member_label = _parse_ensemble_member(member_text)
                member_config = copy.deepcopy(config)
                member_config.model.decoder = decoder
                member_config.model.encoder = encoder
                member_config.name = config.name or "continual_learning"
                LOG.info("CL ensemble member started: strategy=%s member=%s", strategy, member_label)
                seed_everything(config.seed, deterministic=True)
                member_results.append(run_single_strategy(config=member_config, strategy=strategy, run_label=member_label, **common))
                member_configs.append(member_config)
            ensemble_result = evaluate_cl_ensemble(member_results, member_configs, strategy, tasks, val_dataset, eval_dataset,
                                                   output_root, config.name or "continual_learning", ensemble_method,
                                                   eval_split, eval_inference_mode, window_size, window_overlap,
                                                   window_batch_size, fixed_threshold)
            results.append({"strategy": strategy, "model_mode": "ensemble", "ensemble_method": ensemble_method,
                            "members": member_results, "ensemble": ensemble_result})
        else:
            results.append(run_single_strategy(config=config, strategy=strategy, **common))
    if results:  # also for a single strategy, so repeated runs can be aggregated the same way
        _write_strategy_comparison(results, output_root, config.name or "continual_learning")
    return results


def _write_strategy_comparison(results: Sequence[Dict[str, Any]], output_root: Path, base_name: str) -> None:
    """Write one CSV comparing strategies on the CL summary metrics."""
    import csv

    rows = []
    for result in results:
        summary = (result.get("ensemble") or result).get("cl_summary") or {}
        rows.append({"strategy": result.get("strategy"), "model_mode": result.get("model_mode", "single"),
                     "metric": summary.get("metric"),
                     "mean_final_task_score": summary.get("mean_final_task_score"),
                     "mean_plasticity": summary.get("mean_plasticity"),
                     "mean_forgetting": summary.get("mean_forgetting"),
                     "max_forgetting": summary.get("max_forgetting"),
                     "backward_transfer": summary.get("backward_transfer")})
    path = Path(output_root) / f"{base_name}_cl_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("CL strategy comparison written to %s", path)
