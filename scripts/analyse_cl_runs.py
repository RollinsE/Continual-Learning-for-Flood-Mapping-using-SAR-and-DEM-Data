#!/usr/bin/env python
"""Analyse finished continual-learning runs.

Two analyses, neither of which retrains anything:

``metrics`` (no GPU)
    Precision, recall and F1 per strategy and task, read from ``cl_results.json``:
    after each task's own training and after the last task, so forgetting can be split
    into lost recall (missed flood) and lost precision (false flood). Training time per
    strategy is read from the run logs.

``replay`` (GPU recommended)
    What each replay strategy actually replays. The replay buffer is rebuilt exactly as
    during training (same reservoir order and seed), every buffered tile is scored with
    the model saved at the start and at the end of the task in which it was replayed, and
    the sampling weights of each strategy are applied. The output is the expected flood
    content of a replay draw: mean flood-pixel fraction and the share of draws containing
    any flood, compared with uniform (random) replay.

Examples:
    python scripts/analyse_cl_runs.py metrics --runs-dir runs --run cl_region_r34 cl_region_r34_s1 cl_r34
    python scripts/analyse_cl_runs.py replay  --runs-dir runs --run cl_region_r34 cl_r34 --gpu
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

STRATEGIES = ("finetune", "random", "least_confidence", "margin", "entropy", "hybrid", "joint")
REPLAY_STRATEGIES = ("random", "least_confidence", "margin", "entropy", "hybrid")


def strategy_folders(runs_dir: Path, run_id: str) -> Iterable[Tuple[str, Path]]:
    for strategy in STRATEGIES:
        folder = runs_dir / f"{run_id}_{strategy}"
        if (folder / "cl_results.json").is_file():
            yield strategy, folder


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten {path}")


def print_table(rows: List[dict], columns: List[str]) -> None:
    def fmt(value):
        if value is None or value == "":
            return "–"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")


# --------------------------------------------------------------------------- metrics


def training_seconds(runs_dir: Path, run_id: str) -> Dict[str, float]:
    """Longest 'CL <strategy> completed in Xs' per strategy in the run log (resumed runs log ~0 s)."""
    seconds: Dict[str, float] = {}
    for log in [runs_dir / f"{run_id}.log"] + sorted(runs_dir.glob(f"{run_id}_*/output.log")):
        if not log.is_file():
            continue
        for match in re.finditer(r"CL (\w+) completed in ([\d.]+)s", log.read_text(encoding="utf-8", errors="ignore")):
            strategy, value = match.group(1), float(match.group(2))
            seconds[strategy] = max(seconds.get(strategy, 0.0), value)
    return seconds


def metrics_analysis(args: argparse.Namespace) -> int:
    per_task: List[dict] = []
    timing: List[dict] = []
    for run_id in args.run:
        seconds = training_seconds(args.runs_dir, run_id)
        found = False
        for strategy, folder in strategy_folders(args.runs_dir, run_id):
            found = True
            result = json.loads((folder / "cl_results.json").read_text(encoding="utf-8"))
            matrix = [row for row in result.get("detailed_matrix", []) if row.get("metrics")]
            if not matrix:
                continue
            last = max(int(row["after_task"]) for row in matrix)
            names = {int(t["task_id"]): t["name"] for t in result.get("tasks", [])}
            own = {int(row["eval_task"]): row["metrics"] for row in matrix if int(row["after_task"]) == int(row["eval_task"])}
            final = {int(row["eval_task"]): row["metrics"] for row in matrix if int(row["after_task"]) == last}
            for task_id in sorted(final):
                start, end = own.get(task_id, {}), final[task_id]
                per_task.append({
                    "run": run_id, "strategy": strategy, "task": f"{task_id} {names.get(task_id, '')}".strip(),
                    "f1_own": start.get("f1"), "f1_final": end.get("f1"),
                    "precision_own": start.get("precision"), "precision_final": end.get("precision"),
                    "recall_own": start.get("recall"), "recall_final": end.get("recall"),
                    "recall_change": (end["recall"] - start["recall"]) if start and task_id < last else None,
                    "precision_change": (end["precision"] - start["precision"]) if start and task_id < last else None,
                })
            elapsed = seconds.get(strategy) or float(result.get("elapsed_seconds") or 0.0) or None
            timing.append({"run": run_id, "strategy": strategy, "minutes": (elapsed / 60.0) if elapsed else None,
                           "replay_buffer_tiles": result.get("protocol", {}).get("replay_buffer_size")
                           if result.get("protocol", {}).get("uses_replay") else 0,
                           "training_tiles": sum(int(t.get("train_samples", 0)) for t in result.get("tasks", []))})
        if not found:
            print(f"WARNING: no cl_results.json under {args.runs_dir}/{run_id}_<strategy>/", file=sys.stderr)

    for row in timing:
        same_run = [r for r in timing if r["run"] == row["run"] and r["strategy"] == "finetune" and r["minutes"]]
        row["time_vs_finetune"] = (row["minutes"] / same_run[0]["minutes"]) if same_run and row["minutes"] else None
        tiles = row.get("training_tiles") or 0
        row["buffer_share_of_training_tiles"] = (row["replay_buffer_tiles"] / tiles) if tiles and row["replay_buffer_tiles"] else 0.0

    print("Per-class metrics: own = right after training on that task, final = after the last task")
    print_table(per_task, ["run", "strategy", "task", "f1_own", "f1_final", "precision_own", "precision_final",
                           "recall_own", "recall_final", "recall_change", "precision_change"])
    print("\nTraining time (minutes, from the run logs) and replay memory")
    print_table(timing, ["run", "strategy", "minutes", "time_vs_finetune", "replay_buffer_tiles",
                         "buffer_share_of_training_tiles"])
    print("\nTimes are only comparable within one run: different Colab sessions can get different GPUs.")
    if args.output_dir:
        write_csv(args.output_dir / "cl_per_task_metrics.csv", per_task)
        write_csv(args.output_dir / "cl_training_time.csv", timing)
    return 0


# --------------------------------------------------------------------------- replay


def _ranks(values):
    import numpy as np
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(a, b) -> Optional[float]:
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def sampling_weights(strategy: str, entropy, least_confidence):
    """Per-draw selection probabilities used by the replay sampler (proportional mode)."""
    import numpy as np
    n = len(entropy)
    uniform = np.full(n, 1.0 / n)

    def proportional(scores):
        weights = np.clip(np.asarray(scores, dtype=float), 0.0, None) + 1e-6
        return weights / weights.sum()

    if strategy == "random":
        return uniform
    if strategy == "entropy":
        return proportional(entropy)
    if strategy in {"least_confidence", "margin"}:
        return proportional(least_confidence)  # margin = 2 x least_confidence: same weights
    if strategy == "hybrid":
        return 0.5 * proportional(entropy) + 0.5 * uniform  # half the batch weighted, half uniform
    raise ValueError(strategy)


def replay_analysis(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from floods.cl_regions import load_event_regions
    from floods.cli import _load_config_from_yaml
    from floods.config import TrainConfig
    from floods.continual import (ReplayBuffer, _clean_input, _main_logits, build_cl_tasks, build_region_tasks,
                                  load_event_years, make_scoring_dataset, pixel_uncertainty)
    from floods.prepare import prepare_datasets, prepare_model

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    rows: List[dict] = []
    dataset_cache: Dict[tuple, tuple] = {}

    for run_id in args.run:
        for strategy, folder in strategy_folders(args.runs_dir, run_id):
            if strategy not in REPLAY_STRATEGIES:
                continue
            result = json.loads((folder / "cl_results.json").read_text(encoding="utf-8"))
            protocol = result.get("protocol", {})
            buffer_size = int(protocol.get("replay_buffer_size", 0))
            config = _load_config_from_yaml(folder / "config.yaml", TrainConfig)
            config.model.pretrained = False  # weights come from the saved checkpoints
            if args.processed_data_dir:
                config.data.path = str(args.processed_data_dir)

            key = (str(config.data.path), tuple(config.data.input_modalities or []))
            if key not in dataset_cache:
                train_dataset, _ = prepare_datasets(config=config, use_rgb=False)
                dataset_cache[key] = (train_dataset, make_scoring_dataset(train_dataset))
            train_dataset, scoring_dataset = dataset_cache[key]

            task_specs = result["tasks"]
            if task_specs and task_specs[0].get("regions"):
                event_regions, _ = load_event_regions(args.activations_json_path)
                tasks = build_region_tasks(train_dataset, train_dataset, event_regions,
                                           [(t["name"], tuple(t["regions"])) for t in task_specs])
            else:
                event_years = load_event_years(args.activations_json_path)
                tasks = build_cl_tasks(train_dataset, train_dataset, event_years,
                                       [(int(t["start_year"]), int(t["end_year"])) for t in task_specs])
            for task, spec in zip(tasks, task_specs):
                if len(task.train_indices) != int(spec["train_samples"]):
                    raise RuntimeError(f"{run_id}/{strategy}: task {task.task_id} has {len(task.train_indices)} training "
                                       f"tiles here but {spec['train_samples']} during training; use the same processed data.")

            model = prepare_model(config=config, num_classes=1).to(device)
            logged_means = {(int(r["task"]), int(r["epoch_in_task"])): r.get("replay_scores_mean")
                            for r in result.get("validation_history", [])}

            for task in tasks[1:]:
                buffer = ReplayBuffer(buffer_size, seed=int(config.seed))
                for prior in tasks[:task.task_id - 1]:
                    buffer.add_many(prior.train_indices)
                indices = list(buffer.indices)
                for moment, checkpoint in (("start", folder / "models" / f"after_task_{task.task_id - 1}.pth"),
                                           ("end", folder / "models" / f"after_task_{task.task_id}.pth")):
                    if not checkpoint.is_file():
                        print(f"WARNING: missing {checkpoint}; skipped.", file=sys.stderr)
                        continue
                    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    model.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
                    model.eval()
                    entropy, least_conf, flood_frac = [], [], []
                    loader = DataLoader(Subset(scoring_dataset, indices), batch_size=8, shuffle=False, num_workers=2)
                    with torch.no_grad():
                        for batch in loader:
                            x, y = batch[:2]
                            prob = torch.sigmoid(_main_logits(model(_clean_input(x).to(device))).float())
                            if prob.ndim == 4:
                                prob = prob[:, 0]
                            y = y.to(device)
                            if y.ndim == 4:
                                y = y[:, 0]
                            valid = (y != 255)
                            count = valid.flatten(1).sum(1).clamp_min(1)
                            for name, store in (("entropy", entropy), ("least_confidence", least_conf)):
                                score = (pixel_uncertainty(prob, name) * valid).flatten(1).sum(1) / count
                                store.extend(score.cpu().tolist())
                            flood_frac.extend((((y == 1) & valid).flatten(1).sum(1) / count).cpu().tolist())
                    entropy, least_conf, flood_frac = map(np.asarray, (entropy, least_conf, flood_frac))
                    weights = sampling_weights(strategy, entropy, least_conf)
                    uniform = np.full(len(indices), 1.0 / len(indices))
                    own_scores = least_conf if strategy in {"least_confidence", "margin"} else entropy
                    logged = logged_means.get((task.task_id, 1)) if moment == "start" else None
                    rows.append({
                        "run": run_id, "strategy": strategy, "replayed_during_task": f"{task.task_id} {task.name}",
                        "model": moment, "buffer_tiles": len(indices),
                        "buffer_tiles_with_flood": float(np.mean(flood_frac > 0)),
                        "uniform_flood_fraction": float(np.sum(uniform * flood_frac)),
                        "replay_flood_fraction": float(np.sum(weights * flood_frac)),
                        "flood_oversampling": float(np.sum(weights * flood_frac) / max(np.sum(uniform * flood_frac), 1e-12)),
                        "replay_draws_with_flood": float(np.sum(weights * (flood_frac > 0))),
                        "effective_tiles": float(1.0 / np.sum(weights ** 2)),
                        "spearman_uncertainty_vs_flood": spearman(own_scores, flood_frac),
                        "mean_uncertainty": float(own_scores.mean()),
                        "logged_epoch1_mean": logged if strategy != "random" else None,
                    })
                    print(f"{run_id} | {strategy} | task {task.task_id} | {moment} model: replay flood fraction "
                          f"{rows[-1]['replay_flood_fraction']:.4f} vs uniform {rows[-1]['uniform_flood_fraction']:.4f}",
                          flush=True)

    print("\nWhat a replay draw contains (expected values per drawn tile, proportional sampling)")
    print_table(rows, ["run", "strategy", "replayed_during_task", "model", "buffer_tiles_with_flood",
                       "uniform_flood_fraction", "replay_flood_fraction", "flood_oversampling",
                       "replay_draws_with_flood", "effective_tiles", "spearman_uncertainty_vs_flood",
                       "mean_uncertainty", "logged_epoch1_mean"])
    print("\nflood_oversampling > 1 means the strategy replays more flood pixels than uniform replay. "
          "mean_uncertainty for the start model should match logged_epoch1_mean (the value training logged).")
    if args.output_dir:
        write_csv(args.output_dir / "cl_replay_composition.csv", rows)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("metrics", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--runs-dir", type=Path, required=True)
        p.add_argument("--run", nargs="+", required=True, help="CL run ids (the --run-id given to continual-train).")
        p.add_argument("--output-dir", type=Path, help="Write CSV tables here.")
        if name == "replay":
            p.add_argument("--activations-json-path", type=Path, required=True)
            p.add_argument("--processed-data-dir", type=Path, help="Override the data path stored in the run config.")
            p.add_argument("--gpu", action="store_true")
    args = parser.parse_args(argv)
    return metrics_analysis(args) if args.command == "metrics" else replay_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
