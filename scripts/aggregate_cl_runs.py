#!/usr/bin/env python
"""Combine several continual-learning runs (seeds, buffer sizes) into one table.

Each run is given as LABEL=RUN_ID, where RUN_ID is the --run-id passed to
``floodmap continual-train``; the script reads ``<runs-dir>/<RUN_ID>_cl_comparison.csv``.
Runs sharing a label are treated as repeats (e.g. seeds) and reported as mean ± std.

Example:
    python scripts/aggregate_cl_runs.py --runs-dir runs \\
        --run seeds=cl_region_r34 --run seeds=cl_region_r34_s1 --run seeds=cl_region_r34_s2 \\
        --run buffer50=cl_region_r34_buf50 --run buffer150=cl_region_r34_buf150 \\
        --output runs/cl_region_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

METRICS = ("mean_final_task_score", "mean_plasticity", "mean_forgetting", "max_forgetting", "backward_transfer")
SHORT = {"mean_final_task_score": "final F1", "mean_plasticity": "plasticity", "mean_forgetting": "forgetting",
         "max_forgetting": "max forgetting", "backward_transfer": "BWT"}


def _read(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fmt(values):
    if not values:
        return "–"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {statistics.stdev(values):.4f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_ID")
    parser.add_argument("--output", type=Path, help="Optional CSV with one row per label and strategy.")
    args = parser.parse_args(argv)

    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    missing = []
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run expects LABEL=RUN_ID, got {spec!r}")
        label, run_id = spec.split("=", 1)
        path = args.runs_dir / f"{run_id}_cl_comparison.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        for row in _read(path):
            key = (label, row["strategy"])
            entry = groups.setdefault(key, {"runs": [], **{m: [] for m in METRICS}})
            entry["runs"].append(run_id)
            for metric in METRICS:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    entry[metric].append(value)

    for path in missing:
        print(f"WARNING: not found (run not finished?): {path}", file=sys.stderr)
    if not groups:
        print("No results found.", file=sys.stderr)
        return 1

    header = ["label", "strategy", "n"] + [SHORT[m] for m in METRICS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    rows = []
    for (label, strategy), entry in groups.items():
        n = len(entry["runs"])
        print("| " + " | ".join([label, strategy, str(n)] + [_fmt(entry[m]) for m in METRICS]) + " |")
        row = {"label": label, "strategy": strategy, "n": n, "runs": ";".join(entry["runs"])}
        for metric in METRICS:
            values = entry[metric]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else ""
        rows.append(row)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWritten {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
