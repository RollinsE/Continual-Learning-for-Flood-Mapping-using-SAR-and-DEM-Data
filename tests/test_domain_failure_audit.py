from pathlib import Path

import numpy as np
import pandas as pd

from floods.cli import build_cli_parser
from floods.domain_failure_audit import audit_domain_failure_link


def _write_inputs(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    feature_rows = []
    for index in range(80):
        feature_rows.append(
            {
                "domain": "reference",
                "split": "train",
                "event_id": f"EMSR{100 + index % 4}",
                "file": f"train_{index}.tif",
                "vv_mean": float(rng.normal(0.05, 0.01)),
                "vv_std": float(rng.normal(0.03, 0.004)),
                "dem_p50": float(rng.normal(120, 30)),
                "vh_mean": float(rng.normal(0.012, 0.002)),
                "fg_ratio": float(rng.uniform(0.01, 0.2)),
                "vv_flood_mean": float(rng.normal(0.02, 0.004)),
                "all_modalities_finite_ratio": 1.0,
            }
        )
    error_rows = []
    for index in range(30):
        failure = index < 15
        vv = float(rng.normal(0.075 if failure else 0.045, 0.004))
        dem = float(rng.normal(180 if failure else 80, 10))
        file = f"target_{index}.tif"
        feature_rows.append(
            {
                "domain": "target",
                "split": "val",
                "event_id": "EMSR342",
                "file": file,
                "vv_mean": vv,
                "vv_std": float(rng.normal(0.025, 0.003)),
                "dem_p50": dem,
                "vh_mean": float(rng.normal(0.014, 0.002)),
                "fg_ratio": 0.08,
                "vv_flood_mean": vv * 0.7,
                "all_modalities_finite_ratio": 1.0,
            }
        )
        recall = 0.10 if failure else 0.70
        error_rows.append(
            {
                "event_id": "EMSR342",
                "file": file,
                "recall": recall,
                "iou": recall * 0.7,
                "f1": recall * 0.8,
                "fg_pixels": 1000,
                "error_category": "false_negative_low_recall" if failure else "partial_overlap",
            }
        )
    feature_path = root / "tile_features.csv"
    error_path = root / "tile_error_metrics.csv"
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    pd.DataFrame(error_rows).to_csv(error_path, index=False)
    return feature_path, error_path


def test_domain_failure_audit_links_failures_and_writes_analogues(tmp_path):
    feature_path, error_path = _write_inputs(tmp_path)
    output = tmp_path / "out"
    summary = audit_domain_failure_link(
        feature_path,
        error_path,
        output,
        target_events=["EMSR342"],
        max_recall=0.25,
        neighbours=3,
        write_plots=False,
    )
    assert summary["nonempty_target_tiles"] == 30
    assert summary["failure_tiles"] == 15
    assert summary["deployable_features"] >= 4
    assert summary["failure_classifier_roc_auc"] > 0.9
    assert (output / "feature_performance_correlations.csv").exists()
    assert (output / "failure_training_analogue_pairs.csv").exists()
    pairs = pd.read_csv(output / "failure_training_analogue_pairs.csv")
    assert len(pairs) == 45
    assert set(pairs["neighbour_rank"]) == {1, 2, 3}


def test_domain_failure_cli_parses_inputs():
    parser = build_cli_parser()
    args = parser.parse_args(
        [
            "audit-domain-failure-link",
            "--tile-features-csv",
            "/tmp/features.csv",
            "--tile-error-metrics-csv",
            "/tmp/errors.csv",
            "--output-dir",
            "/tmp/out",
            "--target-events",
            "EMSR342",
            "--max-recall",
            "0.2",
            "--neighbours",
            "7",
            "--no-write-plots",
        ]
    )
    assert args.command == "audit-domain-failure-link"
    assert args.target_events == ["EMSR342"]
    assert args.max_recall == 0.2
    assert args.neighbours == 7
    assert args.write_plots is False
