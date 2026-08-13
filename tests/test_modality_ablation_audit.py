import torch

from floods.cli import build_cli_parser
from floods.modality_ablation_audit import _tile_metrics, parse_ablation_spec


def test_parse_modality_ablation_specs_follow_checkpoint_order():
    modalities = ["vv", "vh", "dem"]
    assert parse_ablation_spec("none", modalities) == ("none", tuple(), tuple())
    assert parse_ablation_spec("dem", modalities) == ("dem", (2,), ("dem",))
    assert parse_ablation_spec("vh+vv", modalities) == ("vh+vv", (1, 0), ("vh", "vv"))


def test_parse_modality_ablation_rejects_unknown_channel():
    try:
        parse_ablation_spec("slope", ["vv", "vh", "dem"])
    except ValueError as exc:
        assert "Unknown ablation modalities" in str(exc)
    else:
        raise AssertionError("Expected unknown modality to fail")


def test_tile_metrics_reports_recall_and_false_positives():
    target = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.long)
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    row = _tile_metrics(target, logits, threshold=0.5)[0]
    assert row["tp"] == 1.0
    assert row["fn"] == 1.0
    assert row["fp"] == 1.0
    assert row["recall"] == 0.5
    assert row["precision"] == 0.5
    assert row["f1"] == 0.5


def test_modality_ablation_cli_parses_target_and_combined_ablation():
    parser = build_cli_parser()
    args = parser.parse_args([
        "audit-modality-ablation",
        "--config", "run/config.yaml",
        "--checkpoint", "model.pth",
        "--processed-data-dir", "/tmp/processed",
        "--output-dir", "/tmp/out",
        "--target-events", "EMSR342",
        "--ablations", "none", "dem", "vv+vh",
        "--operating-threshold", "0.55",
        "--no-pretrained",
        "--gpu",
    ])
    assert args.command == "audit-modality-ablation"
    assert args.target_events == ["EMSR342"]
    assert args.ablations == ["none", "dem", "vv+vh"]
    assert args.operating_threshold == 0.55
    assert args.pretrained is False
    assert args.cpu is False
