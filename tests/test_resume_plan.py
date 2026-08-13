from pathlib import Path

import pytest
import yaml

from floods.cli import _load_training_config_for_args, build_cli_parser
from floods.config import TrainConfig
from floods.resume import build_resume_signature, diff_resume_signatures
from floods.utils.common import config_to_plain_dict


def _args(config_path: Path, output: Path, run_id: str, extra=None):
    argv = [
        "train",
        "--config", str(config_path),
        "--artifacts-dir", str(output),
        "--run-id", run_id,
        "--resume",
        "--num-workers", "2",
        "--gpu",
        "--progress",
    ]
    if extra:
        argv.extend(extra)
    return build_cli_parser().parse_args(argv)


def test_resume_loads_authoritative_run_config(tmp_path: Path):
    base = TrainConfig()
    base.output_folder = str(tmp_path)
    base.name = "baseline"
    base.trainer.max_epochs = 30

    run = TrainConfig()
    run.output_folder = str(tmp_path)
    run.name = "hardneg"
    run.trainer.max_epochs = 8
    run.trainer.patience = 4
    run.data.hard_negative_region_sampling = True
    run.data.event_balanced_sampling = False
    run.data.hard_negative_manifest = "/tmp/regions.csv"
    run_dir = tmp_path / "hardneg"
    run_dir.mkdir()
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(config_to_plain_dict(base), sort_keys=False))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_plain_dict(run), sort_keys=False))

    loaded = _load_training_config_for_args(_args(tmp_path / "base.yaml", tmp_path, "hardneg"))
    assert loaded.trainer.max_epochs == 8
    assert loaded.trainer.patience == 4
    assert loaded.data.hard_negative_region_sampling is True
    assert loaded.data.hard_negative_manifest == "/tmp/regions.csv"
    assert loaded.trainer.num_workers == 2
    assert loaded.trainer.cpu is False


def test_resume_rejects_epoch_target_change(tmp_path: Path):
    run = TrainConfig()
    run.output_folder = str(tmp_path)
    run.name = "hardneg"
    run.trainer.max_epochs = 8
    run_dir = tmp_path / "hardneg"
    run_dir.mkdir()
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_to_plain_dict(run), sort_keys=False))

    with pytest.raises(ValueError, match="change the saved training plan"):
        _load_training_config_for_args(_args(config_path, tmp_path, "hardneg", ["--epochs", "30"]))



def test_resume_clears_saved_init_checkpoint_after_plan_validation(tmp_path: Path):
    run = TrainConfig()
    run.output_folder = str(tmp_path)
    run.name = "hardpos"
    run.init_checkpoint = "/tmp/baseline_best.pth"
    run.trainer.max_epochs = 20
    run.data.hard_example_sampling = True
    run.data.event_balanced_sampling = False
    run_dir = tmp_path / "hardpos"
    run_dir.mkdir()
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_to_plain_dict(run), sort_keys=False))

    loaded = _load_training_config_for_args(_args(config_path, tmp_path, "hardpos"))

    assert loaded.resume is True
    assert loaded.init_checkpoint is None
    persisted = _load_training_config_for_args.__globals__["_load_config_from_yaml"](config_path, TrainConfig)
    assert persisted.init_checkpoint == "/tmp/baseline_best.pth"

def test_resume_signature_detects_sampler_change():
    saved = TrainConfig()
    saved.data.hard_negative_region_sampling = True
    saved.data.event_balanced_sampling = False
    saved.data.hard_negative_manifest = "/tmp/regions.csv"
    changed = saved.copy(deep=True)
    changed.data.hard_negative_region_sampling = False
    changed.data.event_balanced_sampling = True
    differences = diff_resume_signatures(build_resume_signature(saved), build_resume_signature(changed))
    assert any("sampler" in item for item in differences)
    assert any("hard_negative_region_sampling" in item for item in differences)
