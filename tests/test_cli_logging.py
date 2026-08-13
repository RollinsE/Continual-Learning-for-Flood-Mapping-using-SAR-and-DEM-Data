import logging
from pathlib import Path

from floods.cli import build_cli_parser
from floods.utils.common import command_logging, prepare_logging


def test_runtime_logging_options_are_available_after_subcommand(tmp_path):
    parser = build_cli_parser()
    log_file = tmp_path / "preprocess.log"
    args = parser.parse_args([
        "preprocess",
        "--config", "configs/preprocess_mmflood.yaml",
        "--log-file", str(log_file),
        "--plain-progress",
        "--heartbeat-seconds", "12",
        "--log-level", "DEBUG",
    ])
    assert args.log_file == log_file
    assert args.plain_progress is True
    assert args.heartbeat_seconds == 12
    assert args.log_level == "DEBUG"


def test_amp_float32_retry_cli_reaches_training_config():
    from floods.cli import _apply_training_overrides, _load_config_from_yaml
    from floods.config import TrainConfig

    parser = build_cli_parser()
    args = parser.parse_args([
        "train",
        "--config", "configs/train_segmentation_vv_vh_dem.yaml",
        "--amp",
        "--amp-full-precision-retry",
    ])
    config = _load_config_from_yaml(Path(args.config), TrainConfig)
    config = _apply_training_overrides(config, args)
    assert config.trainer.amp is True
    assert config.trainer.amp_full_precision_retry is True


def test_command_logging_writes_start_and_completion(tmp_path):
    log_path = tmp_path / "output.log"
    prepare_logging("INFO")
    with command_logging(
        "audit-dataset",
        log_file=log_path,
        argv_text="floodmap audit-dataset --help",
        heartbeat_seconds=0,
    ):
        logging.getLogger("test").info("work item")
    text = log_path.read_text(encoding="utf-8")
    assert "Command started | command=audit-dataset" in text
    assert "work item" in text
    assert "Command completed | command=audit-dataset" in text
