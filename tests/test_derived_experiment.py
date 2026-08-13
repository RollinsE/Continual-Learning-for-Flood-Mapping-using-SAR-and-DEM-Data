import json
from pathlib import Path

import yaml

from floods.config import TrainConfig
from floods.derived_experiment import prepare_derived_experiment_config


def test_prepare_derived_experiment_is_controlled_and_warm_started(tmp_path):
    checkpoint = tmp_path / "baseline.pth"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {
                "channels": [
                    {"channel": name}
                    for name in [
                        "vv",
                        "vh",
                        "dem",
                        "vv_vh_log_ratio",
                        "dem_slope",
                        "dem_tpi",
                    ]
                ]
            }
        ),
        encoding="utf-8",
    )

    config = TrainConfig()
    config.data.in_channels = 3
    config.data.include_dem = True
    config.data.input_modalities = ["vv", "vh", "dem"]
    config.data.event_balanced_sampling = True
    output = tmp_path / "derived.yaml"

    result = prepare_derived_experiment_config(
        config,
        output_config=output,
        run_id="six_channel_test",
        artifacts_dir=tmp_path / "runs",
        baseline_checkpoint=checkpoint,
        normalization_stats_path=stats,
    )

    assert result.data.in_channels == 6
    assert result.data.input_modalities[-3:] == ["vv_vh_log_ratio", "dem_slope", "dem_tpi"]
    assert result.init_channel_adaptation == "zero_extra"
    assert result.init_checkpoint == str(checkpoint)
    assert result.data.event_balanced_sampling is False
    assert result.data.augmentation_profile == "geometric"
    assert result.trainer.amp is False
    assert result.trainer.max_skipped_batch_fraction == 0.02
    saved = yaml.safe_load(output.read_text())
    assert saved["data"]["in_channels"] == 6
