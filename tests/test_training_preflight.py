from pathlib import Path

import pytest

from floods.config import TrainConfig
from floods.utils.data_preflight import validate_training_data_path


def test_training_preflight_fails_before_run_for_empty_processed_path(tmp_path: Path):
    config = TrainConfig()
    config.data.path = tmp_path / "processed"
    config.data.include_dem = True

    with pytest.raises(FileNotFoundError) as exc:
        validate_training_data_path(config.data)

    message = str(exc.value)
    assert "Processed training dataset is not ready" in message
    assert "train/sar" in message
    assert "val/mask" in message


def test_training_preflight_accepts_required_layout(tmp_path: Path):
    config = TrainConfig()
    config.data.path = tmp_path / "processed"
    config.data.include_dem = True

    for split in ("train", "val"):
        for modality in ("sar", "mask", "dem"):
            folder = Path(config.data.path) / split / modality
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "sample.tif").write_bytes(b"test")

    validate_training_data_path(config.data)
