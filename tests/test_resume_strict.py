from pathlib import Path

import pytest

from floods.config import TrainConfig
from floods.resume import resolve_resume_checkpoint


def test_resume_missing_checkpoint_fails_instead_of_restarting(tmp_path: Path):
    cfg = TrainConfig()
    cfg.resume = True

    with pytest.raises(FileNotFoundError, match="Resume was requested"):
        resolve_resume_checkpoint(cfg, tmp_path / "models")


def test_resume_uses_last_checkpoint(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    checkpoint = model_dir / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    cfg = TrainConfig()
    cfg.resume = True

    assert resolve_resume_checkpoint(cfg, model_dir) == checkpoint


def test_resume_falls_back_to_latest_epoch_checkpoint(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "epoch_002.ckpt").write_bytes(b"old")
    latest = model_dir / "epoch_011.ckpt"
    latest.write_bytes(b"new")

    cfg = TrainConfig()
    cfg.resume = True

    assert resolve_resume_checkpoint(cfg, model_dir) == latest


def test_explicit_resume_path_is_strict(tmp_path: Path):
    cfg = TrainConfig()
    cfg.resume_from = str(tmp_path / "missing.ckpt")

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        resolve_resume_checkpoint(cfg, tmp_path / "models")
