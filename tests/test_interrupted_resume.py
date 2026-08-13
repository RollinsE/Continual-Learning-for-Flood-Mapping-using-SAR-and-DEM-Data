from pathlib import Path

import torch
from torch import nn

from floods.trainer.base import Trainer


class DummyAccelerator:
    device = torch.device("cpu")
    mixed_precision = "no"

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def wait_for_everyone():
        return None

    @staticmethod
    def save(payload, path):
        torch.save(payload, path)


def _trainer(tmp_path: Path, resume_from: Path | None = None) -> Trainer:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(
        accelerator=DummyAccelerator(),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        criterion=nn.MSELoss(),
        categories={0: "background", 1: "flood"},
        checkpoint_dir=tmp_path,
        resume_from=resume_from,
        save_last=True,
    )


def test_interruption_preserves_last_completed_checkpoint(tmp_path: Path):
    trainer = _trainer(tmp_path)
    trainer.current_epoch = 5
    trainer.last_completed_epoch = 5
    trainer.global_step = 100
    completed = trainer._save_resume_checkpoint()

    trainer.current_epoch = 6
    trainer.global_step = 130
    partial = trainer._save_interrupted_checkpoint()

    assert completed == tmp_path / "last.ckpt"
    assert partial == tmp_path / "interrupted_partial.ckpt"

    completed_payload = torch.load(completed, map_location="cpu", weights_only=False)
    partial_payload = torch.load(partial, map_location="cpu", weights_only=False)
    assert completed_payload["epoch"] == 5
    assert completed_payload["epoch_complete"] is True
    assert partial_payload["epoch"] == 6
    assert partial_payload["epoch_complete"] is False


def test_partial_checkpoint_restarts_interrupted_epoch(tmp_path: Path):
    trainer = _trainer(tmp_path)
    trainer.current_epoch = 6
    trainer.last_completed_epoch = 5
    partial = trainer._save_resume_checkpoint(
        checkpoint_name="interrupted_partial.ckpt",
        epoch_complete=False,
        write_epoch_copy=False,
    )

    resumed = _trainer(tmp_path, resume_from=partial)
    resumed._load_resume_checkpoint()

    assert resumed.start_epoch == 6
    assert resumed.last_completed_epoch == 5


def test_first_epoch_partial_checkpoint_restarts_epoch_one(tmp_path: Path):
    trainer = _trainer(tmp_path)
    trainer.current_epoch = 0
    trainer.last_completed_epoch = -1
    partial = trainer._save_interrupted_checkpoint()

    assert partial == tmp_path / "last.ckpt"
    payload = torch.load(partial, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 0
    assert payload["epoch_complete"] is False

    resumed = _trainer(tmp_path, resume_from=partial)
    resumed._load_resume_checkpoint()
    assert resumed.start_epoch == 0
    assert resumed.last_completed_epoch == -1
