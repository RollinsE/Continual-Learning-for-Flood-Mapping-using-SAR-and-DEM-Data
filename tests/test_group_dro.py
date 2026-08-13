from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from floods.cli import _apply_training_overrides, _load_config_from_yaml, build_cli_parser
from floods.config import TrainConfig
from floods.group_dro import (
    EventIndexedDataset,
    effective_group_count,
    event_id_from_path,
    group_mean_losses,
    robust_present_group_loss,
    update_group_weights,
)
from floods.resume import build_resume_signature


class TinyDataset(Dataset):
    def __init__(self):
        self.label_files = [
            "/tmp/EMSR001_tile_000.tif",
            "/tmp/EMSR002_tile_000.tif",
            "/tmp/EMSR001_tile_001.tif",
        ]

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        return torch.tensor([float(index)]), torch.tensor(index % 2)

    def categories(self):
        return {0: "background", 1: "flood"}


def test_event_indexed_dataset_preserves_order_and_attributes():
    wrapped = EventIndexedDataset(TinyDataset())
    assert wrapped.event_names == ["EMSR001", "EMSR002"]
    assert wrapped.event_indices == [0, 1, 0]
    x, y, group = wrapped[1]
    assert float(x.item()) == 1.0
    assert int(y.item()) == 1
    assert int(group.item()) == 1
    assert wrapped.categories()[1] == "flood"


def test_event_id_requires_an_emsr_identifier():
    assert event_id_from_path("/tmp/EMSR342-2-5_tile_001.tif") == "EMSR342"
    with pytest.raises(ValueError):
        event_id_from_path("/tmp/tile_001.tif")


def test_group_means_and_robust_loss_are_differentiable():
    losses = torch.tensor([1.0, 3.0, 4.0], requires_grad=True)
    groups = torch.tensor([0, 0, 2])
    means, counts, present = group_mean_losses(losses, groups, 3)
    assert torch.allclose(means, torch.tensor([2.0, 0.0, 4.0]))
    assert torch.allclose(counts, torch.tensor([2.0, 0.0, 1.0]))
    assert present.tolist() == [True, False, True]
    robust = robust_present_group_loss(means, present, torch.tensor([0.25, 0.25, 0.50]))
    assert torch.isclose(robust, torch.tensor(10.0 / 3.0))
    robust.backward()
    assert losses.grad is not None
    assert torch.isfinite(losses.grad).all()


def test_group_dro_update_increases_weight_on_higher_loss_group():
    initial = torch.tensor([0.5, 0.5])
    updated = update_group_weights(
        initial,
        torch.tensor([0.2, 1.2]),
        torch.tensor([True, True]),
        eta=0.5,
        min_weight=0.01,
    )
    assert torch.isclose(updated.sum(), torch.tensor(1.0))
    assert updated[1] > updated[0]
    assert updated.min() >= 0.01
    assert 1.0 <= effective_group_count(updated) <= 2.0


def test_group_dro_absent_group_weight_is_not_directly_penalised():
    initial = torch.tensor([0.2, 0.3, 0.5])
    updated = update_group_weights(
        initial,
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([True, False, False]),
        eta=0.1,
    )
    assert updated[0] > initial[0]
    assert torch.isclose(updated.sum(), torch.tensor(1.0))
    assert updated[1] / updated[2] == pytest.approx(float(initial[1] / initial[2]))


def test_group_dro_cli_and_resume_signature():
    parser = build_cli_parser()
    args = parser.parse_args([
        "train",
        "--config", "configs/train_segmentation_vv_vh_dem.yaml",
        "--group-dro",
        "--group-dro-eta", "0.02",
        "--group-dro-min-weight", "0.002",
        "--group-dro-warmup-epochs", "2",
    ])
    cfg = _load_config_from_yaml(Path(args.config), TrainConfig)
    cfg = _apply_training_overrides(cfg, args)
    assert cfg.trainer.group_dro is True
    assert cfg.trainer.group_dro_eta == pytest.approx(0.02)
    assert cfg.trainer.group_dro_min_weight == pytest.approx(0.002)
    assert cfg.trainer.group_dro_warmup_epochs == 2
    signature = build_resume_signature(cfg)
    assert signature["trainer"]["group_dro"] is True
    assert signature["trainer"]["group_dro_eta"] == pytest.approx(0.02)
