import pytest

from floods.config import TrainConfig
from floods.sampling_modes import active_sampling_modes, validate_sampling_modes


def test_multiple_training_samplers_are_rejected():
    cfg = TrainConfig()
    cfg.data.event_balanced_sampling = True
    cfg.data.stratified_sampling = True
    assert active_sampling_modes(cfg.data) == ['stratified_sampling', 'event_balanced_sampling']
    with pytest.raises(ValueError, match='Only one training sampler'):
        validate_sampling_modes(cfg.data)


def test_single_training_sampler_is_valid():
    cfg = TrainConfig()
    cfg.data.stratified_sampling = True
    validate_sampling_modes(cfg.data)
