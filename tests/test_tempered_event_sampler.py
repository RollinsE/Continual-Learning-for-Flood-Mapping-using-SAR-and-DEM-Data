from pathlib import Path

import numpy as np

from floods.config.training import DatasetConfig


def test_tempered_sampler_config_fields_parse():
    cfg = DatasetConfig(event_balanced_sampling=True, event_balance_power=0.5, event_tile_weight_cap=5.0)
    assert cfg.event_balanced_sampling is True
    assert cfg.event_balance_power == 0.5
    assert cfg.event_tile_weight_cap == 5.0


def test_tempered_event_mass_reduces_small_event_extremity():
    sizes = np.asarray([1.0, 4.0, 100.0])
    equal_mass_per_tile = (sizes ** 0.0) / sizes
    tempered_mass_per_tile = (sizes ** 0.5) / sizes
    assert equal_mass_per_tile[0] / equal_mass_per_tile[-1] == 100.0
    assert np.isclose(tempered_mass_per_tile[0] / tempered_mass_per_tile[-1], 10.0)
