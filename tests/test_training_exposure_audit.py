import numpy as np

from floods.training_exposure_audit import _gini, _tempered_event_weights, _top_share


def test_exposure_summary_helpers():
    uniform = np.ones(100)
    concentrated = np.r_[np.zeros(99), 100.0]
    assert abs(_gini(uniform)) < 1e-12
    assert _gini(concentrated) > 0.98
    assert _top_share(uniform, 0.05) == 0.05
    assert _top_share(concentrated, 0.05) == 1.0


def test_tempered_event_weights_reduce_tiny_event_extremes():
    paths = ["EMSR1_a.tif"] + [f"EMSR2_{i}.tif" for i in range(100)]
    ratios = np.zeros(len(paths), dtype=float)
    current = _tempered_event_weights(paths, ratios, [0.0, 0.005, 0.02, 0.1], [0.2]*5, 0.0, None)
    tempered = _tempered_event_weights(paths, ratios, [0.0, 0.005, 0.02, 0.1], [0.2]*5, 0.5, 5.0)
    assert current[0] / np.median(current[1:]) > 90
    assert tempered[0] / np.median(tempered[1:]) <= 5.000001
    assert np.isclose(tempered.sum(), 1.0)
