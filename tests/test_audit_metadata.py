from types import SimpleNamespace

from floods.error_audit import _normalization_mode


def test_normalization_mode_reads_configured_value():
    config = SimpleNamespace(data=SimpleNamespace(normalization_mode="ROBUST_PERCENTILE"))
    assert _normalization_mode(config) == "robust_percentile"


def test_normalization_mode_defaults_to_fixed():
    assert _normalization_mode(SimpleNamespace()) == "fixed"
    assert _normalization_mode(SimpleNamespace(data=SimpleNamespace(normalization_mode=None))) == "fixed"
