import numpy as np

from floods import water_mask


def test_water_from_occurrence_ignores_no_data():
    occurrence = np.array([[0, 49, 50], [100, 255, 75]], dtype=np.uint8)
    expected = np.array([[False, False, True], [True, False, True]])
    np.testing.assert_array_equal(water_mask.water_from_occurrence(occurrence, 50), expected)


def test_permanent_water_mask_reports_coverage(monkeypatch):
    occurrence = np.array([[10, 90], [255, 60]], dtype=np.uint8)
    monkeypatch.setattr(water_mask, "occurrence_on_grid", lambda crs, transform, shape: occurrence)
    water, info = water_mask.permanent_water_mask({"crs": "EPSG:4326", "transform": None}, (2, 2), 50)
    np.testing.assert_array_equal(water, np.array([[False, True], [False, True]]))
    assert info["applied"] is True
    assert info["water_pixels"] == 2
    assert info["layer_coverage"] == 0.75
    assert info["occurrence_threshold_percent"] == 50


def test_signed_href_appends_token():
    assert water_mask._signed("https://x/a.tif", "st=1&sig=2") == "https://x/a.tif?st=1&sig=2"
    assert water_mask._signed("https://x/a.tif?v=1", "sig=2") == "https://x/a.tif?v=1&sig=2"
