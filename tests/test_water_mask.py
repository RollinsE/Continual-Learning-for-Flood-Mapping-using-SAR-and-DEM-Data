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


def test_tokens_are_requested_per_storage_container(monkeypatch):
    requested = []

    def fake_request(url, payload=None, timeout=60.0):
        requested.append(url)
        return {"token": "sig=" + url.rsplit("/", 1)[-1]}

    monkeypatch.setattr(water_mask, "_request_json", fake_request)
    hrefs = ["https://acct.blob.core.windows.net/jrc/occurrence/a.tif",
             "https://acct.blob.core.windows.net/jrc/occurrence/b.tif",
             "https://other.blob.core.windows.net/data/c.tif"]
    signed = water_mask._sign_all(hrefs)
    assert requested == [water_mask.SAS_TOKEN_URL + "/acct/jrc", water_mask.SAS_TOKEN_URL + "/other/data"]
    assert signed[0] == hrefs[0] + "?sig=jrc" and signed[2] == hrefs[2] + "?sig=data"
