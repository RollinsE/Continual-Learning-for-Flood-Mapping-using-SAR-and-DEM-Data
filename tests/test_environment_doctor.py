from floods.environment import format_environment_report, run_environment_checks


def test_environment_doctor_returns_structured_report():
    payload = run_environment_checks()
    assert payload["status"] in {"ok", "failed"}
    assert {row["name"] for row in payload["checks"]} == {
        "numpy",
        "scipy",
        "rasterio",
        "torch",
    }
    text = format_environment_report(payload)
    assert "Flood Extent Mapping environment:" in text
    assert "Flood Extent Mapping:" in text
    assert "component | version | status | detail" in text
