from __future__ import annotations

import re
from pathlib import Path

from floods.deployment import (
    _notebook_safe_report_fragment,
    _write_deployment_summary_html,
    _write_visual_report,
)


def _assert_no_unscoped_notebook_selectors(fragment: str) -> None:
    css_blocks = re.findall(
        r"<style(?:\s[^>]*)?>(.*?)</style>",
        fragment,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert css_blocks
    css = "\n".join(css_blocks)
    forbidden = [
        r"(?m)^\s*:root\s*\{",
        r"(?m)^\s*\*\s*\{",
        r"(?m)^\s*body\s*\{",
        r"(?m)^\s*html\s*\{",
        r"(?m)^\s*pre\s*\{",
        r"(?m)^\s*code\s*\{",
        r"(?m)^\s*table\s*\{",
        r"(?m)^\s*img\s*\{",
        r"(?m)^\s*h1\s*,",
        r"(?m)^\s*th\s*,",
    ]
    for pattern in forbidden:
        assert re.search(pattern, css) is None, pattern


def test_visual_report_inline_fragment_is_scoped_and_document_free(tmp_path: Path):
    report = _write_visual_report(
        tmp_path / "scene_report.html",
        "Scene report",
        {
            "threshold": 0.8,
            "flood_pixels": 10,
            "flood_fraction": 0.1,
            "output_mask": str(tmp_path / "mask.tif"),
            "output_probability": str(tmp_path / "probability.tif"),
        },
        [],
    )
    source = report.read_text(encoding="utf-8")
    fragment = _notebook_safe_report_fragment(source)

    assert "data-floodmap-report='deployment'" in fragment
    assert ".floodmap-notebook-report .floodmap-deployment-report pre" in fragment
    assert "<html" not in fragment.lower()
    assert "<head" not in fragment.lower()
    assert "<body" not in fragment.lower()
    assert "<!doctype" not in fragment.lower()
    _assert_no_unscoped_notebook_selectors(fragment)


def test_summary_report_inline_fragment_is_scoped_and_document_free(tmp_path: Path):
    report = _write_deployment_summary_html(
        tmp_path / "deployment_summary.html",
        {"output_dir": str(tmp_path), "results": []},
        [],
        None,
    )
    fragment = _notebook_safe_report_fragment(report.read_text(encoding="utf-8"))

    assert "data-floodmap-report='summary'" in fragment
    assert ".floodmap-notebook-report .floodmap-deployment-summary pre" in fragment
    assert "<html" not in fragment.lower()
    assert "<head" not in fragment.lower()
    assert "<body" not in fragment.lower()
    _assert_no_unscoped_notebook_selectors(fragment)



def test_legacy_report_global_css_is_scoped_before_inline_display():
    legacy = """<!doctype html>
<html><head><style>
:root { --panel: #fff; }
* { box-sizing: border-box; }
body { margin: 24px; }
h1, h2 { line-height: 1.2; }
pre { background: #f6f8fa; }
@media (max-width: 700px) {
  body { margin: 12px; }
  th, td { display: block; }
}
</style></head><body><h1>Legacy report</h1><pre>metadata</pre></body></html>"""
    fragment = _notebook_safe_report_fragment(legacy)

    assert "<div class='floodmap-notebook-report'>" in fragment
    assert ".floodmap-notebook-report pre" in fragment
    assert ".floodmap-notebook-report h1" in fragment
    assert ".floodmap-notebook-report h2" in fragment
    assert "<html" not in fragment.lower()
    assert "<body" not in fragment.lower()
    _assert_no_unscoped_notebook_selectors(fragment)
