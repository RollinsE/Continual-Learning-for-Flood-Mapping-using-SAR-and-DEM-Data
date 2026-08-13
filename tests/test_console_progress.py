import logging

from floods.utils.console import LineProgress, configure_console_io, dynamic_progress_supported, progress_iter


def test_configure_console_io_is_safe():
    configure_console_io()


def test_line_progress_emits_start_and_completion(caplog):
    caplog.set_level(logging.INFO)
    progress = LineProgress(range(3), desc="Mask scan", unit="tile", min_seconds=0.0)
    for _ in progress:
        pass
    messages = [record.getMessage() for record in caplog.records]
    assert any("Mask scan: starting" in message for message in messages)
    assert any("3/3 tiles (100.0%)" in message and "complete" in message for message in messages)


def test_progress_iter_uses_line_mode_when_dynamic_progress_is_disabled(monkeypatch):
    monkeypatch.setenv("MMFLOOD_PLAIN_PROGRESS", "1")
    progress = progress_iter(range(2), desc="Plain")
    assert isinstance(progress, LineProgress)
    assert dynamic_progress_supported() is False
