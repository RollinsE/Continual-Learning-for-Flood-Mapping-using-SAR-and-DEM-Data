import logging
import time

from floods.utils.common import command_logging, prepare_file_logging, prepare_logging
from floods.utils.console import ConsoleOutputFilter, LineProgress


def test_plain_progress_uses_about_ten_updates():
    assert LineProgress._choose_interval(497) == 50
    assert LineProgress._choose_interval(41) == 5
    assert LineProgress._choose_interval(3) == 1


def test_console_filter_hides_file_only_records():
    filter_ = ConsoleOutputFilter()
    visible = logging.LogRecord("test", logging.INFO, __file__, 1, "visible", (), None)
    hidden = logging.LogRecord("test", logging.INFO, __file__, 1, "hidden", (), None)
    hidden.floodmap_file_only = True
    assert filter_.filter(visible) is True
    assert filter_.filter(hidden) is False


def test_file_only_record_is_kept_in_file_but_not_console(tmp_path, capsys):
    prepare_logging("INFO")
    log_path = tmp_path / "output.log"
    handler = prepare_file_logging(tmp_path, filename=log_path.name)
    try:
        logging.getLogger("test.cleanup").info(
            "complete threshold table",
            extra={"floodmap_file_only": True},
        )
        for item in logging.getLogger().handlers:
            item.flush()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    assert "complete threshold table" not in capsys.readouterr().out
    assert "complete threshold table" in log_path.read_text(encoding="utf-8")


def test_heartbeat_is_suppressed_while_progress_is_recent(tmp_path, monkeypatch):
    prepare_logging("INFO")
    monkeypatch.setattr("floods.utils.common.seconds_since_console_activity", lambda now=None: 0.0)
    monkeypatch.setattr("floods.utils.common.seconds_since_work_activity", lambda now=None: 0.0)
    log_path = tmp_path / "output.log"
    with command_logging("train", log_file=log_path, heartbeat_seconds=0.02):
        time.sleep(0.07)
    text = log_path.read_text(encoding="utf-8")
    assert "Command running" not in text


def test_existing_log_gets_new_session_separator(tmp_path):
    prepare_logging("INFO")
    log_path = tmp_path / "output.log"
    with command_logging("train", log_file=log_path, heartbeat_seconds=0):
        pass
    with command_logging("train", log_file=log_path, heartbeat_seconds=0):
        pass
    text = log_path.read_text(encoding="utf-8")
    assert "----- New command session | command=train" in text
