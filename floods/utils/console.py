"""Reliable console output for terminals, notebooks, pipes, and Colab.

The CLI is often launched from notebook shell cells or through ``tee``. In
those environments stdout is not a TTY, so carriage-return progress bars may
be buffered or hidden. This module keeps ordinary log records line-buffered
and automatically replaces dynamic tqdm bars with newline-based progress
messages when output is captured.
"""

from __future__ import annotations

from contextlib import nullcontext
import logging
import math
import os
import sys
import threading
import time
from typing import Any, Iterable, Iterator, Mapping, Optional, TextIO

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


_ACTIVITY_LOCK = threading.Lock()
_LAST_CONSOLE_ACTIVITY = time.monotonic()
_LAST_WORK_ACTIVITY = _LAST_CONSOLE_ACTIVITY


def note_console_activity(at: Optional[float] = None) -> None:
    """Record that meaningful console output was emitted.

    The command heartbeat uses this timestamp to avoid printing redundant
    ``Command running`` messages while batch or validation progress is already
    visible.
    """
    global _LAST_CONSOLE_ACTIVITY
    stamp = time.monotonic() if at is None else float(at)
    with _ACTIVITY_LOCK:
        _LAST_CONSOLE_ACTIVITY = stamp


def note_work_activity(at: Optional[float] = None) -> None:
    """Record forward progress even when no console line is due yet."""
    global _LAST_WORK_ACTIVITY
    stamp = time.monotonic() if at is None else float(at)
    with _ACTIVITY_LOCK:
        _LAST_WORK_ACTIVITY = stamp


def reset_console_activity(at: Optional[float] = None) -> None:
    """Reset console and work activity clocks for a command session."""
    stamp = time.monotonic() if at is None else float(at)
    note_console_activity(at=stamp)
    note_work_activity(at=stamp)


def seconds_since_console_activity(now: Optional[float] = None) -> float:
    """Return seconds since the most recent meaningful console record."""
    stamp = time.monotonic() if now is None else float(now)
    with _ACTIVITY_LOCK:
        last = _LAST_CONSOLE_ACTIVITY
    return max(0.0, stamp - last)


def seconds_since_work_activity(now: Optional[float] = None) -> float:
    """Return seconds since an active progress iterator completed an item."""
    stamp = time.monotonic() if now is None else float(now)
    with _ACTIVITY_LOCK:
        last = _LAST_WORK_ACTIVITY
    return max(0.0, stamp - last)


class ConsoleOutputFilter(logging.Filter):
    """Keep the console concise while preserving complete file logs.

    Records carrying ``floodmap_file_only=True`` are omitted from the stream
    handler but remain available to file handlers. Heartbeat records do not
    reset the activity clock; every other visible record does.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        file_only = bool(getattr(record, "floodmap_file_only", False))
        if not file_only and not bool(getattr(record, "floodmap_heartbeat", False)):
            note_console_activity()
        return not file_only


def configure_console_io() -> None:
    """Make Python console streams write through immediately when possible."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(line_buffering=True, write_through=True)
        except (AttributeError, OSError, TypeError, ValueError):
            # Some notebook stream wrappers expose ``reconfigure`` but do not
            # support every TextIO option. Logging still works without it.
            pass


def is_colab_runtime() -> bool:
    """Return True when common Google Colab runtime markers are present."""
    markers = (
        "COLAB_RELEASE_TAG",
        "COLAB_BACKEND_VERSION",
        "COLAB_GPU",
        "COLAB_TPU_ADDR",
    )
    return any(bool(os.environ.get(name)) for name in markers) or "google.colab" in sys.modules


def dynamic_progress_supported(stream: Optional[TextIO] = None) -> bool:
    """Whether carriage-return progress bars are suitable for this console."""
    if os.environ.get("FLOODMAP_PLAIN_PROGRESS", os.environ.get("MMFLOOD_PLAIN_PROGRESS", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return False
    stream = stream or sys.stdout
    try:
        is_tty = bool(stream.isatty())
    except Exception:
        is_tty = False
    return is_tty and not is_colab_runtime()


def progress_logging_context():
    """Redirect logging through tqdm only when dynamic bars are actually used."""
    return logging_redirect_tqdm() if dynamic_progress_supported() else nullcontext()


class LineProgress(Iterable[Any]):
    """A tqdm-compatible iterable that emits newline progress heartbeats.

    Only the small interface used by this project is implemented:
    ``set_description``, ``set_postfix``, iteration, and ``close``.
    """

    def __init__(
        self,
        iterable: Iterable[Any],
        *,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = "item",
        disable: bool = False,
        logger: Optional[logging.Logger] = None,
        min_seconds: float = 5.0,
    ) -> None:
        self.iterable = iterable
        self.total = total if total is not None else self._safe_len(iterable)
        self.desc = (desc or "Progress").strip()
        self.unit = unit
        self.disable = bool(disable)
        self.logger = logger or logging.getLogger("floodmap.progress")
        self.min_seconds = max(float(min_seconds), 0.0)
        self.count = 0
        self.started_at: Optional[float] = None
        self.last_emitted_at: Optional[float] = None
        self.postfix: dict[str, Any] = {}
        self._closed = False
        self._interval = self._choose_interval(self.total)

    @staticmethod
    def _safe_len(iterable: Iterable[Any]) -> Optional[int]:
        try:
            return int(len(iterable))  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return None

    @staticmethod
    def _choose_interval(total: Optional[int]) -> int:
        """Choose roughly ten visible updates for a finite operation."""
        if not total or total <= 0:
            return 25
        return max(1, int(math.ceil(total / 10.0)))

    def set_description(self, desc: Optional[str] = None, refresh: bool = True) -> None:
        del refresh
        if desc is not None:
            self.desc = str(desc).strip()

    def set_postfix(
        self,
        ordered_dict: Optional[Mapping[str, Any]] = None,
        refresh: bool = True,
        **kwargs: Any,
    ) -> None:
        del refresh
        if ordered_dict:
            self.postfix.update(dict(ordered_dict))
        if kwargs:
            self.postfix.update(kwargs)

    def _postfix_text(self) -> str:
        if not self.postfix:
            return ""
        return " | " + ", ".join(f"{key}={value}" for key, value in self.postfix.items())

    def _unit_text(self, count: int) -> str:
        if count == 1:
            return self.unit
        if self.unit == "batch":
            return "batches"
        return self.unit if self.unit.endswith("s") else f"{self.unit}s"

    def _emit(self, *, final: bool = False) -> None:
        if self.disable:
            return
        now = time.monotonic()
        elapsed = 0.0 if self.started_at is None else now - self.started_at
        if self.total:
            pct = min(100.0, 100.0 * self.count / self.total)
            status = f"{self.count}/{self.total} {self._unit_text(self.total)} ({pct:.1f}%)"
        else:
            status = f"{self.count} {self._unit_text(self.count)}"
        suffix = " | complete" if final else ""
        self.logger.info("%s: %s | elapsed %.1fs%s%s", self.desc, status, elapsed, self._postfix_text(), suffix)
        self.last_emitted_at = now

    def __iter__(self) -> Iterator[Any]:
        self.started_at = time.monotonic()
        self.last_emitted_at = self.started_at
        if not self.disable:
            total_text = f"{self.total} {self._unit_text(self.total)}" if self.total is not None else f"unknown {self.unit} count"
            self.logger.info("%s: starting | %s", self.desc, total_text)
        try:
            for item in self.iterable:
                yield item
                self.count += 1
                now = time.monotonic()
                note_work_activity(at=now)
                enough_items = self.count % self._interval == 0
                enough_time = self.last_emitted_at is None or (now - self.last_emitted_at) >= self.min_seconds
                if enough_items and enough_time and (self.total is None or self.count < self.total):
                    self._emit()
        finally:
            if self.total is not None and self.count >= self.total:
                self.close(completed=True)
            elif not self._closed:
                self.close(completed=False)

    def close(self, *, completed: Optional[bool] = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self.disable:
            return
        if completed is None:
            completed = self.total is not None and self.count >= self.total
        if completed:
            self._emit(final=True)
        else:
            self._emit(final=False)


def progress_iter(
    iterable: Iterable[Any],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: str = "item",
    disable: bool = False,
    colour: Optional[str] = "green",
    file: Optional[TextIO] = None,
    postfix: Optional[Mapping[str, Any]] = None,
):
    """Return a dynamic tqdm bar or a newline-based progress iterable."""
    stream = file or sys.stdout
    if dynamic_progress_supported(stream):
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            disable=disable,
            colour=colour,
            file=stream,
            postfix=dict(postfix or {}),
        )
    line_progress = LineProgress(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
    )
    if postfix:
        line_progress.set_postfix(postfix)
    return line_progress
