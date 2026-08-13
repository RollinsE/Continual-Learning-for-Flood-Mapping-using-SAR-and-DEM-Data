from collections.abc import MutableMapping
from contextlib import contextmanager
import logging
import subprocess
import sys
import time
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union
from uuid import uuid4

import yaml
try:
    from pydantic.v1 import BaseModel as BaseSettings
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings

from floods.logging.console import DistributedLogger
from floods.utils.console import (
    ConsoleOutputFilter,
    configure_console_io,
    reset_console_activity,
    seconds_since_console_activity,
    seconds_since_work_activity,
)


def current_timestamp() -> str:
    return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")


def code_revision() -> str:
    """Return the Git commit when available, otherwise the installed release version."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        if revision:
            return revision
    except Exception:
        pass

    from floods.version import __version__
    return f"release-{__version__}"


# Backwards-compatible import for external callers of earlier releases.
git_revision_hash = code_revision


def generate_id() -> str:
    return str(uuid4())


def _logging_level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    name = str(value or "INFO").strip().upper()
    level = getattr(logging, name, None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown logging level: {value}")
    return level


def prepare_logging(level: str | int = "INFO") -> None:
    """Initialise immediate, timestamped console logging.

    The same formatter is used by every CLI command and every file handler.
    ``force=True`` removes notebook/library handlers that may buffer records,
    while line-buffered stdout keeps logs visible in terminals, notebooks, and captured sessions as work happens.
    """
    configure_console_io()
    resolved_level = _logging_level(level)
    logging.basicConfig(
        level=resolved_level,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.addFilter(ConsoleOutputFilter())
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def prepare_file_logging(experiment_folder: Path, filename: str = "output.log") -> logging.Handler:
    """Attach an immediate UTF-8 file log using the console formatter.

    Repeated calls for the same path are idempotent, which prevents duplicate
    log lines when a CLI lifecycle log and a training run log share a file.
    """
    path = Path(experiment_folder) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    resolved = path.resolve()
    for existing in logger.handlers:
        base = getattr(existing, "baseFilename", None)
        if base and Path(base).resolve() == resolved:
            return existing
    handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=False)
    handler.setLevel(logger.level or logging.INFO)
    if logger.handlers:
        handler.setFormatter(logger.handlers[0].formatter)
    else:
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    logger.addHandler(handler)
    return handler


def remove_logging_handler(handler: Optional[logging.Handler]) -> None:
    if handler is None:
        return
    logger = logging.getLogger()
    try:
        logger.removeHandler(handler)
    finally:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass


@contextmanager
def command_logging(
    command: str,
    *,
    log_file: Optional[Path] = None,
    argv_text: Optional[str] = None,
    heartbeat_seconds: float = 30.0,
) -> Iterator[None]:
    """Log a consistent start/progress boundary/completion record for a CLI command."""
    logger = logging.getLogger("floodmap.cli")
    handler: Optional[logging.Handler] = None
    existing_log = False
    if log_file is not None:
        log_path = Path(log_file)
        existing_log = log_path.exists() and log_path.stat().st_size > 0
        handler = prepare_file_logging(log_path.parent, filename=log_path.name)
    started = time.monotonic()
    reset_console_activity(at=started)
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        interval = max(float(heartbeat_seconds), 0.0)
        if interval <= 0.0:
            return
        while not stop_heartbeat.wait(interval):
            now = time.monotonic()
            if min(
                seconds_since_console_activity(now=now),
                seconds_since_work_activity(now=now),
            ) < interval:
                continue
            logger.info(
                "Command running | command=%s | elapsed=%.1fs | no recent progress output",
                command,
                now - started,
                extra={"floodmap_heartbeat": True},
            )

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"floodmap-{command}-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    if existing_log:
        logger.info("----- New command session | command=%s | %s -----", command, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Command started | command=%s", command)
    if argv_text:
        logger.info("Command line | %s", argv_text)
    if log_file is not None:
        logger.info("Command log | %s", Path(log_file))
    try:
        yield
    except KeyboardInterrupt:
        elapsed = time.monotonic() - started
        logger.warning("Command interrupted | command=%s | elapsed=%.1fs", command, elapsed)
        raise
    except Exception:
        elapsed = time.monotonic() - started
        logger.exception("Command failed | command=%s | elapsed=%.1fs", command, elapsed)
        raise
    else:
        elapsed = time.monotonic() - started
        logger.info("Command completed | command=%s | elapsed=%.1fs", command, elapsed)
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)
        remove_logging_handler(handler)


def get_logger(name: str) -> logging.Logger:
    return DistributedLogger(logging.getLogger(name))


def check_or_make_dir(path: Union[str, Path]) -> Path:
    """Create a directory when needed and return it as a Path."""
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_plain_config(value: Any) -> Any:
    """Convert configuration objects to YAML-safe Python primitives."""
    if isinstance(value, BaseSettings):
        return {key: _to_plain_config(item) for key, item in value.dict().items()}
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MutableMapping):
        return {str(key): _to_plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_config(item) for item in value]
    return value


def config_to_plain_dict(config: BaseSettings | dict) -> dict:
    """Return a plain dict that can be written and read with yaml.safe_load."""
    plain = _to_plain_config(config)
    if not isinstance(plain, dict):
        raise TypeError(f"Expected configuration to serialise to dict, got {type(plain).__name__}")
    return plain


def print_config(logger: logging.Logger, config: BaseSettings):
    """Log a configuration object one field per line."""
    for k, v in config_to_plain_dict(config).items():
        logger.info(f"{k:<20s}: {v}")


def prepare_folder(root_folder: Path, experiment_id: str = ""):
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)
    full_path = root_folder / experiment_id
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


def store_config(config: BaseSettings, path: Path) -> None:
    """Write a run configuration as plain YAML without Python object tags."""
    with open(str(path), "w", encoding="utf-8") as file:
        yaml.safe_dump(config_to_plain_dict(config), file, sort_keys=False)


def load_config(path: Path, config_class: Callable) -> BaseSettings:
    if not path.exists():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    with open(str(path), "r", encoding="utf-8") as file:
        try:
            train_params = yaml.safe_load(file)
        except yaml.YAMLError:
            file.seek(0)
            train_params = yaml.load(file, Loader=yaml.FullLoader)
    return config_class(**(train_params or {}))


def flatten_config(config: dict, parent_key: str = "", separator: str = "/") -> Dict[str, Any]:
    items = []
    for k, v in config.items():
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def init_experiment(config: BaseSettings, log_name: str = "output.log"):
    experiment_id = config.name or current_timestamp()
    out_folder = Path(config.output_folder)
    output_folder = prepare_folder(out_folder, experiment_id=experiment_id)
    prepare_file_logging(output_folder, filename=log_name)
    model_folder = prepare_folder(output_folder / "models")
    logs_folder = prepare_folder(output_folder / "logs")
    if config.name is not None:
        if not model_folder.exists() or not logs_folder.exists():
            raise RuntimeError(f"Experiment directories could not be initialised under: {output_folder}")
    return experiment_id, output_folder, model_folder, logs_folder
