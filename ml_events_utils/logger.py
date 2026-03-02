import os
import logging
from pathlib import Path
from typing import Optional, Union


log_file = os.environ.get("ML_EVENTS_UTILS_LOG_FILE", "ml_events_utils.log")

_PACKAGE_LOGGER_NAME = "ml_events_utils"
_LOGGER_CONFIGURED = False


def _parse_log_level(level: Optional[Union[int, str]]) -> int:
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        level_str = level.strip().upper()
        if level_str.isdigit():
            return int(level_str)
        return getattr(logging, level_str, logging.INFO)

    return logging.INFO


def setup_file_logger(
    log_file: Optional[Union[str, Path]] = None,
    level: Optional[Union[int, str]] = None,
    console: bool = False,
    mode: str = "a",
    force: bool = False,
) -> logging.Logger:
    """Configure package logging once.

    Intended usage: call this once at program start.

    All loggers inside this package should use `logging.getLogger(__name__)`.
    Those loggers (e.g. `ml_events_utils.train_loop`) will propagate to the
    package logger (`ml_events_utils`), which owns the file handler.
    """
    global _LOGGER_CONFIGURED
    global _PACKAGE_LOGGER_NAME

    if log_file is None:
        log_file = os.environ.get("ML_EVENTS_UTILS_LOG_FILE", "ml_events_utils.log")

    log_level = _parse_log_level(level)
    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)

    if _LOGGER_CONFIGURED and not force:
        return package_logger

    # Ensure file directory exists.
    log_path = Path(log_file)
    # if log_path.parent and str(log_path.parent) not in (".", ""):
    #     log_path.parent.mkdir(parents=True, exist_ok=True)

    package_logger.setLevel(log_level)

    # Own handlers here; avoid double logging via root.
    package_logger.propagate = False

    # Clear existing handlers (important for repeated runs in notebooks / VS Code).
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    package_logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        package_logger.addHandler(stream_handler)

    _LOGGER_CONFIGURED = True
    # package_logger.info("Initialized %s logger, version %s (log_file=%s)", _PACKAGE_LOGGER_NAME, __version__, str(log_path))
    return package_logger