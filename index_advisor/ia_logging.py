import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any


logger = logging.getLogger("advisor")
logger.propagate = False
logger.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d [%(levelname)-7s] [%(name)s] <%(funcName)s> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler(stream=sys.stdout)
console.setFormatter(console_formatter)
console.setLevel(logging.INFO)
logger.addHandler(console)

_LOG_FILE_ENV = "IA_LOG_FILE"
_LOG_LEVEL_ENV = "IA_LOG_LEVEL"
_file_handler: logging.Handler | None = None


def _coerce_level(level: Any) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        attr = level.upper()
        if hasattr(logging, attr):
            return getattr(logging, attr)
    return logging.INFO


def _attach_file_handler(log_file: str, level: int) -> str:
    global _file_handler
    log_file = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if _file_handler is not None:
        same_path = getattr(_file_handler, "baseFilename", None) == log_file
        if not same_path:
            logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
        else:
            _file_handler.setLevel(level)
            return log_file

    handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
    handler.setFormatter(console_formatter)
    handler.setLevel(level)
    logger.addHandler(handler)
    _file_handler = handler
    return log_file


def _maybe_attach_from_env() -> None:
    log_file = os.environ.get(_LOG_FILE_ENV)
    if not log_file:
        return
    level = _coerce_level(os.environ.get(_LOG_LEVEL_ENV, "INFO"))
    console.setLevel(level)
    _attach_file_handler(log_file, level)


def log_to_file(
    task: str = "train",
    level: int | str = "INFO",
    desc: str = "",
    args: list[str] | None = None,
    log_dir: str | Path = "./ia-logs",
):
    level_value = _coerce_level(level)
    existing = os.environ.get(_LOG_FILE_ENV)
    if existing:
        log_file = existing
    else:
        log_dir = Path(log_dir)
        time = datetime.datetime.now()
        log_file = (
            log_dir
            / f"{task}-{time.year}-{time.month}-{time.day} {time.hour:02}-{time.minute:02}-{time.second:02}.log"
        )

    log_file = _attach_file_handler(str(log_file), level_value)
    console.setLevel(level_value)
    os.environ[_LOG_FILE_ENV] = log_file
    os.environ[_LOG_LEVEL_ENV] = logging.getLevelName(level_value)

    if args is None:
        args = sys.argv
    logger.info(f"Task: {task}, Desc: {desc}, Cmd: {args[0]} {' '.join(args[1:])}")
    return log_file


_maybe_attach_from_env()
