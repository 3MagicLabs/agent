"""Structured logging to stdout (host log tabs) and a rotating file."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from agent.config import Settings, get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"

#: Libraries whose DEBUG output drowns the signal.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "openai", "e2b", "gradio", "markdown_it")

_configured = False


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure root logging once. Safe to call from any entry point."""
    global _configured
    logger = logging.getLogger("agent")
    if _configured:
        return logger

    resolved = settings or get_settings()
    root = logging.getLogger()
    root.setLevel(resolved.log_level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(stream)

    try:
        resolved.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            resolved.log_file, maxBytes=5_000_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:  # read-only filesystem: stdout logging still works
        logger.warning("File logging disabled (%s)", exc)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(f"agent.{name}")


def reset_logging() -> None:
    """Test hook: allow reconfiguration."""
    global _configured
    _configured = False
