from __future__ import annotations

import logging
import sys
from typing import Optional

from src.config import settings


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with standard formatting."""
    logger_name = name or "healthcare_document_ai"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Do not propagate project log records into root handlers that may dump
        # verbose request/model payloads into unrelated application logs.
        logger.propagate = False

        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)

    return logger
