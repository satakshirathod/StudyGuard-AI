"""Structured logging for StudyGuard AI.

Provides a single, consistent logging format across the backend so every
event (session lifecycle, model detections, warnings, errors) is easy to
read and to search.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "studygard", level: int = logging.INFO) -> logging.Logger:
    """Create and return a structured logger.

    Args:
        name: Logger name (module path is fine).
        level: Minimum logging level.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured (e.g. duplicate call)
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger; top-level setup is done in ``main``."""
    return logging.getLogger(name)