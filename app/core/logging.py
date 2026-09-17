"""Logging configuration module for the application."""

import sys
import logging
from typing import Optional
from app.core.config import settings


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """Configure structured console logging for the application.

    Args:
        log_level: Optional logging level override (e.g. 'DEBUG', 'INFO').
                   If None, uses settings.LOG_LEVEL.

    Returns:
        logging.Logger configured logger instance.
    """
    level_str = (log_level or settings.LOG_LEVEL).upper()
    level = getattr(logging, level_str, logging.INFO)

    log_format = (
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Overwrite existing handlers
    )

    logger = logging.getLogger("breed_recognition")
    logger.setLevel(level)
    return logger


# Pre-configured default logger
logger = setup_logging()
