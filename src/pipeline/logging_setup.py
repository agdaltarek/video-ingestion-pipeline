"""Centralised logging configuration for the pipeline."""

import logging
import sys


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging and return the pipeline logger.

    Args:
        level: Logging level name, e.g. "DEBUG", "INFO", "WARNING".

    Returns:
        The configured "pipeline" logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Avoid duplicate handlers if called more than once (e.g. in tests).
    root.handlers.clear()
    root.addHandler(handler)

    return logging.getLogger("pipeline")