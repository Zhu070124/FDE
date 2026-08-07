"""
Unified logging configuration for CQUPT AI Assistant.
Replaces print() calls with file + console logging.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(
    name: str = "cqupt",
    log_dir: Path = None,
    level: str = None,
) -> logging.Logger:
    """Configure and return a logger with file + console handlers.

    Args:
        name: Logger name prefix (child loggers: name.module)
        log_dir: Directory for log files. Defaults to data/logs/
        level: Log level string. Reads LOG_LEVEL env var, defaults to INFO.

    Returns:
        Root logger for the given name.
    """
    import os

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    log_level = getattr(logging, level.upper(), logging.INFO)

    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "data" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure ROOT logger so ALL module loggers (logging.getLogger(__name__)
    # e.g. "guardrails", "main") propagate here and reach the file handlers.
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Prevent duplicate handlers on re-import
    if root_logger.handlers:
        return root_logger

    # Formatter
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # Rotating file handler (5 MB max, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # Also log warnings+ to a separate error log
    error_handler = RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(fmt)
    root_logger.addHandler(error_handler)

    return root_logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Usage:
        from logger_config import get_logger
        logger = get_logger(__name__)
        logger.info("something happened")
    """
    # Ensure root logger is set up
    root = setup_logging()
    return root.getChild(module_name.removeprefix("cqupt."))
