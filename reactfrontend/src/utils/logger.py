"""
Logging utilities for the 3D Medical Imaging project.
"""
import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and return a configured logger instance.

    Guards against adding duplicate handlers when the same logger is
    requested multiple times (e.g. in unit tests).

    Args:
        name:        Logger name — typically ``__name__`` of the caller.
        log_level:   Logging level string: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        log_to_file: Whether to write logs to a file in addition to stderr.
        log_file:    Path to the log file.  If *None*, defaults to
                     ``logs/<name>_<timestamp>.log``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid attaching duplicate handlers (happens in unit tests / reloads).
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler -------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler (optional) -----------------------------------------------
    if log_to_file:
        if log_file is None:
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/{name}_{timestamp}.log"
        else:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
