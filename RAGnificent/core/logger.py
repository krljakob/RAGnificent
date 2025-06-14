"""
Centralized logging configuration for RAGnificent.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Define custom color scheme for logs
LOG_THEME = Theme(
    {
        "logging.level.debug": "cyan",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "red",
        "logging.level.critical": "bold red",
    }
)

# Global console instance with theme
console = Console(theme=LOG_THEME)


def setup_logger(
    name: str = "ragnificent",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    enable_rich: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with rich formatting.

    Args:
        name: Logger name
        level: Logging level (e.g., logging.INFO, "DEBUG")
        log_file: Optional file path to write logs to
        enable_rich: Whether to use rich formatting for console output

    Returns:
        Configured logger instance
    """
    # Convert string level to logging level if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times in case of multiple calls
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler with rich formatting
    if enable_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name, using default configuration.

    Args:
        name: Optional logger name. If None, returns the root logger.
            Use __name__ for module-level logging.

    Returns:
        Configured logger instance
    """
    if name is None:
        name = "ragnificent"

    # If logger doesn't exist or has no handlers, set it up
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)

    return logger


# Default logger instance
logger = get_logger()


def log_execution_time(logger: logging.Logger = None):
    """
    Decorator to log the execution time of a function.

    Args:
        logger: Logger instance to use. If None, uses the default logger.
    """
    if logger is None:
        logger = globals().get("logger", get_logger())

    def decorator(func):
        import time
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                logger.debug(f"Starting {func.__qualname__}")
                return func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                logger.debug(
                    f"Completed {func.__qualname__} in "
                    f"{end_time - start_time:.4f} seconds"
                )

        return wrapper

    return decorator
