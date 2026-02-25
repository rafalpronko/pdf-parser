"""Logging configuration for the application."""

import logging
import sys
from contextvars import ContextVar
from typing import Any

from app.config import get_settings

# Context variable for request ID tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """Add request ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to the log record.

        Args:
            record: Log record to filter

        Returns:
            True to allow the record to be logged
        """
        record.request_id = request_id_var.get() or "N/A"
        return True


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted log string with colors
        """
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format the message
        formatted = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname

        return formatted


def setup_logging() -> None:
    """Configure logging for the application.

    Sets up:
    - Console handler with colored output
    - Request ID tracking
    - Structured log format with timestamps
    - Appropriate log levels
    """
    settings = get_settings()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.log_level)

    # Create formatter
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(request_id)s | "
        "%(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Use colored formatter for console
    if sys.stdout.isatty():
        formatter = ColoredFormatter(log_format, datefmt=date_format)
    else:
        formatter = logging.Formatter(log_format, datefmt=date_format)

    console_handler.setFormatter(formatter)

    # Add request ID filter
    request_id_filter = RequestIdFilter()
    console_handler.addFilter(request_id_filter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    # Set levels for third-party loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {settings.log_level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_request_id(request_id: str) -> None:
    """Set the request ID for the current context.

    Args:
        request_id: Request ID to set
    """
    request_id_var.set(request_id)


def clear_request_id() -> None:
    """Clear the request ID from the current context."""
    request_id_var.set(None)


def log_progress(
    logger: logging.Logger,
    operation: str,
    current: int,
    total: int,
    **kwargs: Any,
) -> None:
    """Log progress for long-running operations.

    Args:
        logger: Logger instance
        operation: Operation name
        current: Current progress count
        total: Total count
        **kwargs: Additional context to log
    """
    percentage = (current / total * 100) if total > 0 else 0
    context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    message = f"{operation}: {current}/{total} ({percentage:.1f}%)"
    if context:
        message += f" | {context}"
    logger.info(message)
