"""
Centralized logging configuration module for the emergents package.

This module provides standardized logging setup following best practices:
- Consistent formatting across all modules
- Proper handler management
- Environment-based configuration
- Rich console output for development
- File logging for production
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text


class CustomRichHandler(RichHandler):
    """Enhanced RichHandler with custom styling for different log types."""

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        """Render log messages with custom styling based on content."""
        message_text = Text(message)
        lower_message = message.lower()

        # Style based on content patterns
        if any(
            keyword in lower_message
            for keyword in ["save", "saved", "saving", "export", "exported"]
        ):
            message_text.stylize("cyan")
        elif any(
            keyword in lower_message
            for keyword in ["evolution", "simulation", "generation"]
        ):
            message_text.stylize("bold blue")
        elif any(
            keyword in lower_message
            for keyword in ["complete", "completed", "finished", "done"]
        ):
            message_text.stylize("bold green")
        elif any(
            keyword in lower_message
            for keyword in ["starting", "started", "initializing", "beginning"]
        ):
            message_text.stylize("bold cyan")
        elif any(
            keyword in lower_message
            for keyword in ["interrupted", "interrupt", "stopped"]
        ):
            message_text.stylize("bold yellow")

        return message_text


def get_log_level_from_env() -> int:
    """Get log level from environment variable with fallback to INFO."""
    log_level_str = os.getenv("EMERGENTS_LOG_LEVEL", "INFO").upper()

    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_mapping.get(log_level_str, logging.INFO)


def create_file_handler(
    log_file: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.handlers.RotatingFileHandler:
    """
    Create a rotating file handler for logging.

    Args:
        log_file: Path to log file. If None, uses default location
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured rotating file handler
    """
    if log_file is None:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        log_file = str(logs_dir / "emergents.log")

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    # File format includes more detail than console
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    return file_handler


def create_console_handler(use_rich: bool = True) -> logging.Handler:
    """
    Create a console handler for logging.

    Args:
        use_rich: Whether to use Rich formatting for console output

    Returns:
        Configured console handler
    """
    if use_rich:
        console = Console(stderr=True)  # Use stderr for logging
        handler = CustomRichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,  # Don't show file paths in console
            show_time=True,
        )
    else:
        handler = logging.StreamHandler(sys.stderr)  # type: ignore
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)

    return handler


def setup_logging(
    level: int | None = None,
    log_file: str | None = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    use_rich: bool = True,
    force_reset: bool = False,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Logging level. If None, reads from environment
        log_file: Path to log file. If None, uses default
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        use_rich: Whether to use Rich formatting for console
        force_reset: Whether to remove existing handlers first
    """
    # Get the root logger for the emergents package
    root_logger = logging.getLogger("emergents")

    if force_reset:
        # Remove all existing handlers
        root_logger.handlers.clear()
        # Also clear handlers from the root logger if they exist
        logging.getLogger().handlers.clear()

    # Set log level
    if level is None:
        level = get_log_level_from_env()

    root_logger.setLevel(level)

    # Prevent propagation to root logger to avoid duplicate messages
    root_logger.propagate = False

    # Add console handler
    if enable_console_logging:
        console_handler = create_console_handler(use_rich=use_rich)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Add file handler
    if enable_file_logging:
        file_handler = create_file_handler(log_file)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Configure specific loggers that might be noisy
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    This is a convenience function that ensures consistent logger naming
    within the emergents package.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        Logger instance
    """
    # Ensure the name is within the emergents namespace
    if not name.startswith("emergents"):
        name = "emergents.main" if name == "__main__" else f"emergents.{name}"

    return logging.getLogger(name)


def configure_for_testing() -> None:
    """Configure minimal logging for testing environments."""
    setup_logging(
        level=logging.WARNING,  # Reduce verbosity during tests
        enable_file_logging=False,  # No file logging during tests
        enable_console_logging=True,
        use_rich=False,  # Simpler output for tests
        force_reset=True,
    )


def configure_for_development() -> None:
    """Configure logging for development environments."""
    setup_logging(
        level=logging.INFO,  # Changed from DEBUG to reduce verbosity
        enable_file_logging=True,
        enable_console_logging=True,
        use_rich=True,
        force_reset=True,
    )


def configure_for_production() -> None:
    """Configure logging for production environments."""
    setup_logging(
        level=logging.INFO,
        enable_file_logging=True,
        enable_console_logging=True,
        use_rich=True,
        force_reset=True,
    )


# Auto-configure based on environment if not already configured
if not logging.getLogger("emergents").handlers:
    env = os.getenv("EMERGENTS_ENV", "development").lower()
    if env == "testing":
        configure_for_testing()
    elif env == "production":
        configure_for_production()
    else:
        configure_for_development()
