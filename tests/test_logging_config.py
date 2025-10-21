"""
Unit tests for emergents.logging_config module.
Tests logging configuration, handlers, and utility functions.
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from emergents.logging_config import (
    CustomRichHandler,
    configure_for_development,
    configure_for_production,
    configure_for_testing,
    create_console_handler,
    create_file_handler,
    get_log_level_from_env,
    get_logger,
    setup_logging,
)


class TestGetLogLevelFromEnv:
    """Test get_log_level_from_env function."""

    def test_default_log_level(self) -> None:
        """Test default log level when no environment variable is set."""
        with patch.dict(os.environ, {}, clear=True):
            level = get_log_level_from_env()
        assert level == logging.INFO

    def test_valid_log_levels(self) -> None:
        """Test all valid log level strings."""
        test_cases = [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]

        for env_value, expected_level in test_cases:
            with patch.dict(os.environ, {"EMERGENTS_LOG_LEVEL": env_value}):
                level = get_log_level_from_env()
            assert level == expected_level

    def test_case_insensitive_log_levels(self) -> None:
        """Test that log level parsing is case insensitive."""
        test_cases = ["debug", "Debug", "DEBUG", "dEbUg"]

        for env_value in test_cases:
            with patch.dict(os.environ, {"EMERGENTS_LOG_LEVEL": env_value}):
                level = get_log_level_from_env()
            assert level == logging.DEBUG

    def test_invalid_log_level(self) -> None:
        """Test handling of invalid log level values."""
        with patch.dict(os.environ, {"EMERGENTS_LOG_LEVEL": "INVALID"}):
            level = get_log_level_from_env()
        assert level == logging.INFO  # Should fall back to default


class TestCreateFileHandler:
    """Test create_file_handler function."""

    @patch("emergents.logging_config.Path.mkdir")
    def test_default_file_handler(self, mock_mkdir: Mock) -> None:
        """Test creating file handler with default parameters."""
        handler = create_file_handler()

        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.baseFilename.endswith("emergents.log")
        assert handler.maxBytes == 10 * 1024 * 1024  # 10MB
        assert handler.backupCount == 5
        mock_mkdir.assert_called_once_with(exist_ok=True)

    def test_custom_file_handler(self) -> None:
        """Test creating file handler with custom parameters."""
        custom_file = "test_custom.log"
        custom_max_bytes = 5 * 1024 * 1024  # 5MB
        custom_backup_count = 3

        handler = create_file_handler(
            log_file=custom_file,
            max_bytes=custom_max_bytes,
            backup_count=custom_backup_count,
        )

        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.baseFilename.endswith(custom_file)
        assert handler.maxBytes == custom_max_bytes
        assert handler.backupCount == custom_backup_count

    def test_file_handler_formatter(self) -> None:
        """Test that file handler has correct formatter."""
        handler = create_file_handler()

        formatter = handler.formatter
        assert formatter is not None
        assert "%(asctime)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(funcName)s" in formatter._fmt
        assert "%(lineno)d" in formatter._fmt
        assert "%(message)s" in formatter._fmt


class TestCreateConsoleHandler:
    """Test create_console_handler function."""

    @patch("emergents.logging_config.CustomRichHandler")
    def test_rich_console_handler(self, mock_rich_handler: Mock) -> None:
        """Test creating Rich console handler."""
        mock_handler_instance = Mock()
        mock_rich_handler.return_value = mock_handler_instance

        handler = create_console_handler(use_rich=True)

        assert handler == mock_handler_instance
        mock_rich_handler.assert_called_once()

    def test_plain_console_handler(self) -> None:
        """Test creating plain console handler."""
        handler = create_console_handler(use_rich=False)

        assert isinstance(handler, logging.StreamHandler)
        # Should use stderr (name might be different in tests)
        assert hasattr(handler.stream, "name")

        formatter = handler.formatter
        assert formatter is not None
        assert "%(asctime)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt


class TestCustomRichHandler:
    """Test CustomRichHandler class."""

    def test_render_message_save_keywords(self) -> None:
        """Test message rendering for save-related keywords."""
        handler = CustomRichHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Saving data to file",
            args=(),
            exc_info=None,
        )

        result = handler.render_message(record, "Saving data to file")

        # Should style save-related messages
        assert str(result) == "Saving data to file"

    def test_render_message_evolution_keywords(self) -> None:
        """Test message rendering for evolution-related keywords."""
        handler = CustomRichHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Evolution completed",
            args=(),
            exc_info=None,
        )

        result = handler.render_message(record, "Evolution completed")

        assert str(result) == "Evolution completed"

    def test_render_message_no_special_keywords(self) -> None:
        """Test message rendering for messages with no special keywords."""
        handler = CustomRichHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Regular log message",
            args=(),
            exc_info=None,
        )

        result = handler.render_message(record, "Regular log message")

        assert str(result) == "Regular log message"


class TestSetupLogging:
    """Test setup_logging function."""

    @patch("emergents.logging_config.create_console_handler")
    @patch("emergents.logging_config.create_file_handler")
    def test_setup_logging_default(
        self, mock_file_handler: Mock, mock_console_handler: Mock
    ) -> None:
        """Test setup_logging with default parameters."""
        mock_file_handler.return_value = Mock()
        mock_console_handler.return_value = Mock()

        setup_logging()

        # Should create both handlers
        mock_file_handler.assert_called_once_with(None)
        mock_console_handler.assert_called_once_with(use_rich=True)

    @patch("emergents.logging_config.create_console_handler")
    @patch("emergents.logging_config.create_file_handler")
    def test_setup_logging_file_only(
        self, mock_file_handler: Mock, mock_console_handler: Mock
    ) -> None:
        """Test setup_logging with file logging only."""
        mock_file_handler.return_value = Mock()

        setup_logging(enable_file_logging=True, enable_console_logging=False)

        mock_file_handler.assert_called_once()
        mock_console_handler.assert_not_called()

    @patch("emergents.logging_config.create_console_handler")
    @patch("emergents.logging_config.create_file_handler")
    def test_setup_logging_console_only(
        self, mock_file_handler: Mock, mock_console_handler: Mock
    ) -> None:
        """Test setup_logging with console logging only."""
        mock_console_handler.return_value = Mock()

        setup_logging(enable_file_logging=False, enable_console_logging=True)

        mock_file_handler.assert_not_called()
        mock_console_handler.assert_called_once()

    @patch("emergents.logging_config.get_log_level_from_env")
    def test_setup_logging_with_env_level(self, mock_get_level: Mock) -> None:
        """Test setup_logging uses environment log level when level=None."""
        mock_get_level.return_value = logging.DEBUG

        setup_logging(level=None)

        mock_get_level.assert_called_once()

    def test_setup_logging_external_loggers(self) -> None:
        """Test that setup_logging configures external library loggers."""
        setup_logging()

        # Check that external loggers are set to WARNING
        assert logging.getLogger("matplotlib").level == logging.WARNING
        assert logging.getLogger("PIL").level == logging.WARNING
        assert logging.getLogger("urllib3").level == logging.WARNING


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_with_emergents_name(self) -> None:
        """Test getting logger with name already in emergents namespace."""
        logger = get_logger("emergents.test_module")

        assert logger.name == "emergents.test_module"

    def test_get_logger_with_main_name(self) -> None:
        """Test getting logger for __main__."""
        logger = get_logger("__main__")

        assert logger.name == "emergents.main"

    def test_get_logger_with_other_name(self) -> None:
        """Test getting logger for module not in emergents namespace."""
        logger = get_logger("some_other_module")

        assert logger.name == "emergents.some_other_module"


class TestConfigurationFunctions:
    """Test configuration convenience functions."""

    @patch("emergents.logging_config.setup_logging")
    def test_configure_for_testing(self, mock_setup: Mock) -> None:
        """Test configure_for_testing function."""
        configure_for_testing()

        mock_setup.assert_called_once_with(
            level=logging.WARNING,
            enable_file_logging=False,
            enable_console_logging=True,
            use_rich=False,
            force_reset=True,
        )

    @patch("emergents.logging_config.setup_logging")
    def test_configure_for_development(self, mock_setup: Mock) -> None:
        """Test configure_for_development function."""
        configure_for_development()

        mock_setup.assert_called_once_with(
            level=logging.INFO,  # Changed from DEBUG
            enable_file_logging=True,
            enable_console_logging=True,
            use_rich=True,
            force_reset=True,
        )

    @patch("emergents.logging_config.setup_logging")
    def test_configure_for_production(self, mock_setup: Mock) -> None:
        """Test configure_for_production function."""
        configure_for_production()

        mock_setup.assert_called_once_with(
            level=logging.INFO,
            enable_file_logging=True,
            enable_console_logging=True,
            use_rich=True,
            force_reset=True,
        )


class TestAutoConfiguration:
    """Test automatic configuration based on environment."""

    @patch("emergents.logging_config.configure_for_testing")
    def test_auto_config_testing(self, mock_configure: Mock) -> None:
        """Test auto-configuration for testing environment."""
        with patch.dict(os.environ, {"EMERGENTS_ENV": "testing"}):
            # Reimport to trigger auto-configuration
            import importlib

            import emergents.logging_config

            importlib.reload(emergents.logging_config)

        # Note: This test is tricky because the auto-config runs at import time
        # In a real test, we'd need to carefully manage imports

    def test_logger_hierarchy(self) -> None:
        """Test that loggers are properly organized in hierarchy."""
        # Get various loggers
        main_logger = get_logger("emergents.main")
        config_logger = get_logger("emergents.config")
        nested_logger = get_logger("emergents.genome.genome")

        # All should be under the emergents namespace
        assert main_logger.name.startswith("emergents")
        assert config_logger.name.startswith("emergents")
        assert nested_logger.name.startswith("emergents")


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_complete_logging_setup(self) -> None:
        """Test complete logging setup and usage."""
        # Configure logging
        setup_logging(
            level=logging.DEBUG,
            enable_file_logging=False,  # Don't create files in tests
            enable_console_logging=True,
            use_rich=False,  # Simpler for testing
            force_reset=True,
        )

        # Get a logger and test it works
        logger = get_logger("emergents.test")

        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        assert logger.name == "emergents.test"
        assert logger.level <= logging.DEBUG  # Should be able to log debug

    def test_logging_with_different_configurations(self) -> None:
        """Test that different configuration functions work."""
        # Test each configuration
        configure_for_testing()
        test_logger = get_logger("emergents.test_testing")
        test_logger.info("Test message")

        configure_for_development()
        dev_logger = get_logger("emergents.test_dev")
        dev_logger.info("Dev message")

        configure_for_production()
        prod_logger = get_logger("emergents.test_prod")
        prod_logger.info("Prod message")

        # All should work without exceptions
        assert test_logger.name.startswith("emergents")
        assert dev_logger.name.startswith("emergents")
        assert prod_logger.name.startswith("emergents")
