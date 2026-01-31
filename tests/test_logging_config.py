import logging

from emergents import logging_config


def test_get_log_level_from_env(monkeypatch):
    monkeypatch.setenv("EMERGENTS_LOG_LEVEL", "DEBUG")
    assert logging_config.get_log_level_from_env() == logging.DEBUG
    monkeypatch.setenv("EMERGENTS_LOG_LEVEL", "WARNING")
    assert logging_config.get_log_level_from_env() == logging.WARNING
    monkeypatch.setenv("EMERGENTS_LOG_LEVEL", "INVALID")
    assert logging_config.get_log_level_from_env() == logging.INFO
    monkeypatch.delenv("EMERGENTS_LOG_LEVEL", raising=False)
    assert logging_config.get_log_level_from_env() == logging.INFO


def test_create_file_handler_creates_file(tmp_path):
    log_file = tmp_path / "test.log"
    handler = logging_config.create_file_handler(
        str(log_file), max_bytes=1000, backup_count=2
    )
    logger = logging.getLogger("emergents.test_file")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("test message")
    handler.flush()
    assert log_file.exists()
    with open(log_file, encoding="utf-8") as f:
        content = f.read()
    assert "test message" in content
    logger.removeHandler(handler)
    handler.close()


def test_create_console_handler_rich_and_plain():
    handler_rich = logging_config.create_console_handler(use_rich=True)
    assert hasattr(handler_rich, "emit")
    handler_plain = logging_config.create_console_handler(use_rich=False)
    assert hasattr(handler_plain, "emit")


def test_setup_logging_force_reset(tmp_path):
    def test_setup_logging_level_none(monkeypatch):
        # Ensure get_log_level_from_env is called
        monkeypatch.setenv("EMERGENTS_LOG_LEVEL", "ERROR")
        # Remove all handlers to avoid duplicate
        logger = logging.getLogger("emergents")
        logger.handlers.clear()
        logging_config.setup_logging(level=None, force_reset=True)
        # Should set the level to ERROR
        assert logger.level == logging.ERROR

    log_file = tmp_path / "reset.log"
    logging_config.setup_logging(
        level=logging.DEBUG, log_file=str(log_file), force_reset=True
    )
    logger = logging.getLogger("emergents.reset")
    logger.debug("reset test")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.exists()
    with open(log_file, encoding="utf-8") as f:
        assert "reset test" in f.read()


def test_get_logger_namespace():
    logger = logging_config.get_logger("emergents.something")
    assert logger.name == "emergents.something"
    logger2 = logging_config.get_logger("__main__")
    assert logger2.name == "emergents.main"
    logger3 = logging_config.get_logger("other")
    assert logger3.name == "emergents.other"


def test_configure_for_testing_and_development_and_production():
    # Should not raise
    logging_config.configure_for_testing()
    logging_config.configure_for_development()
    logging_config.configure_for_production()


def test_custom_rich_handler_render_message_styles():
    handler = logging_config.CustomRichHandler()
    from rich.text import Text

    # Each message triggers a different style branch
    cases = [
        ("File saved successfully", "cyan"),
        ("Evolution in progress", "bold blue"),
        ("Task completed", "bold green"),
        ("Starting up", "bold cyan"),
        ("Process interrupted", "bold yellow"),
        ("No special keyword", None),
    ]
    for msg, expected_style in cases:
        record = logging.LogRecord(
            name="emergents.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        message = handler.render_message(record, record.msg)
        assert isinstance(message, Text)
        if expected_style:
            # At least one span should match the expected style
            assert any(expected_style in span.style for span in message.spans)
        else:
            # No style applied
            assert not message.spans


def test_create_file_handler_default_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handler = logging_config.create_file_handler(None)
    handler.close()
    assert (tmp_path / "logs" / "emergents.log").exists()


def test_setup_logging_no_handlers(monkeypatch):
    # Remove all handlers and test auto-configure for all envs
    import importlib

    logger = logging.getLogger("emergents")
    logger.handlers.clear()
    monkeypatch.setenv("EMERGENTS_ENV", "testing")
    importlib.reload(logging_config)
    assert logger.handlers
    logger.handlers.clear()
    monkeypatch.setenv("EMERGENTS_ENV", "production")
    importlib.reload(logging_config)
    assert logger.handlers
    logger.handlers.clear()
    monkeypatch.setenv("EMERGENTS_ENV", "development")
    importlib.reload(logging_config)
    assert logger.handlers
