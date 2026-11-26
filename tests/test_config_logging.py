"""Unit tests for config loading and logging setup."""

import logging
import pytest
from pathlib import Path


class TestConfig:

    def test_load_valid_config(self, tmp_path):
        """Valid YAML config loads into a dict."""
        from pipeline.config import load_config
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg["logging"]["level"] == "DEBUG"

    def test_missing_config_raises(self, tmp_path):
        """Missing config file raises FileNotFoundError."""
        from pipeline.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_empty_config_returns_dict(self, tmp_path):
        """Empty YAML file returns an empty dict, not None."""
        from pipeline.config import load_config
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg == {}


class TestLogging:

    def test_configure_returns_logger(self):
        """configure_logging returns a Logger instance."""
        from pipeline.logging_setup import configure_logging
        logger = configure_logging("INFO")
        assert isinstance(logger, logging.Logger)

    def test_debug_level_set(self):
        """DEBUG level is applied to the root logger."""
        from pipeline.logging_setup import configure_logging
        configure_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_invalid_level_defaults_to_info(self):
        """Unknown level string falls back to INFO."""
        from pipeline.logging_setup import configure_logging
        configure_logging("NOTAREALLEVEL")
        assert logging.getLogger().level == logging.INFO

    def test_no_duplicate_handlers(self):
        """Calling configure_logging twice does not add duplicate handlers."""
        from pipeline.logging_setup import configure_logging
        configure_logging("INFO")
        configure_logging("INFO")
        assert len(logging.getLogger().handlers) == 1