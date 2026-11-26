"""Configuration loading."""

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load the YAML configuration file into a dictionary.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}