"""
Configuration loader for the 3D Medical Imaging project.

Reads and validates ``config/config.yaml``.  All other modules should obtain
their parameters through this module rather than hard-coding values.
"""
import os
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load the project configuration from a YAML file.

    Args:
        config_path: Path (absolute or relative) to the YAML config file.

    Returns:
        Nested dictionary of configuration parameters.

    Raises:
        FileNotFoundError: If the file does not exist at *config_path*.
        yaml.YAMLError:    If the file is syntactically invalid.
        ValueError:        If required configuration sections are missing or
                           input_shape is inconsistent with target_size.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'.\n"
            "Please ensure config/config.yaml exists in the project root."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Assert that all required top-level sections are present and that
    spatial dimensions are internally consistent.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: On any inconsistency.
    """
    required_sections = ["data", "preprocessing", "model", "training", "paths"]
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: '{section}'.\n"
                "Please check config/config.yaml."
            )

    # Verify model.input_shape == preprocessing.target_size + [num_channels]
    target_size: list = config["preprocessing"]["target_size"]
    num_channels: int = config["preprocessing"]["num_channels"]
    input_shape: list = config["model"]["input_shape"]
    expected_shape = target_size + [num_channels]

    if input_shape != expected_shape:
        raise ValueError(
            f"Inconsistent configuration:\n"
            f"  model.input_shape      = {input_shape}\n"
            f"  preprocessing expected = {expected_shape}  "
            f"(target_size={target_size} + num_channels={num_channels})\n"
            "Update config/config.yaml so that model.input_shape == "
            "preprocessing.target_size + [preprocessing.num_channels]."
        )


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a deeply nested value from a config dictionary.

    Example::

        lr = get_nested(config, "training", "learning_rate", default=0.001)

    Args:
        config:  Configuration dictionary.
        *keys:   Sequence of keys to traverse.
        default: Value returned when the key chain is not found.

    Returns:
        The value at the specified path, or *default*.
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
