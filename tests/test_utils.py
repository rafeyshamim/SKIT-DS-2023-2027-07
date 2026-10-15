"""
Unit tests for utility modules: config_loader, metrics, logger.
"""
import os
import numpy as np
import pytest

from src.utils.config_loader import load_config, _validate_config
from src.utils.logger import setup_logger
from src.utils.metrics import (
    compute_classification_metrics,
    get_classification_report,
    get_confusion_matrix,
)


class TestConfigLoader:
    """Test suite for config loading and validation."""

    def test_load_default_config(self):
        config = load_config("config/config.yaml")
        assert "data" in config
        assert "preprocessing" in config
        assert "model" in config
        assert "training" in config
        assert "paths" in config

    def test_load_config_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_path_to_config.yaml")

    def test_validate_config_missing_section(self, dummy_config):
        cfg = dict(dummy_config)
        del cfg["data"]
        with pytest.raises(ValueError, match="Missing required configuration section"):
            _validate_config(cfg)

    def test_validate_config_shape_mismatch(self, dummy_config):
        cfg = dict(dummy_config)
        cfg["model"] = dict(dummy_config["model"])
        cfg["model"]["input_shape"] = [64, 64, 64, 1]  # target_size is 32, 32, 32
        with pytest.raises(ValueError, match="Inconsistent configuration"):
            _validate_config(cfg)


class TestLogger:
    """Test suite for logger setup."""

    def test_setup_logger(self):
        logger = setup_logger("test_logger", log_level="DEBUG", log_to_file=False)
        assert logger.name == "test_logger"


class TestMetrics:
    """Test suite for metric calculations."""

    def test_compute_classification_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        y_prob = np.eye(3)[[0, 1, 1, 0, 1, 2]]

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_get_classification_report(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        report = get_classification_report(y_true, y_pred, class_labels=["A", "B"])
        assert "precision" in report
        assert "recall" in report

    def test_get_confusion_matrix(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        cm = get_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
