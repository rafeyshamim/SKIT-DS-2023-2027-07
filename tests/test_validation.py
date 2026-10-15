"""
Tests for Sprint 5 final model validation.
"""
import os
import numpy as np
import pytest

from src.model.architecture import build_3d_cnn
from src.validation.validator import ModelValidator


def test_model_validator(dummy_config, tmp_path):
    cfg = dict(dummy_config)
    cfg["paths"] = dict(dummy_config["paths"])
    cfg["paths"]["results_dir"] = str(tmp_path / "results")

    model = build_3d_cnn(
        input_shape=tuple(cfg["model"]["input_shape"]),
        num_classes=11,
        filters=[16, 32],
        dense_units=[64],
    )

    validator = ModelValidator(model, cfg)

    # 3 test cases
    vol1 = np.random.uniform(0, 1, size=(32, 32, 32)).astype(np.float32)
    vol2 = np.random.uniform(0, 1, size=(32, 32, 32)).astype(np.float32)
    vol3 = np.random.uniform(0, 1, size=(32, 32, 32)).astype(np.float32)

    test_vols = [vol1, vol2, vol3]
    labels = [0, 1, 2]

    summary = validator.validate_test_cases(
        test_vols,
        labels,
        confidence_threshold=0.05,
        save_report=True,
    )

    assert summary["total_test_cases"] == 3
    assert "accuracy" in summary
    assert "average_confidence" in summary
    assert len(summary["case_details"]) == 3
    assert os.path.exists(tmp_path / "results" / "final_model_validation_report.json")
