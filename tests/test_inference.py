"""
Tests for Sprint 4 inference packaging and deployment engine.
"""
import os
import numpy as np
import pytest

from src.inference.engine import InferenceEngine
from src.inference.packager import ModelPackager
from src.model.architecture import build_3d_cnn


def test_model_packager_and_inference_engine(dummy_config, tmp_path):
    cfg = dict(dummy_config)
    cfg["paths"] = dict(dummy_config["paths"])
    cfg["paths"]["export_dir"] = str(tmp_path / "exported")

    model = build_3d_cnn(
        input_shape=tuple(cfg["model"]["input_shape"]),
        num_classes=11,
        filters=[16, 32],
        dense_units=[64],
    )

    packager = ModelPackager(cfg)
    artifacts = packager.package_model(
        model,
        export_name="test_pkg",
        export_tflite=True,
        quantize_tflite=False,
    )

    assert os.path.exists(artifacts["keras_model"])
    assert os.path.exists(artifacts["metadata"])

    # Load via InferenceEngine
    pkg_dir = os.path.dirname(artifacts["keras_model"])
    engine = InferenceEngine(package_dir=pkg_dir)

    raw_vol = np.random.uniform(0.0, 1.0, size=(32, 32, 32)).astype(np.float32)
    res = engine.predict_volume(raw_vol, preprocess=True)

    assert "predicted_class_index" in res
    assert "predicted_class_label" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 11
