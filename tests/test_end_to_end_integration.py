"""
Tests for Sprint 6 complete end-to-end integration workflow.
"""
import os
import numpy as np
import pytest

from src.disease_analysis.analyzer import DiseaseAnalyzer
from src.inference.engine import InferenceEngine
from src.inference.packager import ModelPackager
from src.model.architecture import build_3d_cnn
from src.preprocessing.pipeline import PreprocessingPipeline
from src.reconstruction.reconstructor import VolumeReconstructor
from src.validation.validator import ModelValidator


def test_end_to_end_integration_pipeline(dummy_config, tmp_path):
    """
    Test complete pipeline:
    Input -> Preprocessing -> 3D CNN -> Disease Analysis -> Packaging -> Standalone Inference -> Validation
    """
    cfg = dict(dummy_config)
    cfg["paths"] = dict(dummy_config["paths"])
    cfg["paths"]["export_dir"] = str(tmp_path / "export")
    cfg["paths"]["results_dir"] = str(tmp_path / "results")

    # 1. Medical Imaging Input (Synthetic Volumetric Data)
    raw_volume = np.random.uniform(0.0, 100.0, size=(24, 24, 24)).astype(np.float32)

    # 2. Preprocessing Pipeline
    pipeline = PreprocessingPipeline(cfg)
    processed_volume = pipeline.preprocess_single_volume(raw_volume, augment=False)
    assert processed_volume.shape == (32, 32, 32, 1)

    # 3. 3D CNN Model
    model = build_3d_cnn(
        input_shape=tuple(cfg["model"]["input_shape"]),
        num_classes=11,
        filters=[16, 32],
        dense_units=[64],
    )

    # 4. Disease Analysis
    analyzer = DiseaseAnalyzer(model, cfg)
    analysis = analyzer.analyze(processed_volume)
    assert 0 <= analysis["predicted_class_index"] < 11
    assert 0.0 <= analysis["confidence"] <= 1.0

    # 5. 3D Reconstruction
    recon = VolumeReconstructor(output_dir=str(tmp_path / "recon"))
    recon_outputs = recon.generate_all_visualizations(processed_volume, prefix="integration")
    assert recon_outputs["orthogonal_slices"] is not None

    # 6. Model Optimization & Packaging (Sprint 4)
    packager = ModelPackager(cfg)
    artifacts = packager.package_model(
        model,
        export_name="integration_pkg",
        export_tflite=False,
    )
    pkg_dir = os.path.dirname(artifacts["keras_model"])

    # 7. Standalone Inference Engine Execution (Sprint 4)
    engine = InferenceEngine(package_dir=pkg_dir)
    engine_res = engine.predict_volume(raw_volume, preprocess=True)
    assert engine_res["predicted_class_index"] == analysis["predicted_class_index"]

    # 8. Final Model Validation Suite (Sprint 5)
    validator = ModelValidator(model, cfg)
    val_res = validator.validate_test_cases(
        [raw_volume],
        [analysis["predicted_class_index"]],
        confidence_threshold=0.0,
    )
    assert val_res["accuracy"] == 1.0
