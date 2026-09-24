"""
Unit tests for DiseaseAnalyzer module (Sprint 3).
"""
import numpy as np
import pytest

from src.disease_analysis.analyzer import DiseaseAnalyzer
from src.model.architecture import get_model_from_config


class TestDiseaseAnalyzer:
    """Test suite for DiseaseAnalyzer inference and confidence scoring."""

    def test_analyzer_single_volume(self, dummy_config):
        model = get_model_from_config(dummy_config)
        analyzer = DiseaseAnalyzer(model, dummy_config)

        vol = np.random.default_rng(42).random((32, 32, 32, 1)).astype(np.float32)
        result = analyzer.analyze(vol)

        assert "predicted_class_index" in result
        assert isinstance(result["predicted_class_index"], int)
        assert 0 <= result["predicted_class_index"] < 11
        assert "predicted_class_label" in result
        assert result["predicted_class_label"] == f"Class_{result['predicted_class_index']}"
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "probabilities" in result
        assert len(result["probabilities"]) == 11
        assert "above_threshold" in result
        assert isinstance(result["above_threshold"], bool)

    def test_analyzer_single_volume_with_batch_dim(self, dummy_config):
        model = get_model_from_config(dummy_config)
        analyzer = DiseaseAnalyzer(model, dummy_config)

        vol = np.random.default_rng(42).random((1, 32, 32, 32, 1)).astype(np.float32)
        result = analyzer.analyze(vol)
        assert isinstance(result["confidence"], float)

    def test_analyzer_batch(self, dummy_config, random_volume_batch):
        model = get_model_from_config(dummy_config)
        analyzer = DiseaseAnalyzer(model, dummy_config)

        results = analyzer.analyze_batch(random_volume_batch)
        assert len(results) == 4
        for res in results:
            assert 0.0 <= res["confidence"] <= 1.0

    def test_analyzer_invalid_shape(self, dummy_config):
        model = get_model_from_config(dummy_config)
        analyzer = DiseaseAnalyzer(model, dummy_config)

        # Invalid 2D shape
        with pytest.raises(ValueError, match="Expected volume of shape"):
            analyzer.analyze(np.ones((32, 32), dtype=np.float32))
