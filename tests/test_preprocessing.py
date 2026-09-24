"""
Unit tests for the preprocessing module (Sprint 1).
"""
import numpy as np
import pytest

from src.preprocessing.preprocessor import (
    add_channel_dim,
    augment_volume,
    normalize_min_max,
    normalize_volume,
    normalize_z_score,
    resize_volume,
)
from src.preprocessing.pipeline import PreprocessingPipeline


class TestPreprocessor:
    """Test suite for core preprocessor mathematical functions."""

    def test_resize_volume_shape(self, random_volume_3d):
        target = [32, 32, 32]
        resized = resize_volume(random_volume_3d, target)
        assert resized.shape == (32, 32, 32)
        assert resized.dtype == np.float32

    def test_resize_volume_same_shape(self):
        vol = np.ones((16, 16, 16), dtype=np.float32)
        resized = resize_volume(vol, [16, 16, 16])
        assert resized.shape == (16, 16, 16)
        np.testing.assert_allclose(vol, resized)

    def test_resize_volume_invalid_inputs(self):
        with pytest.raises(ValueError, match="expects a 3-D volume"):
            resize_volume(np.ones((10, 10)), [10, 10, 10])

        with pytest.raises(ValueError, match="target_size must have exactly 3 elements"):
            resize_volume(np.ones((10, 10, 10)), [10, 10])

    def test_normalize_min_max(self, random_volume_3d):
        norm = normalize_min_max(random_volume_3d)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0
        assert norm.dtype == np.float32

    def test_normalize_min_max_clipping(self):
        vol = np.array([[-100.0, 50.0], [500.0, 1200.0]]).reshape(2, 2, 1)
        norm = normalize_min_max(vol, clip_min=0.0, clip_max=1000.0)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_normalize_z_score(self, random_volume_3d):
        norm = normalize_z_score(random_volume_3d)
        assert np.isclose(norm.mean(), 0.0, atol=1e-3)
        assert np.isclose(norm.std(), 1.0, atol=1e-2)

    def test_normalize_volume_dispatch(self, random_volume_3d):
        min_max = normalize_volume(random_volume_3d, method="min_max")
        z_score = normalize_volume(random_volume_3d, method="z_score")
        assert min_max.min() >= 0.0 and min_max.max() <= 1.0
        assert np.isclose(z_score.mean(), 0.0, atol=1e-3)

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_volume(random_volume_3d, method="invalid_method")

    def test_add_channel_dim(self, random_volume_3d):
        vol_4d = add_channel_dim(random_volume_3d)
        assert vol_4d.shape == (*random_volume_3d.shape, 1)

    def test_augment_volume_preserves_shape(self, random_volume_3d):
        aug = augment_volume(
            random_volume_3d,
            flip_axes=[0, 1, 2],
            rotation_range=15.0,
            zoom_range=0.1,
        )
        assert aug.shape == random_volume_3d.shape
        assert aug.dtype == np.float32


class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline orchestration."""

    def test_pipeline_preprocess_single_volume(self, dummy_config, random_volume_3d):
        pipeline = PreprocessingPipeline(dummy_config)
        processed = pipeline.preprocess_single_volume(random_volume_3d, augment=False)
        assert processed.shape == (32, 32, 32, 1)
        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_pipeline_tf_dataset(self, dummy_config, random_volume_batch, random_labels):
        pipeline = PreprocessingPipeline(dummy_config)
        ds = pipeline.get_tf_dataset(random_volume_batch, random_labels, shuffle=False)
        
        batch_count = 0
        for batch_x, batch_y in ds:
            batch_count += 1
            assert batch_x.shape[1:] == (32, 32, 32, 1)
            assert len(batch_y.shape) == 1
        assert batch_count == 2  # batch_size=2 for 4 samples
