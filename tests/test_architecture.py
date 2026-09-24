"""
Unit tests for 3D CNN model architecture (Sprint 2).
"""
import numpy as np
import pytest
import tensorflow as tf

from src.model.architecture import build_3d_cnn, conv3d_block, get_model_from_config


class TestArchitecture:
    """Test suite for 3D CNN architecture design and compilation."""

    def test_conv3d_block_with_pool(self):
        inputs = tf.keras.Input(shape=(32, 32, 32, 1))
        x = conv3d_block(inputs, filters=8, pool=True, name_prefix="test1")
        assert tuple(x.shape) == (None, 16, 16, 16, 8)

    def test_conv3d_block_without_pool(self):
        inputs = tf.keras.Input(shape=(16, 16, 16, 8))
        x = conv3d_block(inputs, filters=16, pool=False, name_prefix="test2")
        assert tuple(x.shape) == (None, 16, 16, 16, 16)


    def test_build_3d_cnn(self):
        model = build_3d_cnn(
            input_shape=(32, 32, 32, 1),
            num_classes=5,
            filters=[8, 16],
            dense_units=[16],
            dropout_rate=0.2,
        )
        assert model.input_shape == (None, 32, 32, 32, 1)
        assert model.output_shape == (None, 5)
        assert model.count_params() > 0

    def test_get_model_from_config(self, dummy_config):
        model = get_model_from_config(dummy_config)
        assert model.input_shape == (None, 32, 32, 32, 1)
        assert model.output_shape == (None, 11)

    def test_forward_pass(self, dummy_config, random_volume_batch):
        model = get_model_from_config(dummy_config)
        preds = model(random_volume_batch, training=False).numpy()
        assert preds.shape == (4, 11)
        # Softmax outputs sum to 1.0 across classes
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)
