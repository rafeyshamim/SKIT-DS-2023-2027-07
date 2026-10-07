"""
Shared pytest fixtures for the 3D Medical Imaging test suite.

All tests that need a dummy config, volumes, or labels should use these
fixtures rather than creating their own data to keep tests DRY.
"""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def dummy_config() -> dict:
    """
    Minimal configuration dictionary for fast unit and integration tests.

    Uses smaller volumes (32³) and a tiny network ([8, 16] filters) so
    that tests complete in seconds without requiring a GPU.
    """
    return {
        "data": {
            "dataset_name": "organmnist3d",
            "source": "medmnist",
            "nifti_dir": "data/raw/",
            "num_classes": 11,
            "class_labels": [f"Class_{i}" for i in range(11)],
        },
        "preprocessing": {
            "target_size": [32, 32, 32],
            "num_channels": 1,
            "normalize_method": "min_max",
            "clip_values": False,
            "clip_min": -1000,
            "clip_max": 1000,
            "augmentation": {
                "enabled": False,         # disabled in tests for determinism
                "random_flip_axis": [0, 1, 2],
                "rotation_range": 10,
                "zoom_range": 0.1,
            },
        },
        "model": {
            "input_shape": [32, 32, 32, 1],   # must == target_size + [num_channels]
            "filters": [8, 16],
            "kernel_size": [3, 3, 3],
            "dense_units": [32],
            "dropout_rate": 0.0,
            "l2_reg": 0.0,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
            "seed": 42,
            "early_stopping": {"enabled": False},
            "reduce_lr":      {"enabled": False},
        },
        "paths": {
            "data_dir":           "data/",
            "raw_data_dir":       "data/raw/",
            "processed_data_dir": "data/processed/",
            "model_dir":          "models/",
            "checkpoint_dir":     "models/checkpoints/test/",
            "logs_dir":           "logs/test/",
            "results_dir":        "results/test/",
            "plots_dir":          "results/test/plots/",
        },
        "inference": {
            "confidence_threshold": 0.5,
            "model_path": "models/checkpoints/best_model.keras",
        },
        "logging": {
            "level": "WARNING",
            "log_to_file": False,
        },
    }


@pytest.fixture
def random_volume_3d() -> np.ndarray:
    """A random (D, H, W) float32 volume — represents a raw scan."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64, 64)).astype(np.float32)


@pytest.fixture
def random_volume_batch() -> np.ndarray:
    """A batch of 4 already-preprocessed volumes (N, D, H, W, 1)."""
    rng = np.random.default_rng(42)
    return rng.random((4, 32, 32, 32, 1)).astype(np.float32)


@pytest.fixture
def random_labels() -> np.ndarray:
    """Random integer class labels for a batch of 4, with 11 classes."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 11, size=(4,)).astype(np.int64)
