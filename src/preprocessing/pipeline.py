"""
End-to-end volumetric preprocessing pipeline.

Orchestrates:
    loading  →  resizing  →  normalization  →  augmentation  →  channel expansion
    →  tf.data.Dataset assembly

This class is the main entry-point for data preparation and is used by
both ``train.py`` and ``inference.py``.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing.loader import load_medmnist3d, load_nifti_dataset
from src.preprocessing.preprocessor import (
    add_channel_dim,
    augment_volume,
    normalize_volume,
    resize_volume,
)
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PreprocessingPipeline:
    """
    Configurable, reproducible preprocessing pipeline for volumetric medical
    images.

    **Input format (raw)**
        - NIfTI: ``float32`` array ``(D, H, W)`` in scanner voxel units.
        - MedMNIST3D: ``float32`` array ``(D, H, W)`` in ``[0, 255]``.

    **Output format (preprocessed — fed to the 3D CNN)**
        - Shape : ``(D, H, W, 1)``
        - Dtype : ``float32``
        - Range : ``[0, 1]`` (min-max) or ≈ N(0, 1) (z-score)

    Usage::

        pipeline = PreprocessingPipeline(config)
        splits   = pipeline.load_and_prepare_data()
        train_ds = pipeline.get_tf_dataset(*splits["train"], shuffle=True)
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Args:
            config: Configuration dictionary (from :func:`~src.utils.config_loader.load_config`).
                    If *None*, loads from ``config/config.yaml``.
        """
        if config is None:
            config = load_config()
        self.config = config
        self._extract_params()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_params(self) -> None:
        """Cache frequently accessed config values as instance attributes."""
        prep = self.config["preprocessing"]
        self.target_size: List[int]      = prep["target_size"]
        self.num_channels: int           = prep["num_channels"]
        self.normalize_method: str       = prep["normalize_method"]
        self.clip_values: bool           = bool(prep.get("clip_values", False))
        self.clip_min: Optional[float]   = prep.get("clip_min")
        self.clip_max: Optional[float]   = prep.get("clip_max")

        aug = prep.get("augmentation", {})
        self.aug_enabled: bool           = bool(aug.get("enabled", False))
        self.flip_axes: Optional[List[int]] = aug.get("random_flip_axis")
        self.rotation_range: float       = float(aug.get("rotation_range", 0))
        self.zoom_range: float           = float(aug.get("zoom_range", 0))

        data = self.config["data"]
        self.source: str       = data["source"]
        self.dataset_name: str = data.get("dataset_name", "organmnist3d")

        paths = self.config["paths"]
        self.data_dir: str = paths["data_dir"]
        self.raw_dir: str  = paths["raw_data_dir"]

        train_cfg = self.config["training"]
        self.batch_size: int = int(train_cfg["batch_size"])
        self.seed: int       = int(train_cfg.get("seed", 42))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_single_volume(
        self,
        volume: np.ndarray,
        augment: bool = False,
    ) -> np.ndarray:
        """
        Apply the full preprocessing chain to a single 3-D volume.

        Steps:
            1. Strip trailing channel dim (if present).
            2. Resize to ``target_size``.
            3. Normalize intensities.
            4. Optionally augment (training only).
            5. Add channel dimension.

        Args:
            volume:  Raw volume, shape ``(D, H, W)`` or ``(D, H, W, C)``.
            augment: Pass ``True`` during training to enable augmentation.

        Returns:
            Preprocessed tensor of shape ``(D, H, W, 1)``, dtype ``float32``.
        """
        # Strip channel dim if already present
        if volume.ndim == 4:
            volume = volume[..., 0]

        # 1. Resize
        volume = resize_volume(volume, self.target_size)

        # 2. Normalize
        clip_min = self.clip_min if self.clip_values else None
        clip_max = self.clip_max if self.clip_values else None
        volume = normalize_volume(
            volume,
            method=self.normalize_method,
            clip_min=clip_min,
            clip_max=clip_max,
        )

        # 3. Augment (training only)
        if augment and self.aug_enabled:
            volume = augment_volume(
                volume,
                flip_axes=self.flip_axes,
                rotation_range=self.rotation_range,
                zoom_range=self.zoom_range,
            )

        # 4. Add channel dimension: (D, H, W) -> (D, H, W, 1)
        volume = add_channel_dim(volume)
        return volume

    def preprocess_batch(
        self,
        volumes: List[np.ndarray],
        labels: Optional[np.ndarray] = None,
        augment: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess a list of raw volumes into a model-ready batch.

        Args:
            volumes: List of raw 3-D arrays.
            labels:  Optional integer labels of shape ``(N,)`` or ``(N, 1)``.
            augment: Enable augmentation for training batches.

        Returns:
            Tuple ``(batch_X, batch_y)`` where

            - *batch_X*: ``float32`` array of shape ``(N, D, H, W, 1)``.
            - *batch_y*: ``int64`` array of shape ``(N,)``, or *None*.
        """
        processed = [
            self.preprocess_single_volume(vol, augment=augment)
            for vol in volumes
        ]
        batch_X = np.stack(processed, axis=0)

        batch_y = None
        if labels is not None:
            batch_y = np.asarray(labels, dtype=np.int64).reshape(-1)

        return batch_X, batch_y

    def load_and_prepare_data(
        self,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load the full dataset and preprocess all splits.

        Returns:
            Dictionary with keys ``'train'``, ``'val'``, ``'test'`` (where
            available).  Each value is a tuple ``(X, y)`` where

            - *X*: ``float32`` array of shape ``(N, D, H, W, 1)``.
            - *y*: ``int64`` array of shape ``(N,)``.

        Raises:
            ValueError: If ``data.source`` is not ``'medmnist'`` or
                        ``'nifti'``.
        """
        if self.source == "medmnist":
            return self._load_medmnist()
        if self.source == "nifti":
            return self._load_nifti()
        raise ValueError(
            f"Unknown data source: '{self.source}'. "
            "Set data.source to 'medmnist' or 'nifti' in config.yaml."
        )

    def get_tf_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = False,
    ):
        """
        Wrap numpy arrays in a ``tf.data.Dataset`` for efficient batched
        training.

        Args:
            X:       ``float32`` array of shape ``(N, D, H, W, 1)``.
            y:       ``int64`` array of shape ``(N,)``.
            shuffle: If *True*, shuffle with the configured random seed.

        Returns:
            Batched + prefetched ``tf.data.Dataset``.
        """
        import tensorflow as tf  # deferred to avoid mandatory TF import at module level

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(X),
                seed=self.seed,
                reshuffle_each_iteration=True,
            )
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_medmnist(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load MedMNIST3D and preprocess all splits."""
        raw_splits = load_medmnist3d(self.dataset_name, data_dir=self.data_dir)
        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for split, (volumes_raw, labels_raw) in raw_splits.items():
            augment = split == "train"
            logger.info(
                "Preprocessing '%s' split (%d samples) ...",
                split, len(volumes_raw),
            )

            processed = []
            for vol in volumes_raw:
                # MedMNIST volumes are uint8 [0, 255] — pass as float32;
                # normalize_volume handles the scaling to [0, 1].
                processed.append(
                    self.preprocess_single_volume(
                        vol.astype(np.float32), augment=augment
                    )
                )

            X = np.stack(processed, axis=0)
            y = labels_raw.reshape(-1).astype(np.int64)
            result[split] = (X, y)
            logger.info("  '%s': X=%s  y=%s", split, X.shape, y.shape)

        return result

    def _load_nifti(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load NIfTI volumes from the raw data directory.

        Expects the layout::

            data/raw/
                train/<class_name>/*.nii.gz
                val/<class_name>/*.nii.gz
                test/<class_name>/*.nii.gz
        """
        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for split in ("train", "val", "test"):
            split_dir = os.path.join(self.raw_dir, split)
            if not os.path.isdir(split_dir):
                logger.warning(
                    "NIfTI split directory not found — skipping: '%s'", split_dir
                )
                continue

            augment = split == "train"
            volumes_raw, labels_raw = load_nifti_dataset(split_dir)
            batch_X, batch_y = self.preprocess_batch(
                volumes_raw, labels_raw, augment=augment
            )
            result[split] = (batch_X, batch_y)
            logger.info(
                "  '%s': X=%s  y=%s", split, batch_X.shape, batch_y.shape
            )

        return result
