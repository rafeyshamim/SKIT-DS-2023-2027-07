"""
Model packager for optimizing and serializing trained 3D CNN models for deployment.

Sprint 4 — Model Optimization & Packaging for Inference.
"""
import json
import os
import shutil
from typing import Any, Dict, Optional, Union

import tensorflow as tf
from tensorflow import keras

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelPackager:
    """
    Handles packaging and optimizing trained 3D CNN models for deployment/inference.

    Supports exporting as Keras SavedModel format, TFLite format with optional
    dynamic-range optimization/quantization, and metadata manifest creation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelPackager.

        Args:
            config: Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        self.paths_cfg = config.get("paths", {})
        self.model_cfg = config.get("model", {})
        self.data_cfg = config.get("data", {})

        self.export_dir = self.paths_cfg.get("export_dir", "models/exported")
        os.makedirs(self.export_dir, exist_ok=True)

    def package_model(
        self,
        model: keras.Model,
        export_name: str = "3d_cnn_packaged",
        export_tflite: bool = True,
        quantize_tflite: bool = False,
    ) -> Dict[str, str]:
        """
        Package, optimize, and save the model along with its metadata manifest.

        Args:
            model: Trained Keras model instance.
            export_name: Base name/directory for exported artifact.
            export_tflite: Whether to export a .tflite version.
            quantize_tflite: Whether to apply dynamic range quantization to TFLite.

        Returns:
            Dictionary containing paths to generated artifacts.
        """
        package_path = os.path.join(self.export_dir, export_name)
        os.makedirs(package_path, exist_ok=True)

        artifacts = {}

        # 1. Save Native Keras SavedModel (.keras format)
        keras_model_path = os.path.join(package_path, "model.keras")
        model.save(keras_model_path)
        artifacts["keras_model"] = keras_model_path
        logger.info("Saved Keras packaged model to %s", keras_model_path)

        # 2. Save TFLite Model (if requested)
        if export_tflite:
            tflite_path = os.path.join(package_path, "model.tflite")
            self._export_tflite(model, tflite_path, quantize=quantize_tflite)
            artifacts["tflite_model"] = tflite_path

        # 3. Save Metadata Manifest
        metadata = self._generate_metadata(model)
        metadata_path = os.path.join(package_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        artifacts["metadata"] = metadata_path
        logger.info("Saved export metadata manifest to %s", metadata_path)

        return artifacts

    def _export_tflite(
        self,
        model: keras.Model,
        output_path: str,
        quantize: bool = False,
    ) -> str:
        """Convert and save model to TFLite format."""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                logger.info("Enabling dynamic range quantization for TFLite export.")
            tflite_model = converter.convert()
            with open(output_path, "wb") as f:
                f.write(tflite_model)
            logger.info("Exported TFLite model successfully to %s", output_path)
            return output_path
        except Exception as err:
            logger.warning("Failed to export TFLite model: %s", err)
            return ""

    def _generate_metadata(self, model: keras.Model) -> Dict[str, Any]:
        """Generate metadata dictionary for the packaged model."""
        return {
            "model_name": model.name or "3D_CNN_Medical",
            "input_shape": list(self.model_cfg.get("input_shape", [64, 64, 64, 1])),
            "num_classes": self.data_cfg.get("num_classes", 11),
            "class_names": self.data_cfg.get("class_names", {}),
            "preprocessing": self.config.get("preprocessing", {}),
            "parameters_count": int(model.count_params()),
            "framework": f"TensorFlow {tf.__version__}",
        }
