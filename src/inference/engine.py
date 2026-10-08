"""
Standalone deployable inference engine for 3D CNN medical imaging model.

Sprint 4 — Model Optimization & Packaging for Inference.
"""
import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class InferenceEngine:
    """
    Standalone inference engine capable of loading packaged models independently
    of the training code and executing fast disease analysis inference.
    """

    def __init__(
        self,
        package_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize InferenceEngine.

        Args:
            package_dir: Path to directory containing packaged model artifacts.
            config: Optional override configuration dictionary.
        """
        self.package_dir = package_dir
        self.metadata_path = os.path.join(package_dir, "metadata.json")

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.config = config or self._build_config_from_metadata()
        self.pipeline = PreprocessingPipeline(self.config)

        self.keras_model_path = os.path.join(package_dir, "model.keras")
        self.tflite_model_path = os.path.join(package_dir, "model.tflite")

        self.model = None
        self.tflite_interpreter = None
        self._load_model()

    def _build_config_from_metadata(self) -> Dict[str, Any]:
        """Reconstruct config parameters from metadata manifest if available."""
        prep_meta = self.metadata.get("preprocessing", {})
        target_size = prep_meta.get("target_size", self.metadata.get("input_shape", [64, 64, 64, 1])[:3])
        num_channels = prep_meta.get("num_channels", 1)
        norm_method = prep_meta.get("normalize_method", "min_max")

        return {
            "model": {
                "input_shape": self.metadata.get("input_shape", [64, 64, 64, 1]),
            },
            "data": {
                "source": self.metadata.get("data_source", "medmnist"),
                "num_classes": self.metadata.get("num_classes", 11),
                "class_names": self.metadata.get("class_names", {}),
            },
            "preprocessing": {
                "target_size": target_size,
                "num_channels": num_channels,
                "normalize_method": norm_method,
                "clip_values": False,
                "augmentation": {"enabled": False},
            },
            "paths": {
                "raw_dir": "data/raw",
                "raw_data_dir": "data/raw",
                "processed_dir": "data/processed",
                "data_dir": "data",
                "checkpoint_dir": "models/checkpoints",
                "export_dir": self.package_dir,
                "results_dir": "results",
            },
            "training": {
                "batch_size": 1,
                "seed": 42,
            },
        }

    def _load_model(self):
        """Load available packaged model backend."""
        if os.path.exists(self.keras_model_path):
            logger.info("Loading packaged Keras model from %s", self.keras_model_path)
            self.model = keras.models.load_model(self.keras_model_path, compile=False)
        elif os.path.exists(self.tflite_model_path):
            logger.info("Loading packaged TFLite interpreter from %s", self.tflite_model_path)
            self.tflite_interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
            self.tflite_interpreter.allocate_tensors()
        else:
            raise FileNotFoundError(
                f"No valid packaged model found in {self.package_dir}. "
                "Expected model.keras or model.tflite."
            )

    def predict_volume(
        self,
        volume: np.ndarray,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference on a single 3D volume.

        Args:
            volume: 3D or 4D raw/preprocessed volume numpy array.
            preprocess: Whether to pass the volume through preprocessing first.

        Returns:
            Dictionary containing prediction analysis results.
        """
        if preprocess:
            processed_vol = self.pipeline.preprocess_single_volume(volume, augment=False)
        else:
            processed_vol = volume

        # Ensure batch dimension (1, D, H, W, C)
        if processed_vol.ndim == 4:
            batch_input = np.expand_dims(processed_vol, axis=0)
        else:
            batch_input = processed_vol

        batch_input = batch_input.astype(np.float32)

        if self.model is not None:
            probabilities = self.model.predict(batch_input, verbose=0)[0]
        else:
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()
            self.tflite_interpreter.set_tensor(input_details[0]['index'], batch_input)
            self.tflite_interpreter.invoke()
            probabilities = self.tflite_interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])

        class_names = self.metadata.get("class_names", {})
        label = class_names.get(str(pred_idx), class_names.get(pred_idx, f"Class_{pred_idx}"))

        prob_dict = {}
        num_classes = len(probabilities)
        for i in range(num_classes):
            lbl = class_names.get(str(i), class_names.get(i, f"Class_{i}"))
            prob_dict[lbl] = float(probabilities[i])

        return {
            "predicted_class_index": pred_idx,
            "predicted_class_label": label,
            "confidence": confidence,
            "probabilities": prob_dict,
        }
