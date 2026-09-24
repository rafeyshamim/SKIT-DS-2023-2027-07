"""
Disease analysis and prediction module.

Sprint 3 — Disease Prediction & Confidence Scoring.

Wraps the trained 3-D CNN to produce, per input volume:

    - Predicted disease / organ class label
    - Confidence score (max softmax probability)
    - Full probability distribution over all classes
    - Boolean flag: confidence ≥ configured threshold

⚠️  DISCLAIMER
    This module produces AI-generated research predictions only.
    Results MUST NOT be interpreted as clinical diagnoses.
    Always consult a qualified medical professional.
"""
from typing import Dict, List, Optional

import numpy as np
from tensorflow import keras

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DiseaseAnalyzer:
    """
    High-level interface for disease analysis on a preprocessed volumetric image.

    Accepts volumes that have already been processed by
    :class:`~src.preprocessing.pipeline.PreprocessingPipeline`, i.e.
    shape ``(D, H, W, 1)`` or ``(1, D, H, W, 1)``, dtype ``float32``,
    values in ``[0, 1]``.

    Usage::

        analyzer = DiseaseAnalyzer(model, config)
        result   = analyzer.analyze(preprocessed_volume)
        print(result["predicted_class_label"], result["confidence"])
    """

    def __init__(
        self,
        model: keras.Model,
        config: dict,
        class_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            model:        Loaded and compiled Keras model.
            config:       Configuration dictionary.
            class_labels: Override for class label names.  Defaults to
                          ``config['data']['class_labels']``, then
                          ``['Class_0', 'Class_1', …]``.
        """
        self.model = model

        # Resolve class labels
        num_classes = int(config["data"]["num_classes"])
        self.class_labels: List[str] = (
            class_labels
            or config["data"].get("class_labels")
            or [f"Class_{i}" for i in range(num_classes)]
        )

        self.confidence_threshold: float = float(
            config.get("inference", {}).get("confidence_threshold", 0.5)
        )

        logger.info(
            "DiseaseAnalyzer ready | %d classes | confidence threshold=%.2f",
            len(self.class_labels),
            self.confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Single-volume analysis
    # ------------------------------------------------------------------

    def analyze(self, volume: np.ndarray) -> Dict[str, object]:
        """
        Analyze a single preprocessed volumetric input and return the
        disease prediction with its confidence score.

        Args:
            volume: Preprocessed volume of shape ``(D, H, W, 1)`` or
                    ``(1, D, H, W, 1)``.  dtype ``float32``.

        Returns:
            Dictionary with keys:

            ``predicted_class_index`` (int)
                0-based index of the predicted class.

            ``predicted_class_label`` (str)
                Human-readable class / disease name.

            ``confidence`` (float)
                Confidence score in ``[0, 1]`` — the maximum softmax
                probability across all classes.

            ``probabilities`` (dict)
                Full ``{label: probability}`` distribution.

            ``above_threshold`` (bool)
                ``True`` when ``confidence >= confidence_threshold``.

        Raises:
            ValueError: If *volume* has an incompatible shape.
        """
        volume = self._ensure_batch_dim(volume)
        self._validate_input_shape(volume)

        logger.debug("Running inference | input shape: %s", volume.shape)
        probabilities: np.ndarray = (
            self.model(volume, training=False).numpy()[0]
        )

        predicted_index = int(np.argmax(probabilities))
        confidence      = float(probabilities[predicted_index])
        predicted_label = self._index_to_label(predicted_index)

        prob_distribution = {
            self._index_to_label(i): float(probabilities[i])
            for i in range(len(probabilities))
        }

        result: Dict[str, object] = {
            "predicted_class_index": predicted_index,
            "predicted_class_label": predicted_label,
            "confidence":            confidence,
            "probabilities":         prob_distribution,
            "above_threshold":       confidence >= self.confidence_threshold,
        }

        logger.info(
            "Prediction → '%s'  |  confidence=%.4f (%.1f%%)  |  "
            "above_threshold=%s",
            predicted_label,
            confidence,
            confidence * 100,
            result["above_threshold"],
        )
        return result

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        volumes: np.ndarray,
    ) -> List[Dict[str, object]]:
        """
        Analyze a batch of preprocessed volumetric inputs.

        Args:
            volumes: Batch array of shape ``(N, D, H, W, 1)``.

        Returns:
            List of result dictionaries (one per input volume).

        Raises:
            ValueError: If *volumes* is not 5-D.
        """
        if volumes.ndim != 5:
            raise ValueError(
                f"analyze_batch expects shape (N, D, H, W, 1), "
                f"got: {volumes.shape}"
            )

        logger.info("Batch inference on %d volumes ...", len(volumes))
        all_probs: np.ndarray = self.model(volumes, training=False).numpy()

        results = []
        for probs in all_probs:
            idx         = int(np.argmax(probs))
            confidence  = float(probs[idx])
            results.append(
                {
                    "predicted_class_index": idx,
                    "predicted_class_label": self._index_to_label(idx),
                    "confidence":            confidence,
                    "probabilities": {
                        self._index_to_label(j): float(probs[j])
                        for j in range(len(probs))
                    },
                    "above_threshold": confidence >= self.confidence_threshold,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_batch_dim(self, volume: np.ndarray) -> np.ndarray:
        """Add a batch dimension if the volume is 4-D ``(D, H, W, 1)``."""
        if volume.ndim == 4:
            return np.expand_dims(volume, axis=0)   # → (1, D, H, W, 1)
        if volume.ndim == 5:
            return volume
        raise ValueError(
            f"Expected volume shape (D, H, W, 1) or (1, D, H, W, 1), "
            f"got: {volume.shape}"
        )

    def _validate_input_shape(self, volume: np.ndarray) -> None:
        """Assert that the batch input shape matches the model's expectation."""
        expected = tuple(self.model.input_shape[1:])   # (D, H, W, C)
        actual   = tuple(volume.shape[1:])
        if actual != expected:
            raise ValueError(
                f"Input shape mismatch:\n"
                f"  Model expects : {expected}\n"
                f"  Received      : {actual}\n"
                "Ensure the volume has been preprocessed through "
                "PreprocessingPipeline.preprocess_single_volume()."
            )

    def _index_to_label(self, index: int) -> str:
        """Convert a class index to its human-readable label."""
        if 0 <= index < len(self.class_labels):
            return self.class_labels[index]
        return f"Class_{index}"
