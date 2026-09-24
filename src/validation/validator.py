"""
ModelValidator for Sprint 5 Final Model Validation.

Runs the final AI model against predefined test cases, compares predictions against
expected results, verifies confidence scores, and formats validation reports.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.disease_analysis.analyzer import DiseaseAnalyzer
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelValidator:
    """
    Validation suite for final model verification against known test cases.
    """

    def __init__(self, model: keras.Model, config: Dict[str, Any]):
        """
        Initialize ModelValidator.

        Args:
            model: Keras model instance to validate.
            config: Configuration dictionary.
        """
        self.model = model
        self.config = config
        self.analyzer = DiseaseAnalyzer(model, config)
        self.pipeline = PreprocessingPipeline(config)
        self.paths_cfg = config.get("paths", {})
        self.results_dir = self.paths_cfg.get("results_dir", "results")
        os.makedirs(self.results_dir, exist_ok=True)

    def validate_test_cases(
        self,
        test_volumes: List[np.ndarray],
        expected_labels: List[Union[int, str]],
        confidence_threshold: float = 0.50,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        Run model against predefined test cases and validate disease prediction outputs.

        Args:
            test_volumes: List of 3D or 4D raw test volume arrays.
            expected_labels: List of expected ground truth class indices or label names.
            confidence_threshold: Minimum required confidence threshold.
            save_report: Whether to write validation report JSON to disk.

        Returns:
            Dictionary containing overall validation results and individual case details.
        """
        logger.info("Executing final model validation across %d test cases...", len(test_volumes))

        case_results = []
        correct_count = 0
        total_cases = len(test_volumes)
        threshold_pass_count = 0

        for i, (vol, expected) in enumerate(zip(test_volumes, expected_labels)):
            preprocessed = self.pipeline.preprocess_single_volume(vol, augment=False)
            analysis = self.analyzer.analyze(preprocessed)

            pred_idx = analysis["predicted_class_index"]
            pred_label = analysis["predicted_class_label"]
            conf = analysis["confidence"]

            # Convert expected label comparison
            is_match = False
            if isinstance(expected, int):
                is_match = (pred_idx == expected)
            else:
                is_match = (str(pred_label).lower() == str(expected).lower()) or (str(pred_idx) == str(expected))

            if is_match:
                correct_count += 1

            meets_threshold = (conf >= confidence_threshold)
            if meets_threshold:
                threshold_pass_count += 1

            case_results.append({
                "case_id": i + 1,
                "expected": expected,
                "predicted_index": pred_idx,
                "predicted_label": pred_label,
                "confidence": float(conf),
                "is_correct": bool(is_match),
                "meets_threshold": bool(meets_threshold),
                "probabilities": {k: float(v) for k, v in analysis["probabilities"].items()},
            })

        accuracy = float(correct_count / total_cases) if total_cases > 0 else 0.0
        avg_confidence = float(np.mean([c["confidence"] for c in case_results])) if case_results else 0.0

        summary = {
            "total_test_cases": total_cases,
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "cases_meeting_threshold": threshold_pass_count,
            "average_confidence": avg_confidence,
            "confidence_threshold": confidence_threshold,
            "case_details": case_results,
        }

        logger.info(
            "Validation Summary | Accuracy: %.2f%% (%d/%d) | Avg Confidence: %.2f%%",
            accuracy * 100, correct_count, total_cases, avg_confidence * 100
        )

        if save_report:
            report_path = os.path.join(self.results_dir, "final_model_validation_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved validation report to %s", report_path)

        return summary
