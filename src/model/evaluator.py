"""
Model evaluation pipeline for the 3D CNN medical image classifier.

Sprint 3 — Model Training & Evaluation.

Computes and persists:
    - Accuracy, Precision, Recall, F1-score, AUC-ROC
    - Per-class classification report (text)
    - Confusion matrix (PNG heatmap)
    - Training history curves (loss + accuracy PNG)
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from src.utils.logger import setup_logger
from src.utils.metrics import (
    compute_classification_metrics,
    get_classification_report,
    get_confusion_matrix,
)

logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Evaluates the trained 3-D CNN model on a test dataset.

    Usage::

        evaluator = ModelEvaluator(model, config, class_labels)
        metrics   = evaluator.evaluate(test_dataset, split_name="test")
        evaluator.plot_training_history("results/training_history.json")
    """

    def __init__(
        self,
        model: keras.Model,
        config: dict,
        class_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            model:        Trained (or loaded) Keras model.
            config:       Configuration dictionary.
            class_labels: Human-readable label names.  Falls back to
                          ``config['data']['class_labels']``.
        """
        self.model        = model
        self.config       = config
        self.class_labels = class_labels or config["data"].get("class_labels")
        self.results_dir  = config["paths"].get("results_dir", "results/")
        self.plots_dir    = config["paths"].get("plots_dir",   "results/plots/")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir,   exist_ok=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        dataset: tf.data.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run forward-pass inference on *dataset* and collect outputs.

        Args:
            dataset: Batched ``tf.data.Dataset`` yielding ``(X, y)`` tuples.

        Returns:
            Tuple ``(y_true, y_pred, y_prob)`` where

            - *y_true*: Ground-truth integer labels, shape ``(N,)``.
            - *y_pred*: Argmax predictions,          shape ``(N,)``.
            - *y_prob*: Softmax probabilities,        shape ``(N, C)``.
        """
        all_true: List[np.ndarray] = []
        all_prob: List[np.ndarray] = []

        for X_batch, y_batch in dataset:
            prob_batch = self.model(X_batch, training=False).numpy()
            all_prob.append(prob_batch)
            all_true.append(y_batch.numpy())

        y_prob = np.concatenate(all_prob, axis=0)
        y_true = np.concatenate(all_true, axis=0).reshape(-1)
        y_pred = np.argmax(y_prob, axis=1)
        return y_true, y_pred, y_prob

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        split_name: str = "test",
    ) -> Dict[str, float]:
        """
        Run complete evaluation: metrics + classification report +
        confusion matrix plot.

        Args:
            test_dataset: Batched ``tf.data.Dataset``.
            split_name:   Tag used for file naming (e.g. ``'test'``).

        Returns:
            Dictionary of metric names → float values.
        """
        logger.info("Evaluating on '%s' split ...", split_name)
        y_true, y_pred, y_prob = self.predict(test_dataset)

        metrics = compute_classification_metrics(
            y_true, y_pred,
            y_prob=y_prob,
            class_labels=self.class_labels,
        )

        report = get_classification_report(y_true, y_pred, self.class_labels)
        cm     = get_confusion_matrix(y_true, y_pred)

        # --- Log results ---
        logger.info("=== Evaluation Metrics (%s) ===", split_name)
        for metric_name, value in metrics.items():
            logger.info("  %-22s: %.4f", metric_name, value)
        logger.info("\n%s", report)

        # --- Persist ---
        self._save_metrics(metrics, split_name)
        self._save_classification_report(report, split_name)
        self._plot_confusion_matrix(cm, split_name)

        return metrics

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_metrics(
        self, metrics: Dict[str, float], split_name: str
    ) -> None:
        path = os.path.join(
            self.results_dir, f"evaluation_metrics_{split_name}.json"
        )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Evaluation metrics → %s", path)

    def _save_classification_report(
        self, report: str, split_name: str
    ) -> None:
        path = os.path.join(
            self.results_dir, f"classification_report_{split_name}.txt"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
        logger.info("Classification report → %s", path)

    def _plot_confusion_matrix(
        self, cm: np.ndarray, split_name: str
    ) -> None:
        n = len(cm)
        fig_size = max(8, n)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_labels or list(range(n)),
            yticklabels=self.class_labels or list(range(n)),
            ax=ax,
        )
        ax.set_title(
            f"Confusion Matrix — {split_name.capitalize()} Split",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        path = os.path.join(
            self.plots_dir, f"confusion_matrix_{split_name}.png"
        )
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Confusion matrix → %s", path)

    # ------------------------------------------------------------------
    # Training history plotting
    # ------------------------------------------------------------------

    def plot_training_history(self, history_json_path: str) -> None:
        """
        Plot loss and accuracy curves from a saved ``training_history.json``.

        Args:
            history_json_path: Path to the JSON file written by
                               :meth:`~src.model.trainer.ModelTrainer._save_history`.
        """
        if not os.path.exists(history_json_path):
            logger.warning(
                "Training history not found at '%s' — skipping plot.",
                history_json_path,
            )
            return

        with open(history_json_path, "r", encoding="utf-8") as fh:
            history = json.load(fh)

        epochs = range(1, len(history.get("loss", [])) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Loss ---
        axes[0].plot(epochs, history.get("loss",     []), label="Train Loss",     linewidth=2)
        axes[0].plot(epochs, history.get("val_loss", []), label="Val Loss",       linewidth=2, linestyle="--")
        axes[0].set_title("Loss Curve",  fontsize=12)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Accuracy (handle both 'accuracy' and 'acc' key names) ---
        acc_key     = "accuracy"     if "accuracy"     in history else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
        axes[1].plot(epochs, history.get(acc_key,     []), label="Train Accuracy", linewidth=2)
        axes[1].plot(epochs, history.get(val_acc_key, []), label="Val Accuracy",   linewidth=2, linestyle="--")
        axes[1].set_title("Accuracy Curve", fontsize=12)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("3D CNN — Training History", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.plots_dir, "training_history.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Training history plot → %s", path)
