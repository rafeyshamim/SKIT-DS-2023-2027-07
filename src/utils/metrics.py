"""
Evaluation metrics for the 3D Medical Image classification task.

Wraps scikit-learn metric functions with a consistent interface so that
the rest of the codebase only imports from this module.
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_labels: Optional[List[str]] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Metrics computed:
        - Accuracy
        - Precision  (weighted by default)
        - Recall     (weighted by default)
        - F1-score   (weighted by default)
        - AUC-ROC    (one-vs-rest for multi-class; requires ``y_prob``)

    Args:
        y_true:       Ground-truth integer class labels, shape ``(N,)``.
        y_pred:       Predicted class indices, shape ``(N,)``.
        y_prob:       Predicted class probabilities, shape ``(N, C)``.
                      Required for AUC-ROC computation; ignored otherwise.
        class_labels: Human-readable class names (unused here, kept for
                      signature consistency with ``get_classification_report``).
        average:      Averaging strategy for multi-class metrics.
                      One of ``'weighted'``, ``'macro'``, ``'micro'``.

    Returns:
        Dictionary mapping metric names to their float values.
    """
    metrics: Dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(
        precision_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["f1_score"] = float(
        f1_score(y_true, y_pred, average=average, zero_division=0)
    )

    # AUC-ROC requires probability scores
    if y_prob is not None:
        try:
            num_classes = y_prob.shape[1]
            if num_classes == 2:
                metrics["auc_roc"] = float(
                    roc_auc_score(y_true, y_prob[:, 1])
                )
            else:
                metrics["auc_roc"] = float(
                    roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average=average
                    )
                )
        except Exception:  # noqa: BLE001
            # AUC is undefined when not all classes appear in y_true (small batches)
            metrics["auc_roc"] = float("nan")

    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional explicit list of class indices.

    Returns:
        2-D integer array of shape ``(C, C)``.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None,
) -> str:
    """
    Generate a full per-class classification report as a formatted string.

    Args:
        y_true:       Ground-truth labels.
        y_pred:       Predicted labels.
        class_labels: Optional list of human-readable class names.

    Returns:
        Multi-line string containing precision/recall/F1 per class.
    """
    labels = list(range(len(class_labels))) if class_labels is not None else None
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_labels,
        zero_division=0,
    )

