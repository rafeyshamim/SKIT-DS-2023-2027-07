"""
Training pipeline for the 3D CNN medical image classification model.

Sprint 3 — Model Training & Evaluation.

Handles:
    - Model compilation (optimizer, loss, metrics from config)
    - Callback setup:
        * ModelCheckpoint  (saves best model by val_loss)
        * EarlyStopping
        * ReduceLROnPlateau
        * TensorBoard
        * CSVLogger
    - Training and validation loops via ``keras.Model.fit``
    - Training history persistence (JSON)
    - Best model loading for downstream evaluation/inference
"""
import json
import os
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model.architecture import get_model_from_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimizer(config: Dict[str, Any]) -> keras.optimizers.Optimizer:
    """
    Instantiate the optimizer specified in ``training.optimizer``.

    Args:
        config: Full configuration dictionary.

    Returns:
        Configured Keras optimizer.

    Raises:
        ValueError: If the optimizer name is not supported.
    """
    train_cfg = config["training"]
    lr   = float(train_cfg["learning_rate"])
    name = str(train_cfg["optimizer"]).lower()

    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(
            learning_rate=lr, momentum=0.9, nesterov=True
        )
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)

    raise ValueError(
        f"Unsupported optimizer: '{name}'. "
        "Choose 'adam', 'sgd', or 'rmsprop' in config.yaml."
    )


def build_callbacks(config: Dict[str, Any]) -> list:
    """
    Assemble Keras training callbacks from the configuration.

    Callbacks:
        1. **ModelCheckpoint**  — saves best model weights to
           ``paths.checkpoint_dir/best_model.keras``.
        2. **EarlyStopping**    — halts training when ``val_loss`` stops
           improving (configurable patience).
        3. **ReduceLROnPlateau** — halves the learning rate on plateaus.
        4. **TensorBoard**      — logs scalars and histograms.
        5. **CSVLogger**        — appends per-epoch metrics to a CSV file.

    Args:
        config: Full configuration dictionary.

    Returns:
        List of configured :class:`keras.callbacks.Callback` instances.
    """
    train_cfg   = config["training"]
    paths_cfg   = config["paths"]

    checkpoint_dir = paths_cfg["checkpoint_dir"]
    logs_dir       = paths_cfg["logs_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir,       exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, "best_model.keras")
    callbacks       = []

    # 1 — Model Checkpoint
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )

    # 2 — Early Stopping
    es_cfg = train_cfg.get("early_stopping", {})
    if es_cfg.get("enabled", True):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=str(es_cfg.get("monitor", "val_loss")),
                patience=int(es_cfg.get("patience", 10)),
                restore_best_weights=bool(
                    es_cfg.get("restore_best_weights", True)
                ),
                verbose=1,
            )
        )

    # 3 — Reduce LR on Plateau
    rlr_cfg = train_cfg.get("reduce_lr", {})
    if rlr_cfg.get("enabled", True):
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=str(rlr_cfg.get("monitor", "val_loss")),
                patience=int(rlr_cfg.get("patience", 5)),
                factor=float(rlr_cfg.get("factor", 0.5)),
                min_lr=float(rlr_cfg.get("min_lr", 1e-6)),
                verbose=1,
            )
        )

    # 4 — TensorBoard
    tb_cfg = train_cfg.get("tensorboard", {})
    if tb_cfg.get("enabled", True):
        try:
            import tensorboard  # verify availability
            tb_log_dir = os.path.join(logs_dir, "tensorboard")
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=tb_log_dir,
                    histogram_freq=1,
                    write_graph=True,
                )
            )
        except (ImportError, Exception) as exc:
            logger.warning(
                "TensorBoard callback could not be initialized (%s). Skipping.", exc
            )


    # 5 — CSV Logger
    csv_path = os.path.join(logs_dir, "training_history.csv")
    callbacks.append(keras.callbacks.CSVLogger(csv_path, append=True))

    logger.info("Callbacks ready.  Best model → %s", best_model_path)
    return callbacks


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Encapsulates the complete training workflow for the 3-D CNN model.

    Usage::

        trainer   = ModelTrainer(config)
        history   = trainer.train(train_dataset, val_dataset)
        best_model = trainer.load_best_model()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Configuration dictionary from
                    :func:`~src.utils.config_loader.load_config`.
        """
        self.config     = config
        self.train_cfg  = config["training"]
        self.paths_cfg  = config["paths"]

        # Reproducibility
        seed = int(self.train_cfg.get("seed", 42))
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.model = get_model_from_config(config)
        self._compile_model()

    def _compile_model(self) -> None:
        """Compile with the optimizer, loss, and metrics from config."""
        optimizer = build_optimizer(self.config)
        loss      = self.train_cfg["loss"]
        metrics   = self.train_cfg.get("metrics", ["accuracy"])

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        logger.info(
            "Model compiled | optimizer=%s | loss=%s | metrics=%s",
            self.train_cfg["optimizer"], loss, metrics,
        )
        self.model.summary(print_fn=logger.info)

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
    ) -> keras.callbacks.History:
        """
        Execute the training loop.

        Args:
            train_dataset: Batched ``tf.data.Dataset`` for training.
            val_dataset:   Batched ``tf.data.Dataset`` for validation.

        Returns:
            Keras :class:`~keras.callbacks.History` object.
        """
        epochs    = int(self.train_cfg["epochs"])
        callbacks = build_callbacks(self.config)

        logger.info("Training started — up to %d epochs.", epochs)
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        self._save_history(history.history)
        logger.info("Training complete.")
        return history

    def _save_history(self, history_dict: dict) -> None:
        """Persist the training history to ``results/training_history.json``."""
        results_dir = self.paths_cfg.get("results_dir", "results/")
        os.makedirs(results_dir, exist_ok=True)
        history_path = os.path.join(results_dir, "training_history.json")

        # Convert numpy scalars to plain Python floats for JSON serialisation
        serializable = {
            key: [float(v) for v in vals]
            for key, vals in history_dict.items()
        }
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)

        logger.info("Training history saved → %s", history_path)

    def load_best_model(self) -> keras.Model:
        """
        Load the best-performing model saved during training.

        Returns:
            Loaded :class:`keras.Model`.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        model_path = os.path.join(
            self.paths_cfg["checkpoint_dir"], "best_model.keras"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No saved model at: '{model_path}'.\n"
                "Run train.py first to generate the checkpoint."
            )
        logger.info("Loading best model from: %s", model_path)
        return keras.models.load_model(model_path)
