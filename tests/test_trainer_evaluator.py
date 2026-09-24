"""
Unit tests for trainer and evaluator modules (Sprint 3).
"""
import os
import numpy as np
import pytest
import tensorflow as tf

from src.model.trainer import ModelTrainer, build_optimizer, build_callbacks
from src.model.evaluator import ModelEvaluator
from src.preprocessing.pipeline import PreprocessingPipeline


class TestTrainer:
    """Test suite for ModelTrainer and its helper functions."""

    def test_build_optimizer_valid(self, dummy_config):
        opt = build_optimizer(dummy_config)
        assert isinstance(opt, tf.keras.optimizers.Optimizer)

    def test_build_optimizer_invalid(self, dummy_config):
        cfg = dict(dummy_config)
        cfg["training"] = dict(dummy_config["training"])
        cfg["training"]["optimizer"] = "unsupported_optimizer"
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            build_optimizer(cfg)

    def test_build_callbacks(self, dummy_config, tmp_path):
        cfg = dict(dummy_config)
        cfg["paths"] = dict(dummy_config["paths"])
        cfg["paths"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
        cfg["paths"]["logs_dir"] = str(tmp_path / "logs")
        callbacks = build_callbacks(cfg)
        assert len(callbacks) >= 3

    def test_trainer_train_and_save(self, dummy_config, random_volume_batch, random_labels, tmp_path):
        cfg = dict(dummy_config)
        cfg["paths"] = dict(dummy_config["paths"])
        cfg["paths"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
        cfg["paths"]["logs_dir"] = str(tmp_path / "logs")
        cfg["paths"]["results_dir"] = str(tmp_path / "results")

        pipeline = PreprocessingPipeline(cfg)
        ds = pipeline.get_tf_dataset(random_volume_batch, random_labels, shuffle=False)

        trainer = ModelTrainer(cfg)
        history = trainer.train(ds, ds)

        assert "loss" in history.history
        assert os.path.exists(os.path.join(cfg["paths"]["results_dir"], "training_history.json"))


class TestEvaluator:
    """Test suite for ModelEvaluator."""

    def test_evaluator_evaluate(self, dummy_config, random_volume_batch, random_labels, tmp_path):
        cfg = dict(dummy_config)
        cfg["paths"] = dict(dummy_config["paths"])
        cfg["paths"]["results_dir"] = str(tmp_path / "results")
        cfg["paths"]["plots_dir"] = str(tmp_path / "plots")

        pipeline = PreprocessingPipeline(cfg)
        ds = pipeline.get_tf_dataset(random_volume_batch, random_labels, shuffle=False)

        trainer = ModelTrainer(cfg)
        evaluator = ModelEvaluator(trainer.model, cfg)

        metrics = evaluator.evaluate(ds, split_name="test")
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert os.path.exists(os.path.join(cfg["paths"]["results_dir"], "classification_report_test.txt"))
