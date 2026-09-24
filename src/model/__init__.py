"""Model package — Sprint 2 & 3."""
from src.model.architecture import build_3d_cnn, get_model_from_config
from src.model.trainer import ModelTrainer
from src.model.evaluator import ModelEvaluator

__all__ = ["build_3d_cnn", "get_model_from_config", "ModelTrainer", "ModelEvaluator"]
