"""Application Configuration and Environment Settings.

Defines global configuration for:
- API metadata and routing
- Storage file paths (uploads, processed 3D volumes)
- Database credentials (PostgreSQL relational and MongoDB NoSQL)
- 3D CNN Deep Learning model architecture, target tensor shapes, and checkpoints
"""

import os
from typing import List, Tuple

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    HAS_SETTINGS = True
except ImportError:
    from pydantic import BaseModel
    HAS_SETTINGS = False
    BaseSettings = BaseModel


class Settings(BaseSettings):
    """Global system configuration loaded from environment or defaults."""

    PROJECT_NAME: str = "MedVision 3D CT Backend"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "medvision_secret_jwt_key_2026_supersecure")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

    # Base filesystem directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROJECT_ROOT: str = os.path.dirname(BASE_DIR)

    # Volumetric Storage Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "data", "uploads"))
    PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", os.path.join(BASE_DIR, "data", "processed"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))

    # Relational Database (PostgreSQL with SQLite fallback for local development)
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgrespassword")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "medvision_db")

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'medvision.db')}"
    )

    # MongoDB NoSQL Settings
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://root:mongopassword@localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "medvision_metadata")

    # 3D CNN Model Configuration & Hyperparameters
    MODEL_NAME: str = os.getenv("MODEL_NAME", "MedNet-3D-CNN")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1.2.0")
    MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", os.path.join(BASE_DIR, "models", "weights"))
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cpu")

    # Volumetric Input Specification: (Channels, Depth, Height, Width)
    TARGET_VOLUME_SHAPE: Tuple[int, int, int, int] = (1, 32, 64, 64)
    DEFAULT_CONFIDENCE_THRESHOLD: float = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.45"))

    # Diagnostic Classification Labels
    MODEL_CLASSES: List[str] = [
        "Normal / No Nodule",
        "Benign Pulmonary Nodule",
        "Malignant Suspicion",
    ]

    # CORS Allowed Origins
    CORS_ORIGINS: List[str] = ["*"]

    if HAS_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )

    def get_model_checkpoint_candidates(self) -> List[str]:
        """Returns prioritized list of candidate filesystem paths to search for trained model weights."""
        return [
            os.path.join(self.PROJECT_ROOT, "models", "checkpoints", "best_model.keras"),
            os.path.join(self.BASE_DIR, "models", "checkpoints", "best_model.keras"),
            os.path.join(self.BASE_DIR, "models", "exported", "model.keras"),
            os.path.join(self.MODEL_WEIGHTS_PATH, "best_model.keras"),
            os.path.join(self.MODEL_WEIGHTS_PATH, "model.keras"),
        ]


settings = Settings()

# Ensure runtime directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
os.makedirs(settings.MODEL_WEIGHTS_PATH, exist_ok=True)
