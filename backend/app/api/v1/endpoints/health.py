from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db
from app.db.mongo import get_mongo
from app.services.model_service import model_service
from app.core.config import settings

router = APIRouter()


@router.get("", summary="System Health & Diagnostic Status")
def check_health(db: Session = Depends(get_db)):
    """
    Returns system operational health including:
    - API gateway status
    - Relational Database (PostgreSQL / SQLite) connectivity
    - NoSQL Database (MongoDB) connectivity
    - 3D CNN Model loading status & version
    """
    # Test Relational DB
    relational_db_status = "HEALTHY"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        relational_db_status = f"UNHEALTHY: {str(e)}"

    # Test MongoDB
    mongo = get_mongo()
    mongo_status = "CONNECTED" if mongo.is_connected else "STANDALONE_FALLBACK_ACTIVE"

    return {
        "status": "ONLINE",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "database": {
            "relational_db": relational_db_status,
            "nosql_mongo": mongo_status
        },
        "ml_inference": {
            "model_name": settings.MODEL_NAME,
            "model_version": settings.MODEL_VERSION,
            "is_loaded": model_service.model.is_loaded,
            "target_tensor_shape": settings.TARGET_VOLUME_SHAPE
        }
    }
