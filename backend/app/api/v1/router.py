from fastapi import APIRouter
from app.api.v1.endpoints import health, patients, scans, inference, reports, training

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health & Diagnostics"])
api_router.include_router(patients.router, prefix="/patients", tags=["Patient Records Management"])
api_router.include_router(scans.router, prefix="/scans", tags=["CT Scan Upload & 3D Processing"])
api_router.include_router(inference.router, prefix="/inference", tags=["3D CNN Model Inference"])
api_router.include_router(reports.router, prefix="/reports", tags=["Clinical Diagnostic Reports"])
api_router.include_router(training.router, prefix="/training", tags=["3D CNN Model Training & Dataset"])
