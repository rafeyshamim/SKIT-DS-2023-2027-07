from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.db.session import init_db
from app.db.mongo import mongo_manager
from app.services.model_service import model_service
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle:
    - Sets up logging
    - Initializes relational database tables
    - Connects to MongoDB cluster
    - Loads and warms up the 3D CNN deep learning model
    """
    setup_logging()
    logger.info("Initializing MedVision 3D CT Diagnostic Backend...")
    
    # Initialize Relational Database Schema
    try:
        init_db()
        logger.info("SQL database schema initialized.")
    except Exception as e:
        logger.error(f"SQL database initialization warning: {e}")

    # Connect to MongoDB
    mongo_manager.connect()

    # Preload and warm up 3D CNN model
    if not model_service.model.is_loaded:
        model_service.model._initialize_or_load_weights()
    logger.info("3D CNN inference model is ready.")

    yield

    # Clean shutdown
    mongo_manager.close()
    logger.info("Backend service shut down cleanly.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Enterprise-grade Medical AI Backend for Volumetric 3D CT Scan Analysis.\n\n"
        "Features:\n"
        "- **DICOM / NIfTI Upload**: Multi-part chunked streaming with format validation\n"
        "- **3D Volume Reconstruction**: Hounsfield Unit (HU) conversion, lung tissue windowing, isotropic resampling\n"
        "- **3D CNN Deep Learning Inference**: Volumetric lesion detection, 3D bounding boxes, malignancy scoring\n"
        "- **Dual-Database Architecture**: PostgreSQL for ACID relational integrity (patients, scans, audit runs) "
        "and MongoDB for flexible volumetric DICOM metadata & slice heatmaps\n"
        "- **Clinical Diagnostic Reports**: Automated Fleischner Society-aligned reporting with radiologist sign-off"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred while processing the medical data request."}
    )


# Mount API V1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_v1": settings.API_V1_STR,
        "health_check": f"{settings.API_V1_STR}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
