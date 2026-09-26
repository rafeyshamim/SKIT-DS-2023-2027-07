# MedVision 3D CT Diagnostic Backend

Enterprise-grade medical AI backend for Computed Tomography (CT) scan upload, 3D volumetric reconstruction, deep learning inference with 3D CNNs, and automated clinical diagnostic report generation.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?logo=mongodb&logoColor=white)](https://mongodb.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-Passing-success)](https://pytest.org)

---

## 1. System Architecture & Highlights

- **FastAPI Core**: Async REST API with high-throughput streaming, Pydantic v2 data models, and live interactive Swagger UI (`/docs`).
- **Hybrid Dual-Database Architecture**:
  - **PostgreSQL**: ACID relational transactions for `patients`, `ct_scans`, `inference_runs`, and `reports`.
  - **MongoDB**: Schema-flexible NoSQL store for high-dimensional DICOM header tags, 3D lesion bounding boxes, and per-slice abnormality heatmaps.
- **3D CT Reconstruction Pipeline**:
  - Automatic DICOM / ZIP / NIfTI archive parsing.
  - Conversion from raw attenuation to calibrated **Hounsfield Units (HU)**.
  - Clinical **Lung Windowing** (Window Center: -600 HU, Window Width: 1500 HU).
  - Trilinear isotropic volume resampling to standardized tensor dimensions `(1, 1, 32, 64, 64)`.
- **Packaged 3D CNN Inference Service (`MedNet-3D`)**:
  - Full 3D convolutional receptive fields capturing volumetric continuity across axial slices.
  - Multi-class classification: *Normal*, *Benign Nodule*, *Malignant Suspicion*.
  - 3D Grad-CAM blob localization extracting 3D bounding boxes $[z, y, x]$, centroid coordinates, and lesion volume ($\text{mm}^3$).
- **Automated Clinical Reporting**:
  - Generates diagnostic reports with findings, impression, and follow-up guidance aligned with **Fleischner Society Guidelines**.
- **Containerized Deployment**:
  - Multi-stage Dockerfile running as non-root user (`medvision`) for healthcare security compliance.
  - Complete `docker-compose.yml` orchestrating API, PostgreSQL 16, MongoDB 7.0, Redis 7, and Adminer.

---

## 2. Project Directory Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── health.py        # System health & model status
│   │   │   │   │   ├── patients.py      # Patient registration & CRUD
│   │   │   │   │   ├── scans.py         # CT scan upload & 3D reconstruction
│   │   │   │   │   ├── inference.py     # 3D CNN inference & slice heatmaps
│   │   │   │   │   └── reports.py       # Clinical diagnostic report generation
│   │   │   │   └── router.py            # API V1 router
│   │   ├── core/
│   │   │   ├── config.py                # Environment configuration
│   │   │   └── logging.py               # Structured logging
│   │   ├── db/
│   │   │   ├── session.py               # SQLAlchemy engine & session factory
│   │   │   └── mongo.py                 # MongoDB client & fallback document store
│   │   ├── models/
│   │   │   ├── sql/                     # PostgreSQL models (Patient, CTScan, etc.)
│   │   │   └── nosql/                   # MongoDB document schemas (DicomMetadata, etc.)
│   │   ├── schemas/                     # Pydantic v2 request/response contracts
│   │   ├── services/
│   │   │   ├── storage_service.py       # Chunked streaming upload & storage
│   │   │   ├── dicom_processor.py       # 3D HU conversion & windowing pipeline
│   │   │   ├── model_service.py         # 3D CNN model architecture & Grad-CAM
│   │   │   └── report_service.py        # Fleischner diagnostic report generator
│   │   └── main.py                      # FastAPI lifespan application entrypoint
│   ├── tests/                           # Complete Pytest integration test suite
│   ├── Dockerfile                       # Multi-stage production container
│   ├── requirements.txt                 # Backend Python dependencies
│   └── .env.example                     # Environment variables template
├── scripts/
│   └── demo_pipeline.py                 # Interactive end-to-end pipeline demonstration
├── docker-compose.yml                   # Multi-container production deployment
├── ARCHITECTURE.md                      # Detailed system architecture document
├── API_CONTRACT.md                      # Complete REST API specification
└── README.md                            # Main project overview
```

---

## 3. Quickstart & Local Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
```bash
# Clone and navigate into directory
cd backend

# Install dependencies
pip install -r requirements.txt
```

### Run the FastAPI Server
```bash
# Start backend server with hot-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 4. Running Tests

The test suite validates the entire medical imaging lifecycle (health, patient records, scan upload, 3D volume reconstruction, 3D CNN inference, and report generation):

```bash
# Run all unit and integration tests
python -m pytest backend/tests -v
```

---

## 5. Interactive Pipeline Demo

To execute the entire end-to-end medical workflow in a single command:

```bash
python scripts/demo_pipeline.py
```

This script will:
1. Initialize the backend and warm up the 3D CNN model.
2. Register a patient with a unique Medical Record Number (MRN).
3. Upload and reconstruct a high-resolution 3D thoracic CT volume in Hounsfield Units.
4. Execute 3D CNN forward inference, printing 3D nodule bounding boxes and malignancy scores.
5. Generate and print a clinical diagnostic report adhering to Fleischner Society guidelines.

---

## 6. Docker Deployment

Deploy the complete stack (FastAPI backend, PostgreSQL 16, MongoDB 7.0, Redis 7, and Adminer):

```bash
# Build and start all microservices
docker compose up --build -d

# Check service status
docker compose ps

# View backend logs
docker compose logs -f backend
```

Services exposed:
- **FastAPI Backend**: `http://localhost:8000`
- **PostgreSQL**: `localhost:5432` (`user=postgres`, `password=postgrespassword`, `db=medvision_db`)
- **MongoDB**: `localhost:27017` (`user=root`, `password=mongopassword`)
- **Adminer DB Manager**: `http://localhost:8080`
