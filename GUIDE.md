# MedVision 3D CT Diagnostic System Guide

Welcome to the **AI-Powered 3D Medical Image Reconstruction and Disease Analysis System**. This guide explains how to install, configure, and use all the components integrated in this project.

## 🎯 System Components

The project consists of three main components seamlessly integrated:
1. **Core AI Pipeline (`src/` & `main.py`)**: Handles data preprocessing, 3D CNN model training, 3D volume reconstruction, and evaluation.
2. **Backend API (`backend/`)**: A FastAPI-based robust enterprise-level backend with dual-database architecture (PostgreSQL and MongoDB) for API serving, DICOM management, and model inference.
3. **Frontend Application (`app.py`)**: A Streamlit interactive web interface for clinicians to upload scans and view 3D diagnostic results.

---

## ⚙️ Prerequisites & Setup

### 1. Environment Setup

Ensure you have Python 3.10+ installed. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate
```

### 2. Install Dependencies

All components share a unified `requirements.txt` file (with merge conflicts resolved):

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the System

### 1. Starting the Backend (FastAPI)

The backend provides the API that the frontend communicates with. It uses an in-memory SQLite database for relations (in development) and connects to MongoDB.

**Navigate to the `backend` directory and run:**
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Swagger Documentation**: Available at [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc Documentation**: Available at [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 2. Starting the Frontend (Streamlit)

The frontend provides an interactive User Interface for radiologist and clinical usage.

**Open a new terminal (activate your venv), navigate to the project root, and run:**
```bash
# Make sure you are in the root directory (Final Project)
streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

---

## 🧠 Using the AI Pipeline (CLI)

The core AI functions can be accessed directly from the command line using `main.py`. This is highly useful for research and development.

### Training the Model
Train the 3D CNN model using the configured datasets:
```bash
python main.py train --config config/config.yaml
```

### Evaluate the Model
Evaluate the pre-trained model:
```bash
python main.py evaluate --config config/config.yaml
```

### Perform Disease Analysis on a Scan
Get confidence scores and predictions on an existing NIfTI (`.nii`/`.nii.gz`) or NumPy (`.npy`) volume:
```bash
python main.py predict --input path/to/scan.nii.gz --config config/config.yaml
```

### Generate 3D Reconstructions
Generate Visualizations (MIP, slices, isosurfaces) for a given volume:
```bash
python main.py reconstruct --input path/to/scan.nii.gz --output-dir results/reconstruction/
```

### Run an End-to-End System Demo
This will run a synthetic demo covering preprocessing, inference, and visualization:
```bash
python main.py demo
```

---

## 🧪 Testing the Integration

We use `pytest` for the system's robust test coverage. Ensure you run this from the project root.
*Note: A `pytest.ini` was added to fix `sys.path` resolution for the test suites.*

**To run the Backend test suite:**
```bash
pytest backend/tests -v
```

**To run the Core AI test suite:**
```bash
pytest tests/ -v
```
*(If you encounter Application Control or DLL errors related to `tensorflow`, ensure that your environment permissions allow Python DLL imports, or run within a Docker container.)*

---

## 📖 Key Directories & Files

- `app.py`: Streamlit Frontend entry point.
- `main.py`: Command Line Interface for AI pipeline operations.
- `backend/app/main.py`: FastAPI Backend entry point.
- `config/config.yaml`: Core configuration values for the ML pipelines.
- `results/`: Output directory where plots and reconstructions are saved.
- `requirements.txt`: Unified dependency list for the entire repository.

---
**MedVision 3D CT System** - Designed for academic and clinical research purposes. 
*By Team SKIT/DS/2023-2027/CSE-F-07*
