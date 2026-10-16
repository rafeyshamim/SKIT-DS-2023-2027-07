# AI-Powered 3D Image Reconstruction and Disease Analysis

> **Final Year Project**

An AI-powered system focused on **3D image reconstruction and disease analysis** using Artificial Intelligence, Machine Learning, and Computer Vision techniques. The project aims to transform medical imaging data into meaningful 3D representations and assist with AI-based analysis for identifying potential disease-related patterns.

---

## 👥 Team Members

| Name | Role |
|---|---|
| **Naman Verma** | Team Member |
| **Mohd Rafey** | Team Lead  |
| **Mayuri Agarwal** | Team Member |
| **Mohit Chaudhary** | Team Member |

---

## 📌 Project Overview

Medical imaging plays an important role in the detection, analysis, and monitoring of diseases. However, conventional 2D medical images can sometimes make it difficult to understand the complete three-dimensional structure of anatomical regions.

Our project, **AI-Powered 3D Image Reconstruction and Disease Analysis**, explores the use of AI and computer vision to reconstruct 3D representations from medical imaging data and perform intelligent analysis on the reconstructed information.

The system is designed as an academic and research-oriented project to demonstrate how modern AI techniques can be applied to medical image processing and analysis.

---

## 🎯 Objectives

The major objectives of this project are:

- Develop an AI-based pipeline for **3D image reconstruction**.
- Process and analyze medical imaging data using computer vision techniques.
- Extract meaningful information from medical images.
- Explore AI-assisted **disease analysis**.
- Provide an intuitive interface for visualizing reconstructed 3D data.
- Demonstrate the potential of AI in medical image analysis.
- Build a modular system that can be extended for future research.

---

## ✨ Key Features

- 🧠 **AI-Based Image Analysis**
- 🧊 **3D Image Reconstruction**
- 🔬 **Disease Analysis**
- 🖼️ **Medical Image Processing**
- 📊 **AI-Assisted Results**
- 📈 **Visualization of Analysis**
- 💻 **User-Friendly Interface**
- 🔄 **Modular and Extensible Architecture**

> **Note:** Specific models, datasets, reconstruction techniques, and supported medical conditions will be documented here as the project implementation is finalized.

---

## 🔄 System Workflow

```text
        Medical Imaging Data
                 │
                 ▼
       ┌───────────────────┐
       │ Image Preprocessing│
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ AI-Based Feature  │
       │     Extraction    │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ 3D Reconstruction │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ 3D Visualization  │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ Disease Analysis  │
       └─────────┬─────────┘
                 │
                 ▼
          Analysis / Results
```

---

## 🏗️ High-Level Architecture

```text
┌───────────────────────────────┐
│       User / Medical Data     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Data Preprocessing       │
│  • Cleaning                   │
│  • Normalization              │
│  • Image Preparation          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        AI / ML Pipeline       │
│  • Feature Extraction         │
│  • Image Analysis             │
│  • Reconstruction              │
└───────────────┬───────────────┘
                │
          ┌─────┴─────┐
          ▼           ▼
┌────────────────┐ ┌────────────────┐
│ 3D Reconstruction│ │Disease Analysis│
└───────┬────────┘ └───────┬────────┘
        │                  │
        └────────┬─────────┘
                 ▼
      ┌──────────────────────┐
      │ Visualization &      │
      │ Analysis Results     │
      └──────────────────────┘
```

---

## 🛠️ Technologies

### Technology Stack

- **Programming Language:** Python 3.10+
- **AI / ML Framework:** TensorFlow 2.13+ / Keras
- **Medical Imaging:** NiBabel (NIfTI), MedMNIST3D (OrganMNIST3D, etc.)
- **Computer Vision & Scientific Computing:** NumPy, SciPy (`scipy.ndimage`), scikit-image (Marching Cubes)
- **3D Reconstruction & Visualization:** Plotly (interactive 3D WebGL isosurfaces), Matplotlib, Seaborn
- **Metrics & Evaluation:** scikit-learn (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Configuration & Utilities:** PyYAML, TQDM, Python standard logging
- **Testing:** Pytest, Pytest-cov

---

## 📂 Project Structure

```text
AI-3D-Image-Reconstruction/
│
├── config/
│   └── config.yaml          # Unified parameters (model, preprocessing, training, paths)
│
├── data/
│   ├── raw/                 # Raw datasets (NIfTI scans, MedMNIST downloads)
│   ├── processed/           # Preprocessed cached volumes
│   └── README.md
│
├── docs/
│   ├── sprint1_input_data_specification.md
│   ├── sprint2_architecture_specification.md
│   ├── sprint3_training_evaluation_specification.md
│   ├── sprint4_model_optimization_packaging_specification.md
│   ├── sprint5_final_model_validation_specification.md
│   └── sprint6_integration_testing_specification.md
│
├── models/
│   ├── checkpoints/         # Model weights & best checkpoint saves
│   └── README.md
│
├── results/
│   ├── plots/               # Confusion matrix, ROC curves, training curves
│   └── reconstruction/      # MIP, orthogonal slice PNGs, 3D HTML isosurfaces
│
├── src/
│   ├── disease_analysis/    # Prediction, probability distribution, confidence scoring
│   ├── inference/           # Model packaging, TFLite export, standalone InferenceEngine
│   ├── model/               # 3D CNN architecture, ModelTrainer, ModelEvaluator
│   ├── preprocessing/       # Volume loader, resizer, normalizer, pipeline
│   ├── reconstruction/      # Slicing, MIP projections, 3D marching cubes meshes
│   ├── validation/          # ModelValidator suite for test cases
│   └── utils/               # Config loader, logger factory, metric calculations
│
├── tests/                   # Pytest test suite (fixtures, preprocessing, model, trainer)
│
├── main.py                  # Unified command-line interface (CLI)
├── requirements.txt         # Project dependencies
├── setup.py                 # Package setup and build specification
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd "Final Project"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

#### Windows
```bash
venv\Scripts\activate
```

#### Linux / macOS
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Web Application & Interactive UI (React + FastAPI)

MedVision 3D includes a full-stack, end-to-end clinical interface built with **React 19 + Vite** and a high-performance **FastAPI** backend:

### Architecture:
- **Frontend**: React 19, Lucide Icons, HTML5 Canvas 2D/3D Rendering, Glassmorphic Clinical Cyberpunk Design System (`http://localhost:3000`).
- **Backend**: FastAPI, SQLAlchemy + SQLite, NoSQL Document Store, 3D CNN Inference Engine (`http://127.0.0.1:8000`).
- **Pipeline**: Automated multiplanar slice reformatting, 3D marching cubes isosurface extraction, 3D CNN lesion localization, and Fleischner Society clinical reporting.

### 🌟 4 Core End-to-End Screens:
1. **CT Scan Ingestion & Upload (`Screen 1`)**:
   - Supports multi-part DICOM, NIfTI (.nii, .nii.gz), ZIP slice archives, and Numpy (.npy) volumes.
   - Includes **Instant Demo Mode** button to synthesize calibrated 32-slice thoracic CT scans with lung parenchyma.
   - Patient registration & MRN tracking.
2. **3D Multiplanar Organ & Slice Scrubber (`Screen 2`)**:
   - Multiplanar 2D slice scrubber (Axial, Coronal, Sagittal views).
   - Automated animation playback, scrubbing slider, and slice jumping.
   - Clinical window presets (Lung, Bone, Soft Tissue).
   - Real-time AI Saliency & Heatmap layer with opacity control.
   - Interactive 3D WebGL / Canvas volumetric organ isosurface with rotation, zoom, and 3D bounding box overlay.
3. **AI Disease Analysis & Quantitative Findings (`Screen 3`)**:
   - 3D CNN primary diagnosis and circular confidence percentage meter.
   - Multi-class softmax probability breakdown.
   - Volumetric metrics: Total lung volume (cm³), lesion volume (mm³), lung involvement percentage (%).
   - Detected 3D lesion table with 3D centroid coordinates and malignancy scoring.
   - Printable Fleischner Society clinical diagnostic report with radiologist sign-off.
4. **AI Model Training & Online Dataset Dashboard (`Screen 4`)**:
   - Online MedMNIST3D dataset ingestion & verification status (31.1 MB verified from Zenodo).
   - Background training trigger with selectable epoch intervals.
   - Live loss and accuracy convergence progression graphs.
   - Checkpoint inspection and model metadata.

### 🚀 Running the Full-Stack Application:

#### Option A: One-Command Full-Stack Runner
```bash
python scripts/run_system.py
```
This automatically launches both the FastAPI backend (`http://127.0.0.1:8000`) and the React frontend (`http://localhost:3000`).

#### Option B: Manual Execution

**1. Launch FastAPI Backend:**
```bash
venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

**2. Launch React Frontend:**
```bash
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your web browser.

**3. Run Automated End-to-End Pipeline Test:**
```bash
python scripts/test_e2e.py
```

---

## 🚀 Unified CLI Usage

The project also provides a unified CLI via `main.py`:

### 1. Download / Verify Online Dataset (MedMNIST3D)
```bash
python scripts/download_dataset.py
```

### 2. Run Preprocessing (Sprint 1)
```bash
python main.py preprocess --config config/config.yaml
```

### 3. Train 3D CNN Model on Online Dataset (Sprints 2 & 3)
```bash
python main.py train --config config/config.yaml
```

### 4. Evaluate Model on Test Split (Sprint 3)
```bash
python main.py evaluate --config config/config.yaml
```

### 5. Disease Analysis & Confidence Scoring
```bash
python main.py predict --input path/to/scan.nii.gz --config config/config.yaml
```

### 6. 3D Visual Reconstruction & Projections
```bash
python main.py reconstruct --input path/to/scan.nii.gz --output-dir results/reconstruction/
```

### 7. Package Model for Deployment (Sprint 4)
```bash
python main.py package --export-name 3d_cnn_packaged --export-tflite --quantize
```

### 8. Run Final Model Validation (Sprint 5)
```bash
python main.py validate --num-test-cases 5 --threshold 0.50
```

### 9. Run End-to-End Demonstration
```bash
python main.py demo
```

### 10. Run Pytest Test Suite
```bash
pytest
```

---

## 📊 Results

The final version of this section will contain:

- 3D reconstruction examples
- Input vs. reconstructed output
- Disease analysis results
- Model performance metrics
- Accuracy / precision / recall where applicable
- Visualization screenshots
- Performance comparison

### Example

```text
Input Image
     │
     ▼
AI Processing
     │
     ▼
3D Reconstruction
     │
     ▼
Disease Analysis
     │
     ▼
Final Visualization
```

---

## 🔬 Research & Development

This project explores the intersection of:

- Artificial Intelligence
- Machine Learning
- Computer Vision
- Medical Image Processing
- 3D Reconstruction
- Image Segmentation
- Pattern Recognition
- AI-Assisted Disease Analysis

The system is intended to demonstrate the potential applications of these technologies in medical imaging research.

---

## 🔮 Future Scope

Possible future improvements include:

- Support for additional medical imaging modalities.
- Improved 3D reconstruction quality.
- Integration of more advanced deep learning models.
- Support for additional diseases and conditions.
- Improved segmentation and localization.
- Faster inference and reconstruction.
- Cloud-based deployment.
- Interactive 3D visualization.
- Integration with medical imaging standards.
- Larger and more diverse datasets.
- Extensive validation using clinically relevant datasets.

---

## ⚠️ Disclaimer

This project is developed as a **final-year academic and research project**.

The disease analysis functionality is intended for **research and educational purposes only** and should not be considered a substitute for diagnosis, treatment, or medical advice from a qualified healthcare professional.

Any AI-generated result should be interpreted by appropriately qualified medical professionals before being used for clinical decision-making.

---

## 👨‍💻 Contributors

**Naman Verma**  
**Mohd Rafey**  
**Mayuri Agarwal**  
**Mohit Chaudhary**

---

## 📄 License

This project is currently intended for academic and educational purposes.

A formal license will be added to the repository after the project requirements and ownership terms are finalized.

---

## ⭐ Acknowledgements

We would like to acknowledge our faculty, mentors, research resources, open-source technologies, and datasets that contribute to the development of this project.

---

<p align="center">
  <b>AI-Powered 3D Image Reconstruction and Disease Analysis</b>
  <br>
  Final Year Project
</p>
