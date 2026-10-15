# Sprint 3 — Model Training & Evaluation Specification

**Project Title:** AI-Powered 3D Medical Image Reconstruction and Disease Analysis  
**Project ID:** SKIT/DS/2023-2027/CSE-F-07  
**Team Lead:** Mohammad Rafey  
**Sprint Window:** 01-10-2026 to 15-11-2026  

---

## 1. Training Pipeline Architecture

The training engine is implemented in `src/model/trainer.py` (`ModelTrainer`) and provides:

- **Loss Function:** `sparse_categorical_crossentropy` (or categorical cross-entropy) with numerical stability checks.
- **Optimizer Support:** Adam, SGD (with Nesterov momentum), and RMSprop with customizable initial learning rate.
- **Deterministic Seeding:** Synchronized seeds across NumPy and TensorFlow (`seed: 42`).
- **Callback Orchestration:**
  1. `ModelCheckpoint`: Automatically tracks minimum `val_loss` and persists `best_model.keras`.
  2. `EarlyStopping`: Halts execution if validation loss ceases to improve over a configurable patience window, preventing overfitting and resource exhaustion.
  3. `ReduceLROnPlateau`: Dynamically decays learning rate by a factor of 0.5 upon detecting loss plateaus.
  4. `CSVLogger`: Records per-epoch training and validation loss/accuracy to `training_history.csv`.
  5. `TensorBoard`: Logs scalar curves and weight histograms.

---

## 2. Model Evaluation & Metric Tracking

The evaluation engine is implemented in `src/model/evaluator.py` (`ModelEvaluator`) and `src/utils/metrics.py`:

- **Classification Metrics:**
  - Accuracy: Overall correct predictions fraction.
  - Precision, Recall, and F1-score: Macro- and weighted-averages to account for medical class imbalance.
  - AUC-ROC: Multi-class One-vs-Rest area under the receiver operating characteristic curve calculated directly on softmax probability outputs.
- **Visual Diagnostics Generated:**
  - Confusion Matrix Heatmap (`confusion_matrix_test.png`)
  - Per-class classification report text export (`classification_report_test.txt`)
  - Training and Validation Loss / Accuracy Learning Curves (`training_curves.png`)

---

## 3. Disease Analysis & Confidence Scoring

The disease prediction module is implemented in `src/disease_analysis/analyzer.py` (`DiseaseAnalyzer`):

- **Inference Process:**
  - Input: Preprocessed volume `(D, H, W, 1)` or batched `(N, D, H, W, 1)`.
  - Output:
    - Predicted Class Index & Human-readable Disease Label.
    - Softmax Confidence Score: $c = \max_k P(Y=k \mid X)$.
    - Complete Probability Distribution across all anatomical/disease classes.
    - Clinical Threshold Verification: Boolean flag indicating if $c \ge \tau_{\text{threshold}}$ (default $\tau = 0.50$).

---

## 4. 3D Medical Reconstruction Integration

Implemented in `src/reconstruction/reconstructor.py` (`VolumeReconstructor`):
- Orthogonal planar slicing (Axial, Coronal, Sagittal).
- Maximum Intensity Projections (MIP) across depth, height, and width.
- 3D Isosurface extraction via Marching Cubes (`scikit-image`) rendered into interactive 3D WebGL meshes (`plotly` HTML).
