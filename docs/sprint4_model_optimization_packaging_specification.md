# Sprint 4 — Model Optimization & Packaging Specification

**Project Title:** AI-Powered 3D Medical Image Reconstruction and Disease Analysis
**Project ID:** SKIT/DS/2023-2027/CSE-F-07
**Team Lead:** Mohammad Rafey
**Sprint Window:** 16-11-2026 to 31-12-2026

---

## 1. Overview & Objectives

Sprint 4 focuses on optimizing trained 3D CNN models and packaging them into standalone, deployable inference artifacts. The resulting system ensures that:
- Models can be packaged with all required configuration metadata.
- Inference can be executed independently of the training framework or raw data loading pipeline.
- Lightweight TFLite exports (with optional dynamic-range quantization) are supported for edge/embedded or fast CPU deployment.

---

## 2. Model Packaging Architecture (`ModelPackager`)

The `ModelPackager` (`src/inference/packager.py`) exports:
1. **Keras SavedModel (`model.keras`)**: Standard cross-platform weight and architecture bundle.
2. **TFLite Format (`model.tflite`)**: Quantized or unquantized flatbuffer representation for mobile/edge execution.
3. **Metadata Manifest (`metadata.json`)**: Contains input shape, class mappings, preprocessing specs, parameter counts, and framework versions.

```
models/exported/<export_name>/
├── model.keras         # Packaged Keras model
├── model.tflite        # Optimized TFLite flatbuffer
└── metadata.json       # Metadata & preprocessing parameters
```

---

## 3. Deployment & Standalone Inference Engine (`InferenceEngine`)

The `InferenceEngine` (`src/inference/engine.py`) provides:
- Independent loading of packaged Keras `.keras` or TFLite `.tflite` models.
- Auto-reconstruction of preprocessing specifications directly from `metadata.json`.
- Simple API for volume predictions: `engine.predict_volume(raw_volume)`.

---

## 4. CLI Subcommand Usage

Package a model from checkpoint:
```bash
python main.py package --export-name 3d_cnn_packaged --export-tflite --quantize
```
