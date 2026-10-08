# Sprint 1 — CNN Input Data Specification

**Project Title:** AI-Powered 3D Medical Image Reconstruction and Disease Analysis  
**Project ID:** SKIT/DS/2023-2027/CSE-F-07  
**Team Lead:** Mohammad Rafey  
**Sprint Window:** 10-08-2026 to 31-08-2026  

---

## 1. Input Volume Format & Dimensions

The 3D Convolutional Neural Network processes volumetric medical scans represented as 5D tensors `(batch_size, D, H, W, C)`:

| Dimension | Default Value | Description |
|-----------|---------------|-------------|
| **D** (Depth / Slices) | `64` | Axial slice dimension |
| **H** (Height) | `64` | In-plane image height |
| **W** (Width) | `64` | In-plane image width |
| **C** (Channels) | `1` | Single-channel volumetric grayscale intensity |
| **Dtype** | `float32` | 32-bit floating point precision |
| **Intensity Range** | `[0.0, 1.0]` | Min-Max normalized intensity (or $z \sim \mathcal{N}(0, 1)$ for z-score) |

*Note: For testing and fast prototyping with MedMNIST3D (`organmnist3d`), spatial dimensions can be configured to `[28, 28, 28]` or `[32, 32, 32]` via `config/config.yaml`.*

---

## 2. Supported Raw Input Sources

1. **NIfTI (`.nii`, `.nii.gz`)**: Standard clinical neuroimaging and CT/MRI format loaded using `nibabel`. Scanner voxel coordinate frames and affine transforms are respected.
2. **MedMNIST3D (`.npz`)**: Standardized 3D biomedical benchmark volumes (e.g. `organmnist3d`, `nodulemnist3d`, `synapsemnist3d`, `fracturemnist3d`).

---

## 3. Preprocessing Pipeline Specification

Every raw input volume passes through a strict four-stage preprocessing chain before model entry:

```
Raw Volumetric Data (D_in, H_in, W_in)
               │
               ▼
   [1] Spatial Resampling / Resizing
       • Trilinear interpolation (scipy.ndimage.zoom, order=1)
       • Target dimensions: (D_target, H_target, W_target)
               │
               ▼
   [2] Intensity Normalization
       • Optional HU clipping: [clip_min, clip_max]
       • Min-Max scaling: (x - x_min) / (x_max - x_min + eps) -> [0.0, 1.0]
       • Or Z-Score standardization: (x - μ) / (σ + eps)
               │
               ▼
   [3] Data Augmentation (Training Only)
       • Random 3-axis reflections (50% probability)
       • Random in-plane rotation (±15°)
       • Random volumetric scaling / zooming (±10%)
               │
               ▼
   [4] Channel Dimension Expansion
       • Expand dims: (D, H, W) -> (D, H, W, 1)
               │
               ▼
   Final 3D CNN Input Tensor (1, D, H, W, 1) [float32]
```

---

## 4. Verification and Reproducibility

- Configured through `config/config.yaml` under `preprocessing` and `data`.
- Automated test coverage implemented in `tests/test_preprocessing.py`.
- Validation checks assert tensor rank, spatial bounds, and non-NaN values before feeding into the computational graph.
