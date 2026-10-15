# Medical Image Preprocessing & 3D Organ Reconstruction Pipeline

A clinical-grade, modular Python framework for medical image ingestion, preprocessing, multi-organ segmentation, and 3D surface mesh reconstruction from CT scans and MRI datasets (supporting DICOM, NIfTI, and BraTS HDF5 formats).

---

## 1. Project Overview & Milestones

This repository fulfills the 6 core development milestones:

1. **Data Format & Preprocessing Technique Study**: Comparative study between DICOM and NIfTI formats, LPS vs. RAS coordinate frames, affine spatial representations, and 3D reconstruction algorithms (FBP, Iterative, Marching Cubes).
2. **CT Scan Data Collection & Preprocessing**: SimpleITK-based volume loading, isotropic voxel resampling, clinical Hounsfield Unit (HU) windowing, intensity normalization, and edge-preserving denoising.
3. **3D Organ Reconstruction Pipeline Development**: Voxel-to-mesh reconstruction via Marching Cubes, Laplacian mesh smoothing, Quadric decimation, and multi-format 3D export (STL, OBJ, PLY).
4. **Validation Dataset Preparation & Reconstruction Evaluation**: Quantitative evaluation suite supporting Dice Similarity Coefficient (DSC), Jaccard Index (IoU), 95% Hausdorff Distance (HD95), Average Symmetric Surface Distance (ASSD), PSNR, and SSIM.
5. **Reconstruction Module Integration Support**: End-to-end integrated orchestrator (`Medical3DReconstructionPipeline`) with automatic validation, geometry repair, and JSON execution manifests.
6. **Final Data Pipeline Review & Documentation**: Comprehensive methodology review, unit test coverage, and documentation.

---

## 2. Mathematical Formulations & Methodology

### 2.1 Coordinate Space Transformations: DICOM (LPS) vs. NIfTI (RAS)
- **DICOM Standard (LPS)**:
  - $+X$: Left
  - $+Y$: Posterior (back)
  - $+Z$: Superior (head)
- **NIfTI Standard (RAS)**:
  - $+X$: Right
  - $+Y$: Anterior (front)
  - $+Z$: Superior (head)
- **Transformation Matrix**:
  $$\begin{bmatrix} X_{RAS} \\ Y_{RAS} \\ Z_{RAS} \\ 1 \end{bmatrix} = \begin{bmatrix} -1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_{LPS} \\ Y_{LPS} \\ Z_{LPS} \\ 1 \end{bmatrix}$$

### 2.2 CT Hounsfield Unit (HU) Windowing
Standard CT scanners produce attenuation values in Hounsfield Units:
$$HU = 1000 \times \frac{\mu - \mu_{\text{water}}}{\mu_{\text{water}} - \mu_{\text{air}}}$$
Given Window Center ($WL$) and Window Width ($WW$):
$$\text{Lower Bound} = WL - \frac{WW}{2}, \quad \text{Upper Bound} = WL + \frac{WW}{2}$$

| Clinical Preset | Window Center ($WL$) | Window Width ($WW$) | HU Range | Target Anatomy |
|:---|:---:|:---:|:---:|:---|
| **Bone** | $+400$ | $1800$ | $[-500, +1300]$ | Cortical bone, spine, ribs |
| **Soft Tissue** | $+40$ | $400$ | $[-160, +240]$ | Muscle, abdomen, mediastinum |
| **Liver** | $+60$ | $160$ | $[-20, +140]$ | Hepatic parenchyma, lesions |
| **Brain** | $+40$ | $80$ | $[0, +80]$ | Gray/white matter, infarcts |
| **Lung** | $-600$ | $1500$ | $[-1350, +150]$ | Pulmonary parenchyma |

### 2.3 Isotropic Voxel Resampling
Clinical CT scans typically feature anisotropic voxels (e.g., $0.8 \times 0.8 \times 2.5\text{ mm}$). Reconstructing directly creates severe stair-stepping artifacts along the slice axis. We resample to an isotropic grid ($1.0 \times 1.0 \times 1.0\text{ mm}$):
$$N'_i = \text{round}\left(N_i \times \frac{s_i}{s'_i}\right)$$
- Continuous CT intensities are interpolated using **B-Spline / Trilinear interpolation**.
- Discrete segmentation labels are interpolated using **Nearest-Neighbor interpolation**.

### 2.4 3D Geometric Reconstruction: Marching Cubes
The Marching Cubes algorithm iterates over eight-voxel cubes within the scalar field $f(x, y, z)$. Edges intersecting the target isovalue surface $c$ are determined, and vertex coordinates $v$ along edge $e_{ij}$ between voxels $v_i$ and $v_j$ are computed by linear interpolation:
$$v = v_i + \frac{c - f(v_i)}{f(v_j) - f(v_i)} (v_j - v_i)$$
Vertex positions in physical millimeter space are derived by mapping through origin $O$ and spacing $S$:
$$P_{\text{phys}} = O + (v \odot S)$$

### 2.5 Evaluation Metrics
- **Dice Similarity Coefficient (DSC)**:
  $$DSC = \frac{2 |X \cap Y|}{|X| + |Y|}$$
- **Jaccard Index (IoU)**:
  $$IoU = \frac{|X \cap Y|}{|X \cup Y|} = \frac{DSC}{2 - DSC}$$
- **95th Percentile Hausdorff Distance (HD95)**:
  $$HD_{95}(A, B) = \max\left(P_{95}\left(\min_{b \in B} \|a - b\|\right), P_{95}\left(\min_{a \in A} \|b - a\|\right)\right)$$

---

## 3. Directory Structure

```
data_preprocess/
├── README.md                      # Comprehensive technical documentation
├── requirements.txt               # Dependencies
├── phantom_generator.py           # Anatomical CT phantom generator
├── run_pipeline_demo.py           # End-to-end execution script
├── core/
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Clinical HU window presets & defaults
│   ├── data_formats_study.py      # Format & reconstruction technique study
│   ├── data_loader.py             # DICOM series, NIfTI, and BraTS loaders
│   ├── preprocessor.py            # Resampling, windowing, normalization, denoising
│   ├── reconstructor_3d.py        # Marching Cubes, Laplacian smoothing, decimation, STL/OBJ export
│   ├── evaluator.py               # Dice, IoU, HD95, ASSD, PSNR, SSIM
│   └── pipeline.py                # End-to-end pipeline orchestrator
└── tests/
    └── test_pipeline.py           # Automated unit test suite
```

---

## 4. Quickstart & Usage

### 4.1 Running the Full Pipeline Demo
Execute the complete multi-milestone demonstration:
```bash
python run_pipeline_demo.py
```
This will:
1. Parse the BraTS20 Training Metadata CSV.
2. Generate an anatomical multi-organ CT scan dataset (`synthetic_abdominal_ct.nii.gz` and `synthetic_multiorgan_gt.nii.gz`).
3. Resample the scan to isotropic spacing, apply bone/soft-tissue windowing, and denoise with curvature flow.
4. Extract watertight 3D meshes for Vertebral Bone, Liver, and Kidneys.
5. Export `.stl` and `.obj` models to `pipeline_output/`.
6. Compute validation metrics against ground truth.

### 4.2 Running Unit Tests
```bash
python -m unittest tests/test_pipeline.py
```

### 4.3 Programmatic Python Usage

```python
from core.pipeline import Medical3DReconstructionPipeline, PipelineConfig

config = PipelineConfig(
    target_spacing=(1.0, 1.0, 1.0),
    hu_window="bone",
    denoise_method="curvature_flow",
    smooth_mesh=True,
    decimate_reduction=0.2,
    export_formats=("stl", "obj")
)

pipeline = Medical3DReconstructionPipeline(config=config)
result = pipeline.process(
    input_source="path/to/ct_scan.nii.gz", # or path to DICOM folder
    output_dir="./output_3d",
    organ_name="Vertebral_Spine"
)

print("3D Model exported:", result["exported_models"])
print("Surface Area (mm²):", result["mesh_statistics"]["surface_area_mm2"])
```
