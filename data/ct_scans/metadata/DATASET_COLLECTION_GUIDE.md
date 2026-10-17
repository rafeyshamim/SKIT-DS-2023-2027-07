# Member 1 — Mayuri Agrawal: CT Scan Dataset Collection & Preprocessing Guide

**Role:** Data Preprocessing & 3D Reconstruction Pipeline  
**Assigned Milestones:**
1. Data Format & Preprocessing Technique Study (DICOM vs NIfTI, LPS vs RAS, Marching Cubes)
2. CT Scan Data Collection & Preprocessing (SimpleITK Volume Loading, Isotropic Resampling, HU Windowing)
3. 3D Organ Reconstruction Pipeline Development (Marching Cubes, Laplacian Smoothing, Mesh Decimation)
4. Validation Dataset Preparation & Reconstruction Evaluation (Dice, HD95, IoU, ASSD vs Ground Truth)
5. Reconstruction Module Integration Support (End-to-End Orchestrator & Geometry Repair)
6. Final Data Pipeline Review & Documentation

---

## 1. Curated CT Scan Datasets for Download

Here are the vetted, publicly available datasets selected specifically for this pipeline:

### Dataset 1: TotalSegmentator Benchmark (Zenodo) — *Top Recommendation*
* **Target Anatomies:** 117 anatomical structures (Liver, Spleen, Kidneys, Adrenal Glands, Gallbladder, Pancreas, Stomach, Aorta, IVC, Spine, Ribs, Lungs).
* **Format:** Compressed NIfTI (`.nii.gz`).
* **Why it's ideal:** Contains both high-resolution clinical CT scans and pixel-perfect ground-truth multi-organ masks, directly fulfilling Milestone 2, Milestone 3, and Milestone 4 (evaluation against ground truth).
* **Zenodo Record:** [https://zenodo.org/records/10047292](https://zenodo.org/records/10047292)
* **Sample / v1 Direct Link:** [https://doi.org/10.5281/zenodo.6802614](https://doi.org/10.5281/zenodo.6802614)
* **How to Upload:**
  1. Extract the `.zip` archive.
  2. Put CT scans (e.g., `s0001.nii.gz`) into `data/ct_scans/raw/nifti/volumes/`.
  3. Put corresponding segmentation masks into `data/ct_scans/raw/nifti/ground_truth_masks/`.

---

### Dataset 2: CHAOS Challenge (Combined Healthy Abdominal Organ Segmentation)
* **Target Anatomies:** Liver, Kidneys, Spleen.
* **Format:** Standard Hospital **DICOM series** (`.dcm` slices) + Ground Truth Masks.
* **Why it's ideal:** Satisfies the explicit requirement to **study and load DICOM slice series with SimpleITK (`ImageSeriesReader`)** and preserve slice ordering and spacing along the scanner Z-axis.
* **Zenodo Link:** [https://zenodo.org/record/3362844](https://zenodo.org/record/3362844)
* **Kaggle Mirror:** [https://www.kaggle.com/datasets/sardorabdirayimov/chaos-dataset](https://www.kaggle.com/datasets/sardorabdirayimov/chaos-dataset)
* **How to Upload:**
  1. Create a subfolder for each patient under `data/ct_scans/raw/dicom/<patient_name>/`.
  2. Copy the slice files (`.dcm`) into that subfolder.

---

### Dataset 3: Kaggle Chest CT-Scan Images
* **Target Anatomies:** Thoracic/Lung parenchyma, chest wall, tumor nodules.
* **Format:** Axial volumetric slices.
* **Why it's ideal:** Fastest, no-friction 1-click download from Kaggle for instant testing of lung window presets ($WL=-600, WW=1500$).
* **Kaggle Link:** [https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

---

### Dataset 4: AMOS 2022 (Abdominal Multi-Organ Benchmark)
* **Target Anatomies:** 15 abdominal organs across multi-vendor clinical CT scanners.
* **Format:** NIfTI (`.nii.gz`).
* **Zenodo Link:** [https://zenodo.org/records/7262600](https://zenodo.org/records/7262600)
* **Project GitHub:** [https://github.com/HzFu/AMOS](https://github.com/HzFu/AMOS)

---

## 2. Step-by-Step Upload Instructions

1. **Upload NIfTI CT Scans (`.nii` or `.nii.gz`)**:
   - Save clinical CT volume files to:  
     `data/ct_scans/raw/nifti/volumes/`
   - Save corresponding ground truth masks to:  
     `data/ct_scans/raw/nifti/ground_truth_masks/` (Use matching naming like `<name>_seg.nii.gz`)

2. **Upload DICOM Series**:
   - Save DICOM folders to:  
     `data/ct_scans/raw/dicom/<patient_id>/` (each folder containing `.dcm` slices)

3. **Validate Uploaded Data**:
   - Run the automated verification script:
     ```bash
     .\venv\Scripts\python.exe scripts/verify_ct_dataset.py
     ```
   - This verifies headers, tags, HU ranges, and generates `data/ct_scans/metadata/dataset_manifest.csv`.

---

## 3. Running Preprocessing & 3D Reconstruction Pipeline

Once data is in place, run the end-to-end pipeline:

```bash
$env:PYTHONPATH="data pre processing"
.\venv\Scripts\python.exe "data pre processing/run_pipeline_demo.py"
```

### Programmatic Python Usage:
```python
import sys
sys.path.insert(0, "data pre processing")

from core.pipeline import Medical3DReconstructionPipeline, PipelineConfig

# 1. Configure clinical pipeline parameters
config = PipelineConfig(
    target_spacing=(1.0, 1.0, 1.0),    # Isotropic 1mm voxel resampling
    hu_window="bone",                   # Preset: bone | soft_tissue | liver | lung
    normalize_method="min_max",         # Normalized to [0.0, 1.0]
    denoise_method="curvature_flow",    # Edge-preserving ITK filter
    isovalue=0.5,
    smooth_mesh=True,                   # Laplacian mesh smoothing
    decimate_reduction=0.2,             # 20% triangle decimation
    export_formats=("stl", "obj")
)

pipeline = Medical3DReconstructionPipeline(config=config)

# 2. Process DICOM or NIfTI scan
result = pipeline.process(
    input_source="data/ct_scans/raw/nifti/volumes/ct_scan_sample_01.nii.gz",
    output_dir="data/ct_scans/reconstructed_3d",
    organ_name="Vertebral_Spine"
)

print("Exported 3D Models:", result["exported_models"])
print("Mesh Surface Area (mm²):", result["mesh_statistics"]["surface_area_mm2"])
```
