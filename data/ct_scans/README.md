# CT Scan Dataset Repository
**AI-Powered 3D Image Reconstruction and Disease Analysis**  
**Member 1: Mayuri Agrawal** — *Data Preprocessing & 3D Reconstruction Pipeline*

---

## 📂 Directory Structure

```
data/ct_scans/
├── raw/                                # Raw clinical medical scans (Unmodified scanner data)
│   ├── dicom/                          # Multi-slice DICOM series (.dcm)
│   │   └── patient_001_abdominal_ct/   # [Sample Baseline] 64-slice DICOM series (SimpleITK-ready)
│   └── nifti/                          # Volumetric 3D clinical scans
│       ├── volumes/                    # CT volume files (.nii, .nii.gz)
│       └── ground_truth_masks/         # Multi-organ expert annotations for evaluation
│
├── preprocessed/                       # Output of SimpleITK Preprocessing (Milestone 2)
│   ├── isotropic_1mm/                  # Resampled to 1.0 x 1.0 x 1.0 mm isotropic spacing
│   ├── windowed_bone/                  # Bone preset (WL: +400, WW: 1800) -> [0, 1]
│   ├── windowed_soft_tissue/           # Soft tissue preset (WL: +40, WW: 400) -> [0, 1]
│   └── windowed_lung/                  # Lung preset (WL: -600, WW: 1500) -> [0, 1]
│
├── validation_ground_truth/            # Test datasets reserved for Milestone 4 validation
│   ├── test_volumes/                   # Evaluation CT scans
│   └── test_masks/                     # Ground truth masks (Liver, Kidneys, Spleen, Spine)
│
├── reconstructed_3d/                   # Output of Milestone 3 3D mesh reconstruction
│   ├── stl/                            # Watertight binary STL meshes (3D printing & CAD)
│   ├── obj/                            # Wavefront OBJ files with vertex normals
│   └── interactive_html/               # Plotly interactive 3D WebGL visualizations
│
└── metadata/                           # Dataset documentation and manifests
    ├── dataset_manifest.csv            # Auto-generated catalog of all registered CT scans
    ├── download_sources.json           # Curated direct URLs, Zenodo DOIs, and Kaggle links
    └── DATASET_COLLECTION_GUIDE.md     # Step-by-step instructions for data collection & upload
```

---

## 🚀 Quick Verification

Whenever you upload new CT files (DICOM or NIfTI), run this single verification command:

```bash
.\venv\Scripts\python.exe scripts/verify_ct_dataset.py
```

This will automatically:
1. Verify SimpleITK compatibility for all DICOM slice series and NIfTI volumes.
2. Calculate slice count, matrix dimensions, voxel spacing, and Hounsfield Unit (HU) statistics.
3. Match volumes with corresponding ground-truth segmentation masks.
4. Auto-update `data/ct_scans/metadata/dataset_manifest.csv`.

---

## 📥 Where to Upload Your Datasets

| Scan Type | Format | Target Folder | Example Files |
|:---|:---:|:---|:---|
| **Clinical DICOM Series** | `.dcm` | `data/ct_scans/raw/dicom/<patient_name>/` | `slice_0001.dcm`, `slice_0002.dcm`, ... |
| **Volumetric CT Scan** | `.nii` / `.nii.gz` | `data/ct_scans/raw/nifti/volumes/` | `patient_002_ct.nii.gz` |
| **Ground Truth Organ Mask** | `.nii` / `.nii.gz` | `data/ct_scans/raw/nifti/ground_truth_masks/` | `patient_002_ct_seg.nii.gz` |

---

## 🔬 Recommended Curated Datasets for Download

1. **TotalSegmentator v2 (Zenodo)**: [zenodo.org/records/10047292](https://zenodo.org/records/10047292)  
   *117 anatomical structures in NIfTI format. The gold standard for multi-organ 3D reconstruction.*
2. **CHAOS Challenge (Zenodo / Kaggle)**: [zenodo.org/record/3362844](https://zenodo.org/record/3362844)  
   *Clinical DICOM series of abdominal CT scans with matching liver segmentations.*
3. **Kaggle Chest CT-Scan Images**: [kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)  
   *Fast, immediate download for thoracic/pulmonary analysis.*
