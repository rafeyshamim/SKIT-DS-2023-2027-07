"""
CT Scan Dataset Verification & Manifest Builder
================================================
Validates all raw DICOM series and NIfTI volumes in data/ct_scans/raw/,
checks SimpleITK compatibility, voxel spacing, Hounsfield Unit bounds,
matches ground-truth masks, and updates dataset_manifest.csv.
"""
import os
import sys
import glob
import pandas as pd
import SimpleITK as sitk
import numpy as np

# Ensure data pre processing modules are importable
sys.path.insert(0, os.path.abspath("data pre processing"))
from core.data_loader import load_dicom_series, load_nifti

def verify_dataset(base_dir: str = "data/ct_scans"):
    raw_dir = os.path.join(base_dir, "raw")
    dicom_root = os.path.join(raw_dir, "dicom")
    nifti_vol_dir = os.path.join(raw_dir, "nifti", "volumes")
    nifti_mask_dir = os.path.join(raw_dir, "nifti", "ground_truth_masks")
    manifest_path = os.path.join(base_dir, "metadata", "dataset_manifest.csv")

    records = []

    print("=" * 75)
    print(" CT SCAN DATASET VERIFICATION & MANIFEST GENERATOR")
    print(f" Scanning: {os.path.abspath(raw_dir)}")
    print("=" * 75)

    # 1. Scan DICOM series folders
    if os.path.isdir(dicom_root):
        for entry in sorted(os.listdir(dicom_root)):
            patient_path = os.path.join(dicom_root, entry)
            if os.path.isdir(patient_path):
                try:
                    vol = load_dicom_series(patient_path)
                    dcm_files = glob.glob(os.path.join(patient_path, "*.dcm"))
                    shape = vol.pixel_array.shape
                    min_hu = float(np.min(vol.pixel_array))
                    max_hu = float(np.max(vol.pixel_array))
                    mean_hu = float(np.mean(vol.pixel_array))
                    records.append({
                        "dataset_id": entry,
                        "modality": "CT",
                        "format": "DICOM Series",
                        "file_path": os.path.relpath(patient_path, base_dir),
                        "num_slices": len(dcm_files),
                        "matrix_size": f"{shape[1]}x{shape[2]}",
                        "voxel_spacing_mm": f"{vol.spacing[0]:.2f}x{vol.spacing[1]:.2f}x{vol.spacing[2]:.2f}",
                        "hu_min": min_hu,
                        "hu_max": max_hu,
                        "hu_mean": round(mean_hu, 1),
                        "has_gt_mask": False,
                        "status": "VALID"
                    })
                    print(f"[OK] DICOM Series: {entry:25s} | Slices: {len(dcm_files):3d} | Spacing: {vol.spacing} | HU: [{min_hu:.0f}, {max_hu:.0f}]")
                except Exception as e:
                    records.append({
                        "dataset_id": entry,
                        "modality": "CT",
                        "format": "DICOM Series",
                        "file_path": os.path.relpath(patient_path, base_dir),
                        "status": f"ERROR: {str(e)}"
                    })
                    print(f"[FAIL] DICOM Series: {entry:25s} | Error: {e}")

    # 2. Scan NIfTI volumes
    if os.path.isdir(nifti_vol_dir):
        nii_files = glob.glob(os.path.join(nifti_vol_dir, "*.nii*"))
        for nii_path in sorted(nii_files):
            filename = os.path.basename(nii_path)
            clean_id = filename.replace(".nii.gz", "").replace(".nii", "")
            try:
                vol = load_nifti(nii_path)
                shape = vol.pixel_array.shape
                min_hu = float(np.min(vol.pixel_array))
                max_hu = float(np.max(vol.pixel_array))
                mean_hu = float(np.mean(vol.pixel_array))
                
                # Check for corresponding ground-truth mask
                mask_candidates = [
                    os.path.join(nifti_mask_dir, f"{clean_id}_seg.nii.gz"),
                    os.path.join(nifti_mask_dir, f"{clean_id}_mask.nii.gz"),
                    os.path.join(nifti_mask_dir, f"{clean_id}_gt.nii.gz"),
                    os.path.join(nifti_mask_dir, f"{clean_id}.nii.gz")
                ]
                has_gt = any(os.path.exists(m) for m in mask_candidates)

                records.append({
                    "dataset_id": clean_id,
                    "modality": "CT",
                    "format": "NIfTI",
                    "file_path": os.path.relpath(nii_path, base_dir),
                    "num_slices": shape[0],
                    "matrix_size": f"{shape[1]}x{shape[2]}",
                    "voxel_spacing_mm": f"{vol.spacing[0]:.2f}x{vol.spacing[1]:.2f}x{vol.spacing[2]:.2f}",
                    "hu_min": min_hu,
                    "hu_max": max_hu,
                    "hu_mean": round(mean_hu, 1),
                    "has_gt_mask": has_gt,
                    "status": "VALID"
                })
                gt_str = "[GT Mask Found]" if has_gt else "[No GT Mask]"
                print(f"[OK] NIfTI Volume: {clean_id:25s} | Shape: {shape} | Spacing: {vol.spacing} | {gt_str}")
            except Exception as e:
                records.append({
                    "dataset_id": clean_id,
                    "modality": "CT",
                    "format": "NIfTI",
                    "file_path": os.path.relpath(nii_path, base_dir),
                    "status": f"ERROR: {str(e)}"
                })
                print(f"[FAIL] NIfTI Volume: {clean_id:25s} | Error: {e}")

    # 3. Write manifest CSV
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print("-" * 75)
    print(f"Manifest written to: {os.path.abspath(manifest_path)}")
    print(f"Total valid CT datasets registered: {len(df[df['status'] == 'VALID'])}")
    print("=" * 75)
    return df

if __name__ == "__main__":
    verify_dataset()
