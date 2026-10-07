"""
Synthetic Medical CT Phantom & Ground Truth Generator
======================================================
Generates realistic 3D CT abdominal/torso phantoms with ground truth multi-organ masks
to test and validate preprocessing, 3D reconstruction, and evaluation pipelines without
requiring multi-gigabyte clinical downloads.
"""

import os
from typing import Tuple, Dict
import numpy as np
import SimpleITK as sitk
from core.data_loader import MedicalVolume


def generate_anatomical_ct_phantom(
    dimensions: Tuple[int, int, int] = (64, 128, 128),  # (Z, Y, X)
    spacing: Tuple[float, float, float] = (0.8, 0.8, 2.5),  # (sx, sy, sz) in mm non-isotropic
    noise_sigma: float = 15.0
) -> Tuple[MedicalVolume, MedicalVolume, Dict[int, str]]:
    """
    Creates a simulated abdominal 3D CT volume and matching multi-organ ground truth mask.
    
    Hounsfield values:
    - Air / Background: -1000 HU
    - Fat: -100 HU
    - Soft tissue / Muscle: 40 HU
    - Liver: 60 HU
    - Kidney: 30 HU
    - Spine / Bone: 800 HU
    
    Returns:
        (ct_volume, ground_truth_mask_volume, organ_label_map)
    """
    nz, ny, nx = dimensions
    z_coords, y_coords, x_coords = np.indices((nz, ny, nx))

    # Initialize volume with air (-1000 HU)
    ct_volume = np.full(dimensions, -1000.0, dtype=np.float32)
    gt_mask = np.zeros(dimensions, dtype=np.uint8)

    # 1. Torso body contour (elliptical cylinder)
    cy, cx = ny // 2, nx // 2
    ry, rx = int(ny * 0.40), int(nx * 0.45)
    torso_mask = (((y_coords - cy) / ry) ** 2 + ((x_coords - cx) / rx) ** 2) <= 1.0
    ct_volume[torso_mask] = 40.0  # Muscle / Soft tissue

    # 2. Subcutaneous fat layer (outer rim of torso)
    fat_inner = (((y_coords - cy) / (ry * 0.92)) ** 2 + ((x_coords - cx) / (rx * 0.92)) ** 2) <= 1.0
    fat_mask = torso_mask & (~fat_inner)
    ct_volume[fat_mask] = -90.0

    # 3. Spine / Vertebral Bone (Posterior midline)
    spine_cy, spine_cx = int(cy + ny * 0.25), cx
    spine_r = int(min(ny, nx) * 0.08)
    spine_mask = (((y_coords - spine_cy) / spine_r) ** 2 + ((x_coords - spine_cx) / spine_r) ** 2) <= 1.0
    ct_volume[spine_mask] = 850.0
    gt_mask[spine_mask] = 1  # Label 1: Bone

    # 4. Liver (Right anterior/lateral abdominal quadrant)
    # Right side of body corresponds to left in radiology standard view
    liver_cy, liver_cx = int(cy - ny * 0.05), int(cx - nx * 0.18)
    liver_rz, liver_ry, liver_rx = int(nz * 0.35), int(ny * 0.22), int(nx * 0.22)
    z_mid = nz // 2
    liver_mask = (
        ((z_coords - z_mid) / liver_rz) ** 2 +
        ((y_coords - liver_cy) / liver_ry) ** 2 +
        ((x_coords - liver_cx) / liver_rx) ** 2
    ) <= 1.0
    ct_volume[liver_mask] = 65.0
    gt_mask[liver_mask] = 2  # Label 2: Liver

    # 5. Left Kidney (Posterolateral left)
    kidney_cy, kidney_cx = int(cy + ny * 0.15), int(cx + nx * 0.22)
    kidney_rz, kidney_ry, kidney_rx = int(nz * 0.20), int(ny * 0.10), int(nx * 0.10)
    kidney_mask = (
        ((z_coords - z_mid) / kidney_rz) ** 2 +
        ((y_coords - kidney_cy) / kidney_ry) ** 2 +
        ((x_coords - kidney_cx) / kidney_rx) ** 2
    ) <= 1.0
    ct_volume[kidney_mask] = 35.0
    gt_mask[kidney_mask] = 3  # Label 3: Kidney

    # Add realistic CT Poisson/Gaussian quantum mottle noise
    noise = np.random.normal(0, noise_sigma, dimensions).astype(np.float32)
    # Don't add noise to deep air outside FOV
    ct_volume = ct_volume + noise * (ct_volume > -900).astype(np.float32)

    # Convert to SimpleITK Images with non-isotropic physical spacing
    # Spacing in SimpleITK: (sx, sy, sz)
    sitk_ct = sitk.GetImageFromArray(ct_volume)
    sitk_ct.SetSpacing(spacing)
    sitk_ct.SetOrigin((0.0, 0.0, 0.0))

    sitk_gt = sitk.GetImageFromArray(gt_mask)
    sitk_gt.SetSpacing(spacing)
    sitk_gt.SetOrigin((0.0, 0.0, 0.0))

    organ_map = {
        1: "Vertebral Bone",
        2: "Liver",
        3: "Kidney"
    }

    vol_ct = MedicalVolume.from_sitk(sitk_ct, metadata={"modality": "Simulated_CT"})
    vol_gt = MedicalVolume.from_sitk(sitk_gt, metadata={"modality": "Multi_Organ_Mask"})

    return vol_ct, vol_gt, organ_map


def save_phantom_dataset(output_dir: str) -> Tuple[str, str, Dict[int, str]]:
    """
    Generates and saves the phantom CT volume and ground truth segmentation mask to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    vol_ct, vol_gt, organ_map = generate_anatomical_ct_phantom()
    
    ct_path = os.path.join(output_dir, "synthetic_abdominal_ct.nii.gz")
    gt_path = os.path.join(output_dir, "synthetic_multiorgan_gt.nii.gz")

    vol_ct.save_nifti(ct_path)
    vol_gt.save_nifti(gt_path)

    return ct_path, gt_path, organ_map
