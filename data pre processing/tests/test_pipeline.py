"""
Unit & Integration Test Suite for Medical Preprocessing & 3D Reconstruction Pipeline
"""

import os
import unittest
import numpy as np
import SimpleITK as sitk

from core.data_loader import MedicalVolume
from core.preprocessor import resample_volume, apply_hu_window, normalize_intensity, segment_by_hu_threshold
from core.reconstructor_3d import extract_organ_mesh, smooth_mesh, save_mesh
from core.evaluator import compute_dice_coefficient, compute_jaccard_index, compute_hausdorff_95, compute_psnr
from phantom_generator import generate_anatomical_ct_phantom


class TestMedicalPipeline(unittest.TestCase):

    def setUp(self):
        self.ct_vol, self.gt_vol, self.organ_map = generate_anatomical_ct_phantom(
            dimensions=(32, 64, 64),
            spacing=(1.0, 1.0, 2.0)
        )

    def test_resample_volume(self):
        target_spacing = (1.0, 1.0, 1.0)
        resampled = resample_volume(self.ct_vol, target_spacing=target_spacing)
        self.assertEqual(resampled.spacing, target_spacing)
        # Z should double because spacing went from 2.0 to 1.0
        self.assertEqual(resampled.pixel_array.shape[0], 64)

    def test_hu_window(self):
        windowed = apply_hu_window(self.ct_vol, "bone")
        preset_bone_min = 400.0 - 900.0  # -500
        preset_bone_max = 400.0 + 900.0  # 1300
        self.assertGreaterEqual(float(np.min(windowed.pixel_array)), preset_bone_min)
        self.assertLessEqual(float(np.max(windowed.pixel_array)), preset_bone_max)

    def test_intensity_normalization(self):
        norm_vol = normalize_intensity(self.ct_vol, method="min_max")
        self.assertAlmostEqual(float(np.min(norm_vol.pixel_array)), 0.0, places=4)
        self.assertAlmostEqual(float(np.max(norm_vol.pixel_array)), 1.0, places=4)

    def test_marching_cubes_mesh_extraction(self):
        # Extract bone mesh from ground truth mask
        bone_mask = (self.gt_vol.pixel_array == 1).astype(np.float32)
        vol_bone = MedicalVolume(
            sitk_image=self.gt_vol.to_sitk(),
            pixel_array=bone_mask,
            spacing=self.gt_vol.spacing,
            origin=self.gt_vol.origin,
            direction=self.gt_vol.direction
        )
        mesh = extract_organ_mesh(vol_bone, isovalue=0.5, organ_name="TestBone")
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)
        self.assertGreater(mesh.surface_area_mm2, 0.0)

        # Test smoothing
        smoothed = smooth_mesh(mesh, iterations=5)
        self.assertEqual(len(smoothed.vertices), len(mesh.vertices))

    def test_evaluation_metrics(self):
        mask_a = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_a[5:15, 5:15, 5:15] = 1

        mask_b = np.copy(mask_a)
        # Identical masks
        dice_perfect = compute_dice_coefficient(mask_a, mask_b)
        iou_perfect = compute_jaccard_index(mask_a, mask_b)
        self.assertAlmostEqual(dice_perfect, 1.0)
        self.assertAlmostEqual(iou_perfect, 1.0)

        # Shifted mask
        mask_c = np.zeros_like(mask_a)
        mask_c[6:16, 5:15, 5:15] = 1
        dice_shifted = compute_dice_coefficient(mask_a, mask_c)
        self.assertGreater(dice_shifted, 0.8)
        self.assertLess(dice_shifted, 1.0)

        hd95 = compute_hausdorff_95(mask_a, mask_c, spacing=(1.0, 1.0, 1.0))
        self.assertGreater(hd95, 0.0)
        self.assertLess(hd95, 2.0)


if __name__ == "__main__":
    unittest.main()
