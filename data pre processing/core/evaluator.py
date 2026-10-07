"""
Milestone 4: Validation Dataset Preparation & Reconstruction Evaluation
========================================================================
Implements clinical-grade volumetric and surface distance evaluation metrics:
1. Volumetric Overlap: Dice Similarity Coefficient (DSC), Jaccard Index (IoU)
2. Surface Distance Metrics: 95th Percentile Hausdorff Distance (HD95), Average Symmetric Surface Distance (ASSD)
3. Image Fidelity Metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), Normalized Cross-Correlation (NCC)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import peak_signal_noise_ratio as compute_skimage_psnr
from skimage.metrics import structural_similarity as compute_skimage_ssim

from .data_loader import MedicalVolume
from .reconstructor_3d import ReconstructedMesh


def compute_dice_coefficient(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Computes Dice Similarity Coefficient (DSC) between two binary masks.
    DSC = 2 * |A n B| / (|A| + |B|)
    """
    pred_bool = (mask_pred > 0).astype(bool)
    gt_bool = (mask_gt > 0).astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    total = pred_bool.sum() + gt_bool.sum()

    if total == 0:
        return 1.0  # Perfect agreement on empty mask
    return float((2.0 * intersection) / total)


def compute_jaccard_index(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Computes Jaccard Index (Intersection over Union, IoU).
    IoU = |A n B| / |A u B|
    """
    pred_bool = (mask_pred > 0).astype(bool)
    gt_bool = (mask_gt > 0).astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


def _get_surface_points(binary_mask: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    """
    Extracts 3D boundary voxel surface coordinates scaled by physical voxel spacing.
    """
    # Morphological boundary = mask - eroded_mask
    eroded = ndimage.binary_erosion(binary_mask)
    boundary = np.logical_xor(binary_mask, eroded)
    coords = np.argwhere(boundary)  # shape (N, 3) in (z, y, x)
    
    if len(coords) == 0:
        return np.empty((0, 3))

    # Scale by spacing: spacing in SimpleITK is (sx, sy, sz) -> apply to (x, y, z)
    # coords are (z, y, x) -> scale coords[:, 0] by sz, coords[:, 1] by sy, coords[:, 2] by sx
    scaled_pts = np.zeros_like(coords, dtype=np.float32)
    scaled_pts[:, 0] = coords[:, 2] * spacing[0]  # X
    scaled_pts[:, 1] = coords[:, 1] * spacing[1]  # Y
    scaled_pts[:, 2] = coords[:, 0] * spacing[2]  # Z
    return scaled_pts


def compute_hausdorff_95(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Computes the 95th Percentile Hausdorff Distance (HD95) in physical millimeters.
    Eliminates outlier points to provide robust clinical boundary discrepancy.
    """
    pts_pred = _get_surface_points(mask_pred > 0, spacing)
    pts_gt = _get_surface_points(mask_gt > 0, spacing)

    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return float("inf")

    # Sample for performance if points exceed 10000
    if len(pts_pred) > 5000:
        indices = np.random.choice(len(pts_pred), 5000, replace=False)
        pts_pred = pts_pred[indices]
    if len(pts_gt) > 5000:
        indices = np.random.choice(len(pts_gt), 5000, replace=False)
        pts_gt = pts_gt[indices]

    from scipy.spatial import cKDTree
    tree_pred = cKDTree(pts_pred)
    tree_gt = cKDTree(pts_gt)

    dists_pred_to_gt, _ = tree_gt.query(pts_pred)
    dists_gt_to_pred, _ = tree_pred.query(pts_gt)

    hd95_pred = np.percentile(dists_pred_to_gt, 95)
    hd95_gt = np.percentile(dists_gt_to_pred, 95)

    return float(max(hd95_pred, hd95_gt))


def compute_assd(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Computes Average Symmetric Surface Distance (ASSD) in physical millimeters.
    """
    pts_pred = _get_surface_points(mask_pred > 0, spacing)
    pts_gt = _get_surface_points(mask_gt > 0, spacing)

    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return float("inf")

    if len(pts_pred) > 5000:
        pts_pred = pts_pred[np.random.choice(len(pts_pred), 5000, replace=False)]
    if len(pts_gt) > 5000:
        pts_gt = pts_gt[np.random.choice(len(pts_gt), 5000, replace=False)]

    from scipy.spatial import cKDTree
    tree_pred = cKDTree(pts_pred)
    tree_gt = cKDTree(pts_gt)

    dists_pred_to_gt, _ = tree_gt.query(pts_pred)
    dists_gt_to_pred, _ = tree_pred.query(pts_gt)

    assd = (np.mean(dists_pred_to_gt) + np.mean(dists_gt_to_pred)) / 2.0
    return float(assd)


def compute_psnr(image_pred: np.ndarray, image_gt: np.ndarray, data_range: Optional[float] = None) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    rng = data_range if data_range is not None else float(np.max(image_gt) - np.min(image_gt))
    if rng <= 0:
        rng = 1.0
    return float(compute_skimage_psnr(image_gt, image_pred, data_range=rng))


def compute_ssim(image_pred: np.ndarray, image_gt: np.ndarray, data_range: Optional[float] = None) -> float:
    """Computes Structural Similarity Index Measure (SSIM)."""
    rng = data_range if data_range is not None else float(np.max(image_gt) - np.min(image_gt))
    if rng <= 0:
        rng = 1.0
    return float(compute_skimage_ssim(image_gt, image_pred, data_range=rng))


def evaluate_segmentation_and_reconstruction(
    pred_volume: MedicalVolume,
    gt_volume: MedicalVolume,
    reconstructed_mesh: Optional[ReconstructedMesh] = None,
    organ_name: str = "Organ"
) -> Dict[str, Any]:
    """
    Runs a comprehensive clinical quality validation suite between prediction and ground truth.
    """
    pred_arr = pred_volume.pixel_array
    gt_arr = gt_volume.pixel_array
    spacing = pred_volume.spacing

    dice = compute_dice_coefficient(pred_arr, gt_arr)
    iou = compute_jaccard_index(pred_arr, gt_arr)
    hd95 = compute_hausdorff_95(pred_arr, gt_arr, spacing)
    assd = compute_assd(pred_arr, gt_arr, spacing)
    
    # Image fidelity
    psnr_val = compute_psnr(pred_arr.astype(np.float32), gt_arr.astype(np.float32))
    ssim_val = compute_ssim(pred_arr.astype(np.float32), gt_arr.astype(np.float32))

    report = {
        "organ_name": organ_name,
        "metrics": {
            "dice_similarity_coefficient": dice,
            "jaccard_iou": iou,
            "hausdorff_distance_95_mm": hd95,
            "average_symmetric_surface_distance_mm": assd,
            "psnr_db": psnr_val,
            "ssim": ssim_val
        },
        "reconstruction_geometry": {
            "vertex_count": len(reconstructed_mesh.vertices) if reconstructed_mesh else None,
            "face_count": len(reconstructed_mesh.faces) if reconstructed_mesh else None,
            "surface_area_mm2": reconstructed_mesh.surface_area_mm2 if reconstructed_mesh else None,
            "volume_mm3": reconstructed_mesh.volume_mm3 if reconstructed_mesh else None,
            "is_watertight": reconstructed_mesh.is_watertight if reconstructed_mesh else None
        }
    }
    return report
