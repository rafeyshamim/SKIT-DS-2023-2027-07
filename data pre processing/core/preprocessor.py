"""
Milestone 2: CT Scan & Medical Image Preprocessing Pipeline
===========================================================
Implements:
1. Resampling to isotropic voxel spacing (e.g. 1.0 x 1.0 x 1.0 mm) via SimpleITK
2. CT Hounsfield Unit (HU) windowing and clipping using clinical presets
3. Intensity normalization (Min-Max, Z-Score)
4. Edge-preserving noise filtration (Curvature Flow, Gaussian, Median)
"""

from typing import Tuple, Optional, Union
import numpy as np
import SimpleITK as sitk
from .data_loader import MedicalVolume
from .config import CTWindowPresets, WindowPreset, DEFAULT_ISOTROPIC_SPACING


def resample_volume(
    volume: MedicalVolume,
    target_spacing: Tuple[float, float, float] = DEFAULT_ISOTROPIC_SPACING,
    is_mask: bool = False
) -> MedicalVolume:
    """
    Resamples a 3D medical volume to target voxel spacing (in mm) using SimpleITK.
    
    Args:
        volume: Input MedicalVolume
        target_spacing: Desired voxel spacing (sx, sy, sz) in mm. Default (1.0, 1.0, 1.0)
        is_mask: True if resampling label masks (uses Nearest Neighbor);
                 False for continuous intensity scans (uses B-spline / Linear).
    
    Returns:
        New resampled MedicalVolume
    """
    orig_img = volume.to_sitk()
    orig_spacing = orig_img.GetSpacing()
    orig_size = orig_img.GetSize()

    # Calculate new grid size
    target_size = [
        int(round((orig_size[i] * orig_spacing[i]) / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(orig_img.GetDirection())
    resampler.SetOutputOrigin(orig_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0 if is_mask else -1024.0)  # Air in CT is -1000 to -1024 HU

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)

    resampled_img = resampler.Execute(orig_img)
    return MedicalVolume.from_sitk(resampled_img, metadata=dict(volume.metadata))


def apply_hu_window(
    volume: MedicalVolume,
    preset_or_name: Union[WindowPreset, str, Tuple[float, float]] = "soft_tissue"
) -> MedicalVolume:
    """
    Applies Hounsfield Unit (HU) windowing and clipping.
    
    Args:
        volume: Input MedicalVolume
        preset_or_name: Either a WindowPreset instance, a preset string name,
                        or a tuple of (min_hu, max_hu).
    """
    if isinstance(preset_or_name, WindowPreset):
        min_hu, max_hu = preset_or_name.window_min, preset_or_name.window_max
        preset_name = preset_or_name.name
    elif isinstance(preset_or_name, str):
        preset = CTWindowPresets.get(preset_or_name)
        min_hu, max_hu = preset.window_min, preset.window_max
        preset_name = preset.name
    elif isinstance(preset_or_name, (tuple, list)) and len(preset_or_name) == 2:
        min_hu, max_hu = float(preset_or_name[0]), float(preset_or_name[1])
        preset_name = f"Custom[{min_hu}:{max_hu}]"
    else:
        raise ValueError(f"Invalid window preset specification: {preset_or_name}")

    # Intensity windowing on numpy array
    clipped = np.clip(volume.pixel_array, min_hu, max_hu)
    
    # Update volume
    new_vol = MedicalVolume(
        sitk_image=sitk.GetImageFromArray(clipped),
        pixel_array=clipped,
        spacing=volume.spacing,
        origin=volume.origin,
        direction=volume.direction,
        metadata={**volume.metadata, "hu_window": preset_name, "hu_min": min_hu, "hu_max": max_hu}
    )
    new_vol.sitk_image.SetSpacing(volume.spacing)
    new_vol.sitk_image.SetOrigin(volume.origin)
    new_vol.sitk_image.SetDirection(volume.direction)
    return new_vol


def normalize_intensity(
    volume: MedicalVolume,
    method: str = "min_max",
    target_range: Tuple[float, float] = (0.0, 1.0)
) -> MedicalVolume:
    """
    Normalizes intensity values of the volume.
    
    Args:
        volume: Input MedicalVolume
        method: "min_max" or "z_score"
        target_range: Output range for min_max (default: [0.0, 1.0])
    """
    arr = volume.pixel_array.astype(np.float32)

    if method == "min_max":
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            normalized = (arr - min_val) / (max_val - min_val)
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        else:
            normalized = np.zeros_like(arr)
    elif method == "z_score":
        mean = np.mean(arr)
        std = np.std(arr)
        normalized = (arr - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    new_vol = MedicalVolume(
        sitk_image=sitk.GetImageFromArray(normalized),
        pixel_array=normalized,
        spacing=volume.spacing,
        origin=volume.origin,
        direction=volume.direction,
        metadata={**volume.metadata, "norm_method": method}
    )
    new_vol.sitk_image.SetSpacing(volume.spacing)
    new_vol.sitk_image.SetOrigin(volume.origin)
    new_vol.sitk_image.SetDirection(volume.direction)
    return new_vol


def denoise_volume(
    volume: MedicalVolume,
    method: str = "curvature_flow",
    **kwargs
) -> MedicalVolume:
    """
    Applies edge-preserving noise filtering to a CT volume.
    
    Methods:
        - "curvature_flow": SimpleITK Curvature Flow (preserves anatomical boundaries)
        - "gaussian": Recursive Gaussian smoothing
        - "median": 3D Median filter
    """
    sitk_img = volume.to_sitk()
    
    # Cast to float32 for filter execution
    float_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

    if method == "curvature_flow":
        cf = sitk.CurvatureFlowImageFilter()
        cf.SetTimeStep(kwargs.get("time_step", 0.125))
        cf.SetNumberOfIterations(kwargs.get("iterations", 5))
        denoised = cf.Execute(float_img)
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        denoised = sitk.SmoothingRecursiveGaussian(float_img, sigma)
    elif method == "median":
        radius = kwargs.get("radius", 1)
        denoised = sitk.Median(float_img, [radius, radius, radius])
    else:
        raise ValueError(f"Unknown denoising method: {method}")

    new_vol = MedicalVolume.from_sitk(denoised, metadata={**volume.metadata, "denoised": method})
    return new_vol


def segment_by_hu_threshold(
    volume: MedicalVolume,
    min_hu: float = 300.0,
    max_hu: float = 3000.0
) -> MedicalVolume:
    """
    Directly segments anatomical structures by Hounsfield Unit (HU) thresholds.
    Examples:
      - Bone: min_hu=300, max_hu=3000
      - Lung: min_hu=-900, max_hu=-400
      - Soft tissue: min_hu=20, max_hu=80
    """
    mask = ((volume.pixel_array >= min_hu) & (volume.pixel_array <= max_hu)).astype(np.float32)
    new_vol = MedicalVolume(
        sitk_image=sitk.GetImageFromArray(mask),
        pixel_array=mask,
        spacing=volume.spacing,
        origin=volume.origin,
        direction=volume.direction,
        metadata={**volume.metadata, "segmented_hu_range": (min_hu, max_hu)}
    )
    new_vol.sitk_image.SetSpacing(volume.spacing)
    new_vol.sitk_image.SetOrigin(volume.origin)
    new_vol.sitk_image.SetDirection(volume.direction)
    return new_vol

