"""
Milestone 1: Data Format & Medical Image Reconstruction Technique Study
========================================================================

This module provides comprehensive educational and programmatic tools to study:
1. Medical Image Formats: DICOM (Digital Imaging and Communications in Medicine) vs NIfTI (Neuroimaging Informatics Technology Initiative) vs HDF5.
2. Coordinate Systems & Affine Mathematics: LPS (DICOM standard) vs RAS (NIfTI standard).
3. Image Reconstruction Techniques: Analytical (Filtered Back Projection), Iterative (ART, SIRT, MBIR), Surface/Mesh (Marching Cubes), and Modern Deep Learning (Implicit Neural Representations / NeRFs).
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import SimpleITK as sitk
try:
    import pydicom
except ImportError:
    pydicom = None

STUDY_DOCUMENTATION = """
===================================================================================================
                                MEDICAL IMAGE FORMATS COMPARATIVE STUDY
===================================================================================================

1. DICOM (Digital Imaging and Communications in Medicine):
   - Scope: Universal clinical standard used across hospitals, PACS (Picture Archiving and Communication System), and scanners (Siemens, GE, Philips).
   - Structure: Typically stored as 1 file per 2D slice (e.g., .dcm). An entire 3D volume comprises a series of hundreds of individual files.
   - Metadata: Extremely rich, containing Patient ID, Acquisition Parameters, Slice Thickness, Pixel Spacing (x, y), Image Position (Patient), Image Orientation (Patient), Rescale Slope, and Rescale Intercept.
   - Coordinate System: LPS (Left-Posterior-Superior).
     * +X: towards patient's Left
     * +Y: towards patient's Posterior (back)
     * +Z: towards patient's Superior (head)
   - Challenge: Slices must be sorted physically by `ImagePositionPatient[2]` (or projection along slice normal), not alphabetically by filename!

2. NIfTI (Neuroimaging Informatics Technology Initiative):
   - Scope: Standard scientific and research format for 3D/4D volumes (MRI, fMRI, CT, PET).
   - Structure: Single monolithic file (.nii or compressed .nii.gz) containing the entire 3D or 4D dataset.
   - Metadata: 348-byte binary header containing dimensions (dim), voxel dimensions (pixdim in mm), datatype, and rigid/affine spatial transformation matrices (qform and sform).
   - Coordinate System: RAS (Right-Anterior-Superior).
     * +X: towards patient's Right
     * +Y: towards patient's Anterior (front)
     * +Z: towards patient's Superior (head)
   - Conversion: LPS <-> RAS requires inverting X and Y signs: diag(-1, -1, 1, 1).

3. HDF5 (.h5 / .hdf5):
   - Hierarchical data format widely used for storing multi-modal arrays, slice-level machine learning datasets (like BraTS 2D preprocessed slices), and feature embeddings.

===================================================================================================
                         MEDICAL IMAGE RECONSTRUCTION TECHNIQUES COMPARISON
===================================================================================================

A. CT Raw Projection to Voxel Volume (Analytical & Iterative Reconstruction):
   1. Filtered Back Projection (FBP):
      - Analytical Radon transform inversion via Fourier slice theorem (Ramp / Shepp-Logan filter followed by backprojection).
      - Fast (O(N^3)), but sensitive to noise and photon starvation artifacts at low radiation doses.
   2. Iterative Reconstruction (IR / MBIR):
      - Minimizes objective function: argmin_x ||Ax - y||^2_W + R(x), where A is the system matrix, y is detector sinogram, R(x) is regularizer (Total Variation).
      - Significantly reduces radiation dose (up to 70-80%) while suppressing quantum mottle noise.

B. Voxel Volume to 3D Organ Surface Mesh (Geometric Reconstruction):
   1. Marching Cubes (Lorensen & Cline, 1987):
      - Operates on a regular 3D scalar grid (CT Hounsfield units or segmentation probabilities).
      - Divides volume into 8-voxel cubes. An isovalue threshold creates a 256-case lookup table (14 unique symmetries) determining triangular topology intersecting each cube.
      - Linearly interpolates vertex positions along cube edges for smooth anatomical boundaries.
   2. Flying Edges Algorithm:
      - Multi-threaded scanline-based optimization of Marching Cubes that bypasses redundant cube lookups, delivering 5x-10x speedup.

C. Surface Post-Processing:
   1. Laplacian Smoothing:
      - Updates each vertex v_i = v_i + lambda * sum(v_j - v_i)/N to remove stair-stepping artifacts caused by non-isotropic slice thickness.
   2. Quadric Error Metric Decimation:
      - Reduces triangle count while preserving sharp anatomical boundaries and curvature for real-time 3D rendering and 3D printing.
===================================================================================================
"""


def print_study_report() -> None:
    """Prints the comprehensive format and reconstruction technique study."""
    print(STUDY_DOCUMENTATION)


def inspect_nifti_properties(file_path: str) -> Dict[str, Any]:
    """
    Inspects and parses critical spatial properties and affine matrices from a NIfTI file.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    reader.ReadImageInformation()
    
    spacing = reader.GetSpacing()
    origin = reader.GetOrigin()
    direction = reader.GetDirection()
    size = reader.GetSize()
    pixel_type = reader.GetPixelIDTypeAsString()
    
    # SimpleITK uses LPS coordinate frame by default
    affine_matrix = np.eye(4)
    dir_3x3 = np.array(direction).reshape((3, 3))
    affine_matrix[:3, :3] = dir_3x3 * np.array(spacing)
    affine_matrix[:3, 3] = np.array(origin)
    
    return {
        "format": "NIfTI",
        "file_path": file_path,
        "size_voxels": size,
        "spacing_mm": spacing,
        "origin_mm": origin,
        "direction_cosine_matrix": dir_3x3.tolist(),
        "pixel_type": pixel_type,
        "affine_lps": affine_matrix.tolist(),
        "is_isotropic": abs(spacing[0] - spacing[1]) < 1e-4 and abs(spacing[1] - spacing[2]) < 1e-4
    }


def inspect_dicom_series_properties(dicom_dir: str) -> Dict[str, Any]:
    """
    Inspects a DICOM series directory and extracts acquisition geometry.
    """
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    if not series_IDs:
        raise ValueError(f"No DICOM series identified in directory: {dicom_dir}")
        
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
    
    # Read first slice header using SimpleITK
    reader = sitk.ImageFileReader()
    reader.SetFileName(series_file_names[0])
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    
    slice_count = len(series_file_names)
    spacing = reader.GetSpacing()
    size_2d = reader.GetSize()
    
    # Extract tags if pydicom is available
    meta_tags = {}
    if pydicom:
        try:
            dcm = pydicom.dcmread(series_file_names[0], stop_before_pixels=True)
            meta_tags = {
                "Modality": getattr(dcm, "Modality", "CT"),
                "Manufacturer": getattr(dcm, "Manufacturer", "Unknown"),
                "PatientPosition": getattr(dcm, "PatientPosition", "Unknown"),
                "RescaleIntercept": float(getattr(dcm, "RescaleIntercept", 0.0)),
                "RescaleSlope": float(getattr(dcm, "RescaleSlope", 1.0)),
                "SliceThickness": float(getattr(dcm, "SliceThickness", spacing[2] if len(spacing) > 2 else 1.0)),
                "PixelSpacing": [float(x) for x in getattr(dcm, "PixelSpacing", [spacing[0], spacing[1]])]
            }
        except Exception as e:
            meta_tags["error"] = str(e)
            
    return {
        "format": "DICOM Series",
        "series_id": series_IDs[0],
        "slice_count": slice_count,
        "in_plane_dimensions": size_2d,
        "inferred_volume_size": (size_2d[0], size_2d[1], slice_count),
        "spacing_x_y": (spacing[0], spacing[1]),
        "dicom_metadata": meta_tags
    }
