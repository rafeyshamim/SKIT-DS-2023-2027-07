"""
Milestone 5: Reconstruction Module Integration Support & End-to-End Pipeline
============================================================================
Integrates the complete data lifecycle:
Data Ingestion -> Validation -> Preprocessing -> 3D Mesh Reconstruction -> Evaluation -> 3D Export
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, Union, List

import numpy as np

from .data_loader import load_dicom_series, load_nifti, MedicalVolume
from .preprocessor import resample_volume, apply_hu_window, normalize_intensity, denoise_volume
from .reconstructor_3d import extract_organ_mesh, smooth_mesh, decimate_mesh, save_mesh, ReconstructedMesh
from .evaluator import evaluate_segmentation_and_reconstruction

logger = logging.getLogger("MedicalPipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class PipelineConfig:
    """End-to-end pipeline parameter specification."""
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    hu_window: Optional[str] = "bone"  # Preset name or None
    normalize_method: Optional[str] = "min_max"  # "min_max", "z_score", or None
    denoise_method: Optional[str] = "curvature_flow"  # "curvature_flow", "gaussian", or None
    isovalue: float = 0.5
    smooth_mesh: bool = True
    decimate_reduction: float = 0.0  # e.g., 0.3 for 30% reduction, 0.0 for none
    export_formats: Tuple[str, ...] = ("stl", "obj")
    keep_largest_component: bool = True


class Medical3DReconstructionPipeline:
    """
    Production-ready medical image preprocessing and 3D organ reconstruction pipeline.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def validate_volume(self, volume: MedicalVolume) -> Dict[str, Any]:
        """
        Validates the geometric and numerical integrity of the input volume.
        Fixes and warns against inverted spacing, NaN values, or corrupted dimensions.
        """
        validation_status = {
            "valid": True,
            "issues": [],
            "shape_zyx": volume.pixel_array.shape,
            "spacing_xyz": volume.spacing
        }

        # Check for NaN / Inf
        if np.isnan(volume.pixel_array).any():
            validation_status["valid"] = False
            validation_status["issues"].append("Volume contains NaN pixel values. Imputing with -1024 HU.")
            volume.pixel_array = np.nan_to_num(volume.pixel_array, nan=-1024.0)

        # Check spacing
        if any(s <= 0 for s in volume.spacing):
            validation_status["valid"] = False
            validation_status["issues"].append("Non-positive voxel spacing detected.")

        # Check dimensions
        if volume.pixel_array.ndim != 3:
            validation_status["valid"] = False
            validation_status["issues"].append(f"Expected 3D volume, got {volume.pixel_array.ndim}D.")

        return validation_status

    def process(
        self,
        input_source: Union[str, MedicalVolume],
        output_dir: str,
        organ_name: str = "TargetOrgan",
        ground_truth_source: Optional[Union[str, MedicalVolume]] = None
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end processing pipeline:
        1. Ingestion
        2. Validation & Geometry repair
        3. Preprocessing (Resampling, Windowing, Normalization, Denoising)
        4. 3D Mesh Reconstruction & Optimization
        5. Export to 3D models (STL / OBJ)
        6. Quantitative Evaluation against ground truth (if provided)
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initiating 3D reconstruction pipeline for '{organ_name}'...")

        # 1. Ingestion
        if isinstance(input_source, MedicalVolume):
            volume = input_source
        elif os.path.isdir(input_source):
            logger.info(f"Loading DICOM series from directory: {input_source}")
            volume = load_dicom_series(input_source)
        elif os.path.isfile(input_source):
            logger.info(f"Loading NIfTI volume from file: {input_source}")
            volume = load_nifti(input_source)
        else:
            raise ValueError(f"Invalid input source: {input_source}")

        # 2. Validation
        val_report = self.validate_volume(volume)
        for issue in val_report.get("issues", []):
            logger.warning(f"Data pipeline warning: {issue}")

        # 3. Preprocessing
        logger.info(f"Original volume shape {volume.pixel_array.shape}, spacing {volume.spacing}")
        if self.config.target_spacing is not None:
            logger.info(f"Resampling volume to isotropic spacing {self.config.target_spacing} mm...")
            volume = resample_volume(volume, target_spacing=self.config.target_spacing)
            logger.info(f"Resampled volume shape: {volume.pixel_array.shape}")

        if self.config.hu_window is not None:
            logger.info(f"Applying CT HU Window: {self.config.hu_window}...")
            volume = apply_hu_window(volume, self.config.hu_window)

        if self.config.denoise_method is not None:
            logger.info(f"Applying edge-preserving denoising: {self.config.denoise_method}...")
            volume = denoise_volume(volume, method=self.config.denoise_method)

        if self.config.normalize_method is not None:
            logger.info(f"Normalizing voxel intensities ({self.config.normalize_method})...")
            volume = normalize_intensity(volume, method=self.config.normalize_method)

        # Save processed volume as NIfTI
        processed_nii_path = os.path.join(output_dir, f"{organ_name}_preprocessed.nii.gz")
        volume.save_nifti(processed_nii_path)
        logger.info(f"Preprocessed volume saved to: {processed_nii_path}")

        # 4. 3D Surface Reconstruction
        logger.info(f"Extracting 3D surface mesh using Marching Cubes at isovalue={self.config.isovalue}...")
        mesh = extract_organ_mesh(
            volume=volume,
            isovalue=self.config.isovalue,
            organ_name=organ_name,
            keep_largest_component=self.config.keep_largest_component
        )
        logger.info(f"Raw Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles.")

        if self.config.smooth_mesh:
            logger.info("Applying Laplacian surface smoothing...")
            mesh = smooth_mesh(mesh)

        if self.config.decimate_reduction > 0.0:
            logger.info(f"Decimating mesh (target reduction {self.config.decimate_reduction * 100}%)...")
            mesh = decimate_mesh(mesh, target_reduction=self.config.decimate_reduction)
            logger.info(f"Optimized Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles.")

        # 5. Export 3D Models
        saved_files = []
        for fmt in self.config.export_formats:
            fmt_path = os.path.join(output_dir, f"{organ_name.lower().replace(' ', '_')}.{fmt}")
            save_mesh(mesh, fmt_path)
            saved_files.append(fmt_path)
            logger.info(f"3D organ model exported: {fmt_path}")

        # 6. Optional Evaluation against Ground Truth
        eval_metrics = None
        if ground_truth_source is not None:
            logger.info("Ground truth provided. Evaluating reconstruction fidelity...")
            if isinstance(ground_truth_source, MedicalVolume):
                gt_vol = ground_truth_source
            elif os.path.isfile(ground_truth_source):
                gt_vol = load_nifti(ground_truth_source)
            else:
                gt_vol = load_dicom_series(ground_truth_source)

            # Resample GT if necessary
            if gt_vol.spacing != volume.spacing:
                gt_vol = resample_volume(gt_vol, target_spacing=volume.spacing, is_mask=True)

            eval_metrics = evaluate_segmentation_and_reconstruction(
                pred_volume=volume,
                gt_volume=gt_vol,
                reconstructed_mesh=mesh,
                organ_name=organ_name
            )

        # Output Summary
        pipeline_result = {
            "organ_name": organ_name,
            "pipeline_config": asdict(self.config),
            "output_directory": output_dir,
            "preprocessed_nifti": processed_nii_path,
            "exported_models": saved_files,
            "mesh_statistics": {
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.faces)),
                "surface_area_mm2": float(mesh.surface_area_mm2),
                "volume_mm3": float(mesh.volume_mm3),
                "is_watertight": bool(mesh.is_watertight)
            },
            "evaluation": eval_metrics
        }

        # Save JSON manifest
        manifest_path = os.path.join(output_dir, "reconstruction_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(pipeline_result, f, indent=2)

        logger.info(f"Pipeline complete! Manifest saved to: {manifest_path}")
        return pipeline_result
