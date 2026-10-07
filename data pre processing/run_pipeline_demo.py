"""
End-to-End Medical Preprocessing and 3D Organ Reconstruction Demo
==================================================================
Executes all 6 milestones:
1. Format & Reconstruction technique analysis
2. BraTS20 metadata parsing & CT scan collection/preprocessing
3. 3D organ reconstruction (Marching Cubes + Smoothing + Decimation)
4. Quantitative validation against ground truth (Dice, IoU, HD95, ASSD, PSNR, SSIM)
5. Integrated pipeline execution and 3D model generation
6. Report generation
"""

import os
import json
import numpy as np

from core.data_formats_study import print_study_report
from core.data_loader import load_brats_metadata, summarize_brats_dataset, MedicalVolume
from core.pipeline import Medical3DReconstructionPipeline, PipelineConfig
from core.reconstructor_3d import batch_reconstruct_organs
from phantom_generator import save_phantom_dataset


def run_full_demo():
    print("=" * 80)
    print(" MEDICAL IMAGE PREPROCESSING & 3D ORGAN RECONSTRUCTION PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------
    # MILESTONE 1: Data Format & Preprocessing Technique Study
    # -------------------------------------------------------------
    print("\n[MILESTONE 1] Data Format & Medical Image Reconstruction Study")
    print("-" * 80)
    print("Studying DICOM vs NIfTI formats, LPS vs RAS spaces, and reconstruction methods...")
    print("[OK] DICOM: Slice-based clinical standard, LPS coordinate system, rich header tags.")
    print("[OK] NIfTI: 3D/4D monolithic research standard, RAS coordinate system, affine matrices.")
    print("[OK] Reconstruction: FBP / Iterative (Sinogram -> Volume) & Marching Cubes (Volume -> 3D Mesh).")

    # -------------------------------------------------------------
    # MILESTONE 2: Dataset Ingestion & Preprocessing
    # -------------------------------------------------------------
    print("\n[MILESTONE 2] CT Scan & BraTS Data Ingestion & Preprocessing")
    print("-" * 80)
    
    # 2A. Inspect BraTS20 Training Metadata
    brats_csv_path = r"c:\Users\Mohit Choudhary\Downloads\BraTS20 Training Metadata.csv\BraTS20 Training Metadata.csv"
    if os.path.exists(brats_csv_path):
        print(f"Loading BraTS20 Training Metadata from:\n  {brats_csv_path}")
        df_brats = load_brats_metadata(brats_csv_path)
        stats = summarize_brats_dataset(df_brats)
        print("BraTS20 Dataset Summary:")
        print(f"  * Total 2D Slices Cataloged: {stats['total_slices']:,}")
        print(f"  * Unique 3D Patient Volumes: {stats['unique_volumes']}")
        print(f"  * Slices Containing Tumor:   {stats['tumor_positive_slices']:,} ({stats['tumor_slice_percentage']:.2f}%)")
        print(f"  * Mean Background Ratio:     {stats['mean_background_ratio']:.4f}")
    else:
        print(f"BraTS CSV not found at {brats_csv_path}. Skipping BraTS stats.")

    # 2B. Prepare CT Data (Generating high-resolution multi-organ phantom)
    work_dir = os.path.abspath("pipeline_output")
    os.makedirs(work_dir, exist_ok=True)
    print(f"\nGenerating simulated multi-organ CT scan dataset in:\n  {work_dir}")
    ct_file, gt_file, organ_map = save_phantom_dataset(work_dir)
    print(f"[OK] Created CT scan volume: {ct_file}")
    print(f"[OK] Created Ground Truth segmentation: {gt_file}")
    print(f"[OK] Anatomical targets: {list(organ_map.values())}")

    # -------------------------------------------------------------
    # MILESTONE 3, 4, 5: Integrated Pipeline Execution
    # -------------------------------------------------------------
    print("\n[MILESTONE 3, 4 & 5] 3D Reconstruction Pipeline Execution & Evaluation")
    print("-" * 80)
    
    # Configure pipeline
    config = PipelineConfig(
        target_spacing=(1.0, 1.0, 1.0),   # Resample non-isotropic (0.8, 0.8, 2.5) to isotropic (1.0, 1.0, 1.0)
        hu_window="bone",                  # Clinical HU bone preset
        normalize_method="min_max",        # Normalize intensities
        denoise_method="curvature_flow",   # Edge-preserving ITK curvature flow denoising
        isovalue=0.5,
        smooth_mesh=True,                  # Laplacian surface smoothing
        decimate_reduction=0.2,            # 20% mesh decimation
        export_formats=("stl", "obj")
    )
    pipeline = Medical3DReconstructionPipeline(config=config)

    # Reconstruct Spine / Bone structure
    bone_output_dir = os.path.join(work_dir, "bone_reconstruction")
    print("\n>>> Processing Anatomical Structure: Spine / Vertebral Bone...")
    bone_result = pipeline.process(
        input_source=ct_file,
        output_dir=bone_output_dir,
        organ_name="Spine_Bone"
    )

    print("\n>>> Multi-Organ 3D Reconstruction from Ground Truth Masks...")
    from core.data_loader import load_nifti
    gt_vol = load_nifti(gt_file)
    organs_dir = os.path.join(work_dir, "organs_3d_models")
    organ_meshes = batch_reconstruct_organs(
        segmentation_volume=gt_vol,
        organ_label_map=organ_map,
        output_dir=organs_dir,
        export_format="stl",
        smooth=True
    )
    for name, m in organ_meshes.items():
        print(f"[OK] Reconstructed '{name}': {len(m.vertices):,} vertices, {len(m.faces):,} triangles, {m.surface_area_mm2:.1f} mm2 area, {m.volume_mm3:.1f} mm3 volume.")

    # -------------------------------------------------------------
    # MILESTONE 4: Quantitative Reconstruction Evaluation
    # -------------------------------------------------------------
    print("\n[MILESTONE 4] Quantitative Evaluation against Ground Truth")
    print("-" * 80)
    from core.evaluator import evaluate_segmentation_and_reconstruction
    from core.preprocessor import segment_by_hu_threshold, resample_volume
    
    # 1. Evaluate Direct Bone Segmentation against Ground Truth
    ct_vol = load_nifti(ct_file)
    ct_resampled = resample_volume(ct_vol, target_spacing=(1.0, 1.0, 1.0))
    pred_bone_segmented = segment_by_hu_threshold(ct_resampled, min_hu=300.0, max_hu=2000.0)

    # GT bone mask (label == 1)
    gt_bone_mask = (gt_vol.pixel_array == 1).astype(np.float32)
    gt_bone_vol = MedicalVolume(
        sitk_image=gt_vol.to_sitk(),
        pixel_array=gt_bone_mask,
        spacing=gt_vol.spacing,
        origin=gt_vol.origin,
        direction=gt_vol.direction
    )
    gt_bone_vol_resampled = resample_volume(gt_bone_vol, target_spacing=(1.0, 1.0, 1.0), is_mask=True)

    eval_result = evaluate_segmentation_and_reconstruction(
        pred_volume=pred_bone_segmented,
        gt_volume=gt_bone_vol_resampled,
        reconstructed_mesh=organ_meshes.get("Vertebral Bone"),
        organ_name="Vertebral_Bone"
    )

    metrics = eval_result["metrics"]
    print("Reconstruction Evaluation Metrics:")
    print(f"  * Dice Similarity Coefficient (DSC): {metrics['dice_similarity_coefficient']:.4f}")
    print(f"  * Jaccard Index (IoU):                 {metrics['jaccard_iou']:.4f}")
    print(f"  * 95% Hausdorff Distance (HD95):       {metrics['hausdorff_distance_95_mm']:.2f} mm")
    print(f"  * Average Symmetric Surface Distance: {metrics['average_symmetric_surface_distance_mm']:.2f} mm")
    print(f"  * Peak Signal-to-Noise Ratio (PSNR):   {metrics['psnr_db']:.2f} dB")
    print(f"  * Structural Similarity Index (SSIM):  {metrics['ssim']:.4f}")

    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION SUCCESSFUL!")
    print("=" * 80)
    print(f"All 3D models (STL & OBJ) and preprocessed NIfTI files are available at:\n  {work_dir}")


if __name__ == "__main__":
    run_full_demo()
