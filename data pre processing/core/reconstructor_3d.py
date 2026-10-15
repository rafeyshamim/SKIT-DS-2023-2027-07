"""
Milestone 3: 3D Organ Reconstruction Pipeline Development
=========================================================
Implements:
1. 3D surface mesh reconstruction via Marching Cubes from preprocessed CT volumes / masks
2. Multi-organ segmentation extraction with topological island filtering
3. Mesh smoothing (Laplacian relaxation) and decimation (poly-reduction)
4. Physical coordinate mapping (Voxel grid -> Millimeter Patient Space)
5. Export to standard 3D formats: STL, OBJ, PLY
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy import ndimage
import trimesh

from .data_loader import MedicalVolume
from .config import DEFAULT_SMOOTHING_ITERATIONS, DEFAULT_SMOOTHING_RELAXATION, DEFAULT_DECIMATION_TARGET_REDUCTION


@dataclass
class ReconstructedMesh:
    """
    Encapsulates a 3D triangular surface mesh representing an anatomical organ.
    """
    organ_name: str
    vertices: np.ndarray  # Shape (V, 3) in physical millimeter space
    faces: np.ndarray     # Shape (F, 3) triangle indices
    normals: np.ndarray   # Shape (V, 3) normal vectors
    surface_area_mm2: float
    volume_mm3: float
    is_watertight: bool

    def to_trimesh(self) -> trimesh.Trimesh:
        """Converts into a Trimesh object for advanced geometric processing."""
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            process=False
        )
        return mesh

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh, organ_name: str = "Organ") -> "ReconstructedMesh":
        # Check if watertight
        is_closed = bool(mesh.is_watertight)
        # Compute volume if watertight or convex hull volume
        vol = float(mesh.volume) if is_closed else float(mesh.convex_hull.volume)
        area = float(mesh.area)

        return cls(
            organ_name=organ_name,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
            surface_area_mm2=area,
            volume_mm3=vol,
            is_watertight=is_closed
        )


def extract_organ_mesh(
    volume: MedicalVolume,
    isovalue: float = 0.5,
    organ_name: str = "Reconstructed_Organ",
    keep_largest_component: bool = True,
    step_size: int = 1
) -> ReconstructedMesh:
    """
    Extracts a 3D surface mesh from a 3D volume or segmentation mask using Marching Cubes.
    
    Args:
        volume: Input MedicalVolume
        isovalue: Surface threshold. For binary masks, 0.5; for raw CT HU, e.g. 300 for bone.
        organ_name: Descriptive name of the organ.
        keep_largest_component: Removes floating spurious noise islands.
        step_size: Step size in voxels for Marching Cubes (default: 1 for highest fidelity).
        
    Returns:
        ReconstructedMesh with physical coordinates in millimeters.
    """
    arr = volume.pixel_array  # (Z, Y, X)
    
    # Optional morphological cleanup: keep largest connected component
    if keep_largest_component:
        binary_mask = arr >= isovalue
        if np.any(binary_mask):
            labeled, num_features = ndimage.label(binary_mask)
            if num_features > 1:
                component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
                largest_label = np.argmax(component_sizes) + 1
                arr = (labeled == largest_label).astype(np.float32)

    # SimpleITK spacing is (sx, sy, sz) in (X, Y, Z) order
    # skimage.measure.marching_cubes expects spacing in the same axis order as array: (sz, sy, sx)
    spacing_zyx = (volume.spacing[2], volume.spacing[1], volume.spacing[0])

    # Execute Marching Cubes
    # verts returned are in (z, y, x) physical units
    verts, faces, normals, _ = measure.marching_cubes(
        volume=arr,
        level=isovalue,
        spacing=spacing_zyx,
        step_size=step_size
    )

    # Convert coordinates from (z, y, x) to standard 3D cartesian (x, y, z)
    verts_xyz = np.zeros_like(verts)
    verts_xyz[:, 0] = verts[:, 2] + volume.origin[0]  # X
    verts_xyz[:, 1] = verts[:, 1] + volume.origin[1]  # Y
    verts_xyz[:, 2] = verts[:, 0] + volume.origin[2]  # Z

    # Also rearrange normals to (x, y, z)
    normals_xyz = np.zeros_like(normals)
    normals_xyz[:, 0] = normals[:, 2]
    normals_xyz[:, 1] = normals[:, 1]
    normals_xyz[:, 2] = normals[:, 0]

    raw_mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, vertex_normals=normals_xyz, process=True)
    return ReconstructedMesh.from_trimesh(raw_mesh, organ_name=organ_name)


def smooth_mesh(
    mesh: ReconstructedMesh,
    iterations: int = DEFAULT_SMOOTHING_ITERATIONS,
    relaxation: float = DEFAULT_SMOOTHING_RELAXATION
) -> ReconstructedMesh:
    """
    Applies Laplacian surface smoothing to eliminate voxel discretization stair-stepping.
    """
    tm = mesh.to_trimesh()
    try:
        # Trimesh laplacian filter
        smoothed_tm = trimesh.smoothing.filter_laplacian(tm, lamb=relaxation, iterations=iterations)
        return ReconstructedMesh.from_trimesh(smoothed_tm, organ_name=mesh.organ_name)
    except Exception:
        # Fallback if trimesh smoothing fails on open non-manifold edges
        return mesh


def decimate_mesh(
    mesh: ReconstructedMesh,
    target_reduction: float = DEFAULT_DECIMATION_TARGET_REDUCTION
) -> ReconstructedMesh:
    """
    Reduces polygon count (decimates triangles) while preserving surface topology.
    
    Args:
        mesh: Input ReconstructedMesh
        target_reduction: Fraction of faces to remove (e.g. 0.5 = 50% fewer faces)
    """
    tm = mesh.to_trimesh()
    target_faces = int(len(tm.faces) * (1.0 - target_reduction))
    
    try:
        decimated_tm = tm.simplify_quadric_decimation(target_faces)
        return ReconstructedMesh.from_trimesh(decimated_tm, organ_name=mesh.organ_name)
    except Exception:
        # If openctm/quadric decimation backend is not compiled, return original
        return mesh


def save_mesh(mesh: ReconstructedMesh, output_filepath: str) -> str:
    """
    Saves the reconstructed mesh to STL, OBJ, or PLY format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    tm = mesh.to_trimesh()
    tm.export(output_filepath)
    return output_filepath


def batch_reconstruct_organs(
    segmentation_volume: MedicalVolume,
    organ_label_map: Dict[int, str],
    output_dir: str,
    export_format: str = "stl",
    smooth: bool = True
) -> Dict[str, ReconstructedMesh]:
    """
    Reconstructs 3D meshes for all organs labeled in a multi-class segmentation mask.
    
    Args:
        segmentation_volume: Multi-label segmentation volume
        organ_label_map: Dict mapping integer label to organ name, e.g. {1: "Liver", 2: "Spleen"}
        output_dir: Directory where 3D meshes will be saved
        export_format: "stl", "obj", or "ply"
        smooth: Whether to apply Laplacian smoothing
    """
    results: Dict[str, ReconstructedMesh] = {}
    
    for label_id, organ_name in organ_label_map.items():
        # Create binary mask for this organ
        organ_mask_arr = (segmentation_volume.pixel_array == label_id).astype(np.float32)
        if not np.any(organ_mask_arr):
            continue

        temp_vol = MedicalVolume(
            sitk_image=sitk.GetImageFromArray(organ_mask_arr),
            pixel_array=organ_mask_arr,
            spacing=segmentation_volume.spacing,
            origin=segmentation_volume.origin,
            direction=segmentation_volume.direction
        )
        
        organ_mesh = extract_organ_mesh(
            volume=temp_vol,
            isovalue=0.5,
            organ_name=organ_name,
            keep_largest_component=True
        )

        if smooth:
            organ_mesh = smooth_mesh(organ_mesh)

        out_path = os.path.join(output_dir, f"{organ_name.lower().replace(' ', '_')}.{export_format}")
        save_mesh(organ_mesh, out_path)
        results[organ_name] = organ_mesh

    return results
