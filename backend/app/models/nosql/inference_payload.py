"""Pydantic NoSQL document models for rich 3D volumetric inference payloads."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class BoundingBox3D(BaseModel):
    """3D bounding box bounds in voxel coordinate space."""
    z_min: int = Field(..., description="Minimum axial slice index")
    z_max: int = Field(..., description="Maximum axial slice index")
    y_min: int = Field(..., description="Minimum coronal height index")
    y_max: int = Field(..., description="Maximum coronal height index")
    x_min: int = Field(..., description="Minimum sagittal width index")
    x_max: int = Field(..., description="Maximum sagittal width index")


class Centroid(BaseModel):
    """3D coordinate centroid in voxel or physical millimeter space."""
    z: float = Field(..., description="Z-axis coordinate (depth / slice)")
    y: float = Field(..., description="Y-axis coordinate (height / coronal)")
    x: float = Field(..., description="X-axis coordinate (width / sagittal)")


class LesionDetection(BaseModel):
    """Detected focal pulmonary nodule or lesion with spatial 3D metrics."""
    lesion_id: str
    nodule_type: str = "Solid Pulmonary Nodule"
    centroid_voxel: Centroid
    centroid_mm: Centroid
    bounding_box_3d: BoundingBox3D
    volume_mm3: float
    malignancy_score: float
    calcification_pattern: Optional[str] = "Non-calcified"
    spiculation_score: Optional[float] = 0.85
    lobulation_score: Optional[float] = 0.72


class InferencePayloadDocument(BaseModel):
    """Full MongoDB document payload for an inference run."""
    id: Optional[str] = Field(None, alias="_id")
    inference_run_id: int
    scan_id: int
    model_name: str = "MedNet-3D-CNN"
    model_version: str = "v1.2.0"
    
    # Class probabilities
    class_probabilities: Dict[str, float] = Field(
        default_factory=lambda: {
            "Normal / No Finding": 0.04,
            "Benign Nodule": 0.11,
            "Malignant Suspicion": 0.85
        }
    )
    
    # Detected 3D lesions with spatial coordinates
    lesion_detections: List[LesionDetection] = Field(default_factory=list)
    
    # Per-slice abnormality heatmap scores (for axial slice scrolling)
    slice_abnormality_scores: List[float] = Field(default_factory=list)
    
    # Volumetric quantification
    total_lung_volume_cm3: float = 3450.0
    total_lesion_volume_mm3: float = 840.5
    lung_involvement_percentage: float = 0.024
    
    # 3D CAM / Attention peaks
    saliency_peak_voxels: List[List[int]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)
