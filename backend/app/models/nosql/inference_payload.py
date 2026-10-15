from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BoundingBox3D(BaseModel):
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int


class Centroid(BaseModel):
    z: float
    y: float
    x: float


class LesionDetection(BaseModel):
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
