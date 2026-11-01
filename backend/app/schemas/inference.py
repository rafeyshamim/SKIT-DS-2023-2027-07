"""Pydantic Data Transfer Objects (DTOs) for AI Model Inference.

Validates input parameters for triggering 3D CNN inference runs and
serializes output prediction results, volumetric metrics, and slice heatmaps.
"""

import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from app.models.nosql.inference_payload import LesionDetection


class InferenceTriggerRequest(BaseModel):
    """Payload to trigger 3D CNN deep learning inference on a preprocessed scan."""
    confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for pulmonary nodule reporting."
    )
    run_gradcam_saliency: bool = Field(
        True,
        description="Whether to compute 3D Grad-CAM attention/saliency maps."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "confidence_threshold": 0.45,
                "run_gradcam_saliency": True
            }
        }
    )


class InferenceTriggerResponse(BaseModel):
    """Immediate acknowledgement upon queuing or starting an inference run."""
    inference_run_id: int = Field(..., description="Unique database ID of the inference run.")
    scan_id: int = Field(..., description="ID of the CT scan processed.")
    status: str = Field(..., description="Execution status (e.g. RUNNING, QUEUED).")
    message: str = Field(..., description="Operational status message.")
    started_at: datetime.datetime = Field(..., description="Timestamp when execution began.")


class VolumetricMetrics(BaseModel):
    """Quantified volumetric measurements of thoracic anatomy and lesions."""
    total_lung_volume_cm3: float = Field(3420.0, description="Total segmented lung volume in cm³.")
    total_lesion_volume_mm3: float = Field(0.0, description="Aggregate volume of detected nodules in mm³.")
    lung_involvement_percentage: float = Field(0.0, description="Percentage of parenchymal volume affected.")


class InferenceResultResponse(BaseModel):
    """Complete prediction output, diagnostic classifications, and localization."""
    id: int = Field(..., description="Inference run primary key.")
    scan_id: int = Field(..., description="Associated CT Scan ID.")
    patient_id: int = Field(..., description="Associated Patient ID.")
    model_name: str = Field(..., description="Neural network architecture identifier.")
    model_version: str = Field(..., description="Model release version.")
    status: str = Field(..., description="Terminal run status (COMPLETED, FAILED).")
    
    primary_prediction: Optional[str] = Field(None, description="Primary diagnostic classification.")
    confidence_score: Optional[float] = Field(None, description="Primary classification confidence (0.0 to 1.0).")
    risk_level: Optional[str] = Field(None, description="Risk categorization (LOW, MODERATE, HIGH, CRITICAL).")
    processing_time_ms: Optional[float] = Field(None, description="Inference execution duration in milliseconds.")
    
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized softmax probability distribution across diagnostic classes."
    )
    lesions_detected_count: int = Field(0, description="Number of suspicious focal lesions detected.")
    lesions: List[LesionDetection] = Field(
        default_factory=list,
        description="List of detected 3D pulmonary lesions with spatial coordinates."
    )
    slice_abnormality_scores: List[float] = Field(
        default_factory=list,
        description="Per-slice axial abnormality scores for interactive DICOM scrubber."
    )
    volumetric_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Quantified lung and nodule volumetric metrics."
    )
    started_at: datetime.datetime = Field(..., description="Timestamp when execution started.")
    completed_at: Optional[datetime.datetime] = Field(None, description="Timestamp when execution completed.")

    model_config = ConfigDict(from_attributes=True)


class InferenceRunSummaryResponse(BaseModel):
    """Condensed summary of an inference run for list views."""
    id: int
    scan_id: int
    model_name: str
    model_version: str
    status: str
    primary_prediction: Optional[str] = None
    confidence_score: Optional[float] = None
    risk_level: Optional[str] = None
    processing_time_ms: Optional[float] = None
    started_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(from_attributes=True)


class HeatmapSliceResponse(BaseModel):
    """2D axial slice attention heatmap and intersecting lesion indicators."""
    scan_id: int = Field(..., description="ID of the CT scan.")
    inference_run_id: int = Field(..., description="Associated inference run ID.")
    slice_index: int = Field(..., description="0-indexed axial slice number.")
    total_slices: int = Field(..., description="Total axial slices in volumetric scan.")
    abnormality_score: float = Field(..., description="Model abnormality index for this specific slice.")
    lesions_on_slice: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Lesions whose 3D bounding box intersects this axial slice."
    )
    normalized_heatmap_grid: List[List[float]] = Field(
        ...,
        description="16x16 normalized Grad-CAM attention matrix for frontend rendering."
    )
