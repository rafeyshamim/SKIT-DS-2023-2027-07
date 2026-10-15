import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from app.models.nosql.inference_payload import LesionDetection


class InferenceTriggerRequest(BaseModel):
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold for lesion reporting")
    run_gradcam_saliency: bool = Field(True, description="Whether to compute 3D Grad-CAM saliency maps")


class InferenceTriggerResponse(BaseModel):
    inference_run_id: int
    scan_id: int
    status: str
    message: str
    started_at: datetime.datetime


class InferenceResultResponse(BaseModel):
    id: int
    scan_id: int
    patient_id: int
    model_name: str
    model_version: str
    status: str
    primary_prediction: Optional[str] = None
    confidence_score: Optional[float] = None
    risk_level: Optional[str] = None
    processing_time_ms: Optional[float] = None
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    lesions_detected_count: int = 0
    lesions: List[LesionDetection] = Field(default_factory=list)
    slice_abnormality_scores: List[float] = Field(default_factory=list)
    volumetric_metrics: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(from_attributes=True)


class HeatmapSliceResponse(BaseModel):
    scan_id: int
    inference_run_id: int
    slice_index: int
    total_slices: int
    abnormality_score: float
    lesions_on_slice: List[Dict[str, Any]]
    normalized_heatmap_grid: List[List[float]]
