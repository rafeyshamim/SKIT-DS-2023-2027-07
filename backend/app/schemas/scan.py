import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class ScanBase(BaseModel):
    modality: str = Field("CT", description="Imaging modality", json_schema_extra={"example": "CT"})
    anatomical_region: str = Field("Chest / Thorax", description="Body region imaged", json_schema_extra={"example": "Chest / Thorax"})


class ScanUploadResponse(BaseModel):
    scan_id: int
    scan_uid: str
    patient_id: int
    modality: str
    anatomical_region: str
    original_filename: str
    file_size_bytes: int
    status: str
    message: str
    created_at: datetime.datetime


class ScanResponse(ScanBase):
    id: int
    scan_uid: str
    patient_id: int
    original_filename: str
    file_path: str
    processed_volume_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    slice_count: int
    slice_thickness_mm: Optional[float] = None
    pixel_spacing_xy: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class ScanDetailResponse(ScanResponse):
    patient_mrn: Optional[str] = None
    patient_name: Optional[str] = None
    dicom_metadata: Optional[Dict[str, Any]] = None


class ScanListResponse(BaseModel):
    total: int
    items: List[ScanResponse]
