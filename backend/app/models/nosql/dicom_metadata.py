from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DicomMetadataDocument(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    scan_id: int
    scan_uid: str
    series_instance_uid: Optional[str] = None
    study_instance_uid: Optional[str] = None
    sop_instance_uids: List[str] = Field(default_factory=list)
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    scanner_manufacturer: Optional[str] = "Siemens Healthineers"
    scanner_model: Optional[str] = "SOMATOM Definition Flash"
    kvp: Optional[float] = 120.0
    x_ray_tube_current_ma: Optional[float] = 150.0
    slice_thickness_mm: Optional[float] = 1.25
    pixel_spacing: List[float] = Field(default_factory=lambda: [0.703125, 0.703125])
    rows: int = 512
    columns: int = 512
    rescale_intercept: float = -1024.0
    rescale_slope: float = 1.0
    window_center: float = -600.0
    window_width: float = 1500.0
    patient_position: str = "HFS"
    slice_positions: List[float] = Field(default_factory=list)
    raw_header_tags: Dict[str, Any] = Field(default_factory=dict)
