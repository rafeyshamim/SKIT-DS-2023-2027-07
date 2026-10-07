import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


class ReportCreate(BaseModel):
    scan_id: int = Field(..., description="ID of the CT scan")
    inference_run_id: Optional[int] = Field(None, description="Optional associated inference run ID")
    radiologist_name: str = Field("Dr. AI Diagnostic System, M.D.", description="Reporting physician/system")
    clinical_history: Optional[str] = Field("Annual low-dose CT lung cancer screening. Patient reports intermittent cough.", description="Patient clinical indication")
    technique: Optional[str] = Field("Helical high-resolution volumetric chest CT acquisition without IV contrast.", description="Scan protocol description")
    findings: Optional[str] = Field(None, description="Detailed radiological findings (if omitted, auto-generated from inference)")
    impression: Optional[str] = Field(None, description="Summary diagnostic impression (if omitted, auto-generated from inference)")
    recommendations: Optional[str] = Field(None, description="Clinical management & follow-up recommendations (e.g. Fleischner Society guidelines)")


class ReportUpdate(BaseModel):
    radiologist_name: Optional[str] = None
    findings: Optional[str] = None
    impression: Optional[str] = None
    recommendations: Optional[str] = None
    status: Optional[str] = None


class ReportResponse(BaseModel):
    id: int
    scan_id: int
    patient_id: int
    inference_run_id: Optional[int] = None
    radiologist_name: str
    clinical_history: Optional[str] = None
    technique: Optional[str] = None
    findings: str
    impression: str
    recommendations: Optional[str] = None
    status: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class ReportListResponse(BaseModel):
    total: int
    items: List[ReportResponse]
