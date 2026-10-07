import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


class PatientBase(BaseModel):
    medical_record_number: str = Field(..., description="Unique Medical Record Number (MRN)", json_schema_extra={"example": "MRN-2026-9042"})
    full_name: str = Field(..., description="Full name of patient", json_schema_extra={"example": "Jane Doe"})
    date_of_birth: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)", json_schema_extra={"example": "1968-04-12"})
    gender: Optional[str] = Field("OTHER", description="Gender (MALE, FEMALE, OTHER)", json_schema_extra={"example": "FEMALE"})
    contact_email: Optional[str] = Field(None, json_schema_extra={"example": "jane.doe@example.com"})
    contact_phone: Optional[str] = Field(None, json_schema_extra={"example": "+1-555-0199"})
    medical_history_notes: Optional[str] = Field(None, description="Clinical background, smoking history, prior oncologic history")


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    medical_history_notes: Optional[str] = None


class PatientResponse(PatientBase):
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class PatientListResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: List[PatientResponse]
