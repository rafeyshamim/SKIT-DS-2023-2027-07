"""SQLAlchemy model for clinical radiological diagnostic reports."""

import enum
from typing import Any, Dict, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Index,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship
from app.db.session import Base
from app.core.utils import get_utc_now


class ReportStatusEnum(str, enum.Enum):
    """Clinical sign-off status of a diagnostic report."""
    DRAFT = "DRAFT"
    FINALIZED = "FINALIZED"
    REVISED = "REVISED"


def _enum_values(enum_cls):
    """Persist enum values in database for cross-engine compatibility (PostgreSQL/SQLite)."""
    return [member.value for member in enum_cls]


class Report(Base):
    """Clinical Radiological Diagnostic Report.

    Synthesizes patient demographic information, imaging acquisition parameters,
    and 3D CNN inference outputs (lesion volume, location, malignancy risk stratification)
    into structured clinical documentation following Fleischner Society guidelines.
    """

    __tablename__ = "reports"
    __table_args__ = (
        Index("ix_reports_scan_status", "scan_id", "status"),
        Index("ix_reports_patient_created", "patient_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(
        Integer,
        ForeignKey("ct_scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    patient_id = Column(
        Integer,
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    inference_run_id = Column(
        Integer,
        ForeignKey("inference_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    radiologist_name = Column(
        String(128),
        default="Dr. AI Diagnostic System, M.D.",
        nullable=False,
    )
    clinical_history = Column(Text, nullable=True)
    technique = Column(
        Text,
        default="Helical High-Resolution Multi-Detector Computed Tomography (MDCT) of the chest without IV contrast.",
        nullable=True,
    )
    findings = Column(Text, nullable=False)
    impression = Column(Text, nullable=False)
    recommendations = Column(Text, nullable=True)

    status = Column(
        SAEnum(
            ReportStatusEnum,
            native_enum=False,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        default=ReportStatusEnum.FINALIZED,
        nullable=False,
        index=True,
    )

    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    # Relationships
    scan = relationship("CTScan", back_populates="reports")
    patient = relationship("Patient", back_populates="reports")
    inference_run = relationship("InferenceRun", back_populates="reports")

    # ------------------------------------------------------------------ #
    # Domain Helper Properties and Methods
    # ------------------------------------------------------------------ #
    @property
    def is_finalized(self) -> bool:
        """Checks if the report has received clinical sign-off."""
        return self.status == ReportStatusEnum.FINALIZED

    def finalize(self, radiologist_name: Optional[str] = None) -> None:
        """Mark report as finalized with optional signing radiologist."""
        if radiologist_name:
            self.radiologist_name = radiologist_name
        self.status = ReportStatusEnum.FINALIZED
        self.updated_at = get_utc_now()

    def revise(self, impression: Optional[str] = None, recommendations: Optional[str] = None) -> None:
        """Mark report as revised with updated clinical findings."""
        if impression:
            self.impression = impression
        if recommendations:
            self.recommendations = recommendations
        self.status = ReportStatusEnum.REVISED
        self.updated_at = get_utc_now()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to dictionary with optional populated relation fields."""
        data = {
            "id": self.id,
            "scan_id": self.scan_id,
            "patient_id": self.patient_id,
            "inference_run_id": self.inference_run_id,
            "radiologist_name": self.radiologist_name,
            "clinical_history": self.clinical_history,
            "technique": self.technique,
            "findings": self.findings,
            "impression": self.impression,
            "recommendations": self.recommendations,
            "status": self.status.value if isinstance(self.status, ReportStatusEnum) else str(self.status),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        # Include patient / scan metadata if relations are loaded without triggering lazy IO
        if "patient" in self.__dict__ and self.patient:
            data["patient_mrn"] = getattr(self.patient, "medical_record_number", None)
            data["patient_name"] = getattr(self.patient, "full_name", None)
        if "scan" in self.__dict__ and self.scan:
            data["scan_uid"] = getattr(self.scan, "scan_uid", None)

        return data

    def __repr__(self) -> str:
        return (
            f"<Report id={self.id} scan_id={self.scan_id} "
            f"patient_id={self.patient_id} status={self.status} "
            f"radiologist='{self.radiologist_name}'>"
        )
