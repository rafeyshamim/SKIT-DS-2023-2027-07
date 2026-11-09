"""SQLAlchemy model for computed tomography (CT) scans."""

import enum
from typing import Any, Dict, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Float,
    Index,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship
from app.db.session import Base
from app.core.utils import get_utc_now


class ScanStatusEnum(str, enum.Enum):
    PENDING = "PENDING"
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


def _enum_values(enum_cls):
    """Persist enum values in database for cross-engine consistency."""
    return [member.value for member in enum_cls]


class CTScan(Base):
    """Represents a thoracic CT scan volumetric dataset."""

    __tablename__ = "ct_scans"
    __table_args__ = (
        Index("ix_ct_scans_patient_created", "patient_id", "created_at"),
        Index("ix_ct_scans_status", "status"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_uid = Column(String(64), unique=True, index=True, nullable=False)
    patient_id = Column(
        Integer,
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    modality = Column(String(32), default="CT")
    anatomical_region = Column(String(64), default="Chest / Thorax")
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    processed_volume_path = Column(String(512), nullable=True)

    file_size_bytes = Column(Integer, nullable=True)
    slice_count = Column(Integer, default=0)
    slice_thickness_mm = Column(Float, nullable=True)
    pixel_spacing_xy = Column(String(64), nullable=True)

    status = Column(
        SAEnum(
            ScanStatusEnum,
            native_enum=False,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        default=ScanStatusEnum.UPLOADED,
        nullable=False,
        index=True,
    )
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    # Relationships
    patient = relationship("Patient", back_populates="scans")
    inference_runs = relationship(
        "InferenceRun",
        back_populates="scan",
        cascade="all, delete-orphan",
        order_by="desc(InferenceRun.id)",
    )
    reports = relationship(
        "Report",
        back_populates="scan",
        cascade="all, delete-orphan",
        order_by="desc(Report.id)",
    )

    @property
    def is_processed(self) -> bool:
        """Returns True if the 3D volume reconstruction pipeline completed successfully."""
        return self.status == ScanStatusEnum.PROCESSED

    def __repr__(self) -> str:
        status_str = self.status.value if isinstance(self.status, ScanStatusEnum) else str(self.status)
        return f"<CTScan id={self.id} uid='{self.scan_uid}' patient_id={self.patient_id} status='{status_str}'>"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes CT scan record to dictionary."""
        status_val = self.status.value if isinstance(self.status, ScanStatusEnum) else str(self.status) if self.status else None
        return {
            "id": self.id,
            "scan_uid": self.scan_uid,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "anatomical_region": self.anatomical_region,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "processed_volume_path": self.processed_volume_path,
            "file_size_bytes": self.file_size_bytes,
            "slice_count": self.slice_count,
            "slice_thickness_mm": self.slice_thickness_mm,
            "pixel_spacing_xy": self.pixel_spacing_xy,
            "status": status_val,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
