import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
import enum
from app.db.session import Base


class ScanStatusEnum(str, enum.Enum):
    PENDING = "PENDING"
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class CTScan(Base):
    __tablename__ = "ct_scans"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_uid = Column(String(64), unique=True, index=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    
    modality = Column(String(32), default="CT")
    anatomical_region = Column(String(64), default="Chest / Thorax")
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    processed_volume_path = Column(String(512), nullable=True)
    
    file_size_bytes = Column(Integer, nullable=True)
    slice_count = Column(Integer, default=0)
    slice_thickness_mm = Column(Float, nullable=True)
    pixel_spacing_xy = Column(String(64), nullable=True)
    
    status = Column(String(32), default=ScanStatusEnum.UPLOADED.value, nullable=False)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    # Relationships
    patient = relationship("Patient", back_populates="scans")
    inference_runs = relationship("InferenceRun", back_populates="scan", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="scan", cascade="all, delete-orphan")

    def to_dict(self):
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
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
