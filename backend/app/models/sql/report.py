import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
import enum
from app.db.session import Base


class ReportStatusEnum(str, enum.Enum):
    DRAFT = "DRAFT"
    FINALIZED = "FINALIZED"
    REVISED = "REVISED"


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey("ct_scans.id", ondelete="CASCADE"), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    inference_run_id = Column(Integer, ForeignKey("inference_runs.id", ondelete="SET NULL"), nullable=True, index=True)
    
    radiologist_name = Column(String(128), default="Dr. AI Diagnostic System, M.D.", nullable=False)
    clinical_history = Column(Text, nullable=True)
    technique = Column(Text, default="Helical High-Resolution Multi-Detector Computed Tomography (MDCT) of the chest without IV contrast.", nullable=True)
    findings = Column(Text, nullable=False)
    impression = Column(Text, nullable=False)
    recommendations = Column(Text, nullable=True)
    
    status = Column(String(32), default=ReportStatusEnum.FINALIZED.value, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    # Relationships
    scan = relationship("CTScan", back_populates="reports")
    patient = relationship("Patient", back_populates="reports")
    inference_run = relationship("InferenceRun", back_populates="reports")

    def to_dict(self):
        return {
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
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
