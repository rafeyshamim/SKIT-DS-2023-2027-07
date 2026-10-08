import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
import enum
from app.db.session import Base


class InferenceStatusEnum(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RiskLevelEnum(str, enum.Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class InferenceRun(Base):
    __tablename__ = "inference_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey("ct_scans.id", ondelete="CASCADE"), nullable=False, index=True)
    
    model_name = Column(String(64), default="MedNet-3D-CNN", nullable=False)
    model_version = Column(String(32), default="v1.2.0", nullable=False)
    
    status = Column(String(32), default=InferenceStatusEnum.QUEUED.value, nullable=False)
    primary_prediction = Column(String(128), nullable=True)
    confidence_score = Column(Float, nullable=True)
    risk_level = Column(String(32), nullable=True)
    
    # ID reference to MongoDB inference_payloads collection for rich 3D bounding boxes & slice heatmaps
    mongo_payload_id = Column(String(64), nullable=True, index=True)
    
    processing_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    started_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    scan = relationship("CTScan", back_populates="inference_runs")
    reports = relationship("Report", back_populates="inference_run")

    def to_dict(self):
        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status,
            "primary_prediction": self.primary_prediction,
            "confidence_score": self.confidence_score,
            "risk_level": self.risk_level,
            "mongo_payload_id": self.mongo_payload_id,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
