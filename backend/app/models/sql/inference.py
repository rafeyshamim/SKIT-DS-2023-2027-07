"""SQLAlchemy model for AI inference runs executed against CT scans."""

import enum
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship, validates
from app.core.utils import get_utc_now
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

def _enum_values(enum_cls):
    """Persist enum *values* (not member names) in the database."""
    return [member.value for member in enum_cls]


def _as_utc(dt: datetime) -> datetime:
    """Treat naive datetimes (e.g. from SQLite) as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class InferenceRun(Base):
    """A single model inference executed on a CT scan.

    Lifecycle: QUEUED -> RUNNING -> COMPLETED | FAILED
    """

    __tablename__ = "inference_runs"
    __table_args__ = (
        CheckConstraint(
            "confidence_score IS NULL "
            "OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="ck_inference_runs_confidence_range",
        ),
        CheckConstraint(
            "status <> 'COMPLETED' "
            "OR (completed_at IS NOT NULL AND primary_prediction IS NOT NULL)",
            name="ck_inference_runs_completed_has_result",
        ),
        CheckConstraint(
            "status <> 'FAILED' OR error_message IS NOT NULL",
            name="ck_inference_runs_failed_has_error",
        ),
        # Covers "all runs for a scan" and "runs for a scan in a given status".
        Index("ix_inference_runs_scan_status", "scan_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_id = Column(
        Integer,
        ForeignKey("ct_scans.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Set explicitly by the inference service so every run is traceable
    # to the exact model that produced it.
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=False)

    status = Column(
        SAEnum(
            InferenceStatusEnum,
            native_enum=False,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        default=InferenceStatusEnum.QUEUED,
        nullable=False,
        index=True,  # workers/dashboards poll by status
    )
    primary_prediction = Column(String(128), nullable=True)
    confidence_score = Column(Float, nullable=True)
    risk_level = Column(
        SAEnum(
            RiskLevelEnum,
            native_enum=False,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=True,
    )

    # Reference to MongoDB `inference_payloads` (3D bounding boxes, slice heatmaps).
    # No FK integrity across databases: clean up the payload when deleting a run.
    mongo_payload_id = Column(String(64), nullable=True, index=True)

    processing_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    scan = relationship("CTScan", back_populates="inference_runs")
    reports = relationship(
        "Report",
        back_populates="inference_run",
        passive_deletes=True,
    )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @validates("confidence_score")
    def _validate_confidence(self, _key: str, value: Optional[float]) -> Optional[float]:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("confidence_score must be between 0 and 1")
        return value

    # ------------------------------------------------------------------ #
    # State transitions
    # ------------------------------------------------------------------ #
    def mark_running(self) -> None:
        self.status = InferenceStatusEnum.RUNNING
        self.started_at = get_utc_now()

    def mark_completed(
        self,
        prediction: str,
        confidence: float,
        risk_level: Optional[RiskLevelEnum] = None,
        mongo_payload_id: Optional[str] = None,
    ) -> None:
        now = get_utc_now()
        self.status = InferenceStatusEnum.COMPLETED
        self.primary_prediction = prediction
        self.confidence_score = confidence
        self.risk_level = risk_level
        self.mongo_payload_id = mongo_payload_id
        self.completed_at = now
        self.error_message = None
        if self.started_at:
            delta = _as_utc(now) - _as_utc(self.started_at)
            self.processing_time_ms = delta.total_seconds() * 1000

    def mark_failed(self, error_message: str) -> None:
        now = get_utc_now()
        self.status = InferenceStatusEnum.FAILED
        self.error_message = error_message
        self.completed_at = now
        if self.started_at:
            delta = _as_utc(now) - _as_utc(self.started_at)
            self.processing_time_ms = delta.total_seconds() * 1000

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def is_finished(self) -> bool:
        return self.status in (
            InferenceStatusEnum.COMPLETED,
            InferenceStatusEnum.FAILED,
        )

    def __repr__(self) -> str:
        status = self.status.value if self.status else None
        return (
            f"<InferenceRun id={self.id} scan_id={self.scan_id} "
            f"model={self.model_name}:{self.model_version} status={status}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        def iso(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None

        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status.value if self.status else None,
            "primary_prediction": self.primary_prediction,
            "confidence_score": self.confidence_score,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "mongo_payload_id": self.mongo_payload_id,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
            "created_at": iso(self.created_at),
            "started_at": iso(self.started_at),
            "completed_at": iso(self.completed_at),
        }