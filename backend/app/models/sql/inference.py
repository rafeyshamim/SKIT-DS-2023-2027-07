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
    """Persist enum values (not member names) in the database for cross-engine consistency."""
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
        # Composite index for querying all runs or runs by status for a specific scan
        Index("ix_inference_runs_scan_status", "scan_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    scan_id = Column(
        Integer,
        ForeignKey("ct_scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Set explicitly by the inference service so every run is traceable
    # to the exact model that produced it.
    model_name = Column(String(64), nullable=False, default="MedNet-3D-CNN")
    model_version = Column(String(32), nullable=False, default="v1.2.0")

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
        index=True,
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
    mongo_payload_id = Column(String(64), nullable=True, index=True)

    processing_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, default=get_utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    scan = relationship("CTScan", back_populates="inference_runs")
    reports = relationship(
        "Report",
        back_populates="inference_run",
        passive_deletes=True,
    )

    # ------------------------------------------------------------------ #
    # Compatibility properties
    # ------------------------------------------------------------------ #
    @property
    def created_at(self) -> datetime:
        """Alias for started_at for backward-compatibility with schemas."""
        return self.started_at

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @validates("confidence_score")
    def _validate_confidence(self, _key: str, value: Optional[float]) -> Optional[float]:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return value

    # ------------------------------------------------------------------ #
    # State transitions
    # ------------------------------------------------------------------ #
    def mark_running(self) -> None:
        """Transitions state to RUNNING and records execution start timestamp."""
        self.status = InferenceStatusEnum.RUNNING
        self.started_at = get_utc_now()

    def mark_completed(
        self,
        prediction: str,
        confidence: float,
        risk_level: Optional[Any] = None,
        mongo_payload_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Transitions state to COMPLETED and stores prediction deliverables."""
        now = get_utc_now()
        self.status = InferenceStatusEnum.COMPLETED
        self.primary_prediction = prediction
        self.confidence_score = confidence
        self.risk_level = risk_level
        self.mongo_payload_id = mongo_payload_id
        self.completed_at = now
        self.error_message = None

        if duration_ms is not None:
            self.processing_time_ms = duration_ms
        elif self.started_at:
            delta = _as_utc(now) - _as_utc(self.started_at)
            self.processing_time_ms = round(delta.total_seconds() * 1000.0, 2)

    def mark_failed(self, error_message: str) -> None:
        """Transitions state to FAILED and records exception diagnostics."""
        now = get_utc_now()
        self.status = InferenceStatusEnum.FAILED
        self.error_message = error_message
        self.completed_at = now
        if self.started_at:
            delta = _as_utc(now) - _as_utc(self.started_at)
            self.processing_time_ms = round(delta.total_seconds() * 1000.0, 2)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def is_finished(self) -> bool:
        """True if execution has reached a terminal state (COMPLETED or FAILED)."""
        return self.status in (
            InferenceStatusEnum.COMPLETED,
            InferenceStatusEnum.FAILED,
        )

    def __repr__(self) -> str:
        status_str = self.status.value if isinstance(self.status, InferenceStatusEnum) else str(self.status)
        return (
            f"<InferenceRun id={self.id} scan_id={self.scan_id} "
            f"model='{self.model_name}:{self.model_version}' status='{status_str}'>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes inference run record to dictionary."""
        def iso(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None

        status_val = self.status.value if isinstance(self.status, InferenceStatusEnum) else str(self.status) if self.status else None
        risk_val = self.risk_level.value if isinstance(self.risk_level, RiskLevelEnum) else str(self.risk_level) if self.risk_level else None

        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": status_val,
            "primary_prediction": self.primary_prediction,
            "confidence_score": self.confidence_score,
            "risk_level": risk_val,
            "mongo_payload_id": self.mongo_payload_id,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
            "created_at": iso(self.started_at),
            "started_at": iso(self.started_at),
            "completed_at": iso(self.completed_at),
        }