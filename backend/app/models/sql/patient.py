"""SQLAlchemy model for patient demographics and clinical history records."""

import enum
from typing import Any, Dict, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Index,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship
from app.db.session import Base
from app.core.utils import get_utc_now


class GenderEnum(str, enum.Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"


def _enum_values(enum_cls):
    """Persist enum values in database for cross-engine consistency."""
    return [member.value for member in enum_cls]


class Patient(Base):
    """Represents a patient undergoing clinical CT diagnostic evaluation."""

    __tablename__ = "patients"
    __table_args__ = (
        Index("ix_patients_mrn", "medical_record_number", unique=True),
        Index("ix_patients_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    medical_record_number = Column(String(64), unique=True, index=True, nullable=False)
    full_name = Column(String(128), nullable=False)
    date_of_birth = Column(String(32), nullable=True)
    gender = Column(
        SAEnum(
            GenderEnum,
            native_enum=False,
            length=16,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        default=GenderEnum.OTHER,
        nullable=False,
    )
    contact_email = Column(String(128), nullable=True)
    contact_phone = Column(String(32), nullable=True)
    medical_history_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    # Relationships
    scans = relationship(
        "CTScan",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="desc(CTScan.id)",
    )
    reports = relationship(
        "Report",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="desc(Report.id)",
    )

    def __repr__(self) -> str:
        return f"<Patient id={self.id} mrn='{self.medical_record_number}' name='{self.full_name}'>"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes patient record to dictionary."""
        gender_val = self.gender.value if isinstance(self.gender, GenderEnum) else str(self.gender) if self.gender else None
        return {
            "id": self.id,
            "medical_record_number": self.medical_record_number,
            "full_name": self.full_name,
            "date_of_birth": self.date_of_birth,
            "gender": gender_val,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "medical_history_notes": self.medical_history_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
