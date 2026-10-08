import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, Enum
from sqlalchemy.orm import relationship
import enum
from app.db.session import Base


class GenderEnum(str, enum.Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    medical_record_number = Column(String(64), unique=True, index=True, nullable=False)
    full_name = Column(String(128), nullable=False)
    date_of_birth = Column(String(32), nullable=True)
    gender = Column(String(16), default=GenderEnum.OTHER.value)
    contact_email = Column(String(128), nullable=True)
    contact_phone = Column(String(32), nullable=True)
    medical_history_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    # Relationships
    scans = relationship("CTScan", back_populates="patient", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="patient", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "medical_record_number": self.medical_record_number,
            "full_name": self.full_name,
            "date_of_birth": self.date_of_birth,
            "gender": self.gender,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "medical_history_notes": self.medical_history_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
