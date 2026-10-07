from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.sql.patient import Patient
from app.schemas.patient import (
    PatientCreate,
    PatientUpdate,
    PatientResponse,
    PatientListResponse
)
from app.core.logging import logger

router = APIRouter()


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED, summary="Register New Patient")
def create_patient(patient_in: PatientCreate, db: Session = Depends(get_db)):
    """
    Registers a new patient record with a unique Medical Record Number (MRN).
    """
    existing = db.query(Patient).filter(Patient.medical_record_number == patient_in.medical_record_number).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Patient with MRN '{patient_in.medical_record_number}' already exists."
        )

    patient = Patient(
        medical_record_number=patient_in.medical_record_number,
        full_name=patient_in.full_name,
        date_of_birth=patient_in.date_of_birth,
        gender=patient_in.gender,
        contact_email=patient_in.contact_email,
        contact_phone=patient_in.contact_phone,
        medical_history_notes=patient_in.medical_history_notes,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    logger.info(f"Registered patient ID {patient.id} with MRN {patient.medical_record_number}")
    return patient


@router.get("", response_model=PatientListResponse, summary="List All Patients")
def list_patients(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieves paginated list of patients with optional search by name or MRN.
    """
    query = db.query(Patient)
    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            (Patient.full_name.ilike(search_filter)) | 
            (Patient.medical_record_number.ilike(search_filter))
        )

    total = query.count()
    patients = query.order_by(Patient.id.desc()).offset((page - 1) * limit).limit(limit).all()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": patients
    }


@router.get("/{patient_id}", response_model=PatientResponse, summary="Get Patient by ID")
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse, summary="Update Patient Details")
def update_patient(patient_id: int, patient_update: PatientUpdate, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    update_data = patient_update.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        setattr(patient, field, val)

    db.commit()
    db.refresh(patient)
    logger.info(f"Updated patient ID {patient.id}")
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete Patient Record")
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    db.delete(patient)
    db.commit()
    logger.info(f"Deleted patient ID {patient_id}")
    return None
