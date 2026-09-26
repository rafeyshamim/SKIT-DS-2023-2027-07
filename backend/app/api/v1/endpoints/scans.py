import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.mongo import get_mongo
from app.models.sql.scan import CTScan, ScanStatusEnum
from app.models.sql.patient import Patient
from app.schemas.scan import (
    ScanUploadResponse,
    ScanResponse,
    ScanDetailResponse,
    ScanListResponse
)
from app.services.storage_service import storage_service
from app.services.dicom_processor import dicom_processor
from app.core.logging import logger

router = APIRouter()


def execute_scan_processing_task(scan_id: int):
    """Background task to run 3D volumetric reconstruction and HU windowing pipeline."""
    from app.db.session import SessionLocal
    db = SessionLocal()
    mongo = get_mongo()
    try:
        scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
        if not scan:
            return

        scan.status = ScanStatusEnum.PROCESSING.value
        db.commit()

        # Check if file is a zip archive
        file_to_process = storage_service.extract_zip_if_needed(scan.file_path, scan.scan_uid)

        # Execute 3D pipeline
        processed_path, metadata, final_shape = dicom_processor.process_and_save_pipeline(
            file_to_process, scan.scan_uid
        )

        # Update SQL record
        scan.processed_volume_path = processed_path
        scan.slice_count = metadata.get("slice_count", 32)
        scan.slice_thickness_mm = metadata.get("slice_thickness_mm", 1.25)
        pixel_spacing = metadata.get("pixel_spacing", [0.703, 0.703])
        scan.pixel_spacing_xy = f"{pixel_spacing[0]}x{pixel_spacing[1]}"
        scan.status = ScanStatusEnum.PROCESSED.value
        db.commit()

        # Update MongoDB DICOM metadata document
        mongo_doc = {
            "scan_id": scan.id,
            "scan_uid": scan.scan_uid,
            "series_instance_uid": metadata.get("series_instance_uid"),
            "study_instance_uid": metadata.get("study_instance_uid"),
            "scanner_manufacturer": metadata.get("scanner_manufacturer"),
            "slice_count": scan.slice_count,
            "slice_thickness_mm": scan.slice_thickness_mm,
            "pixel_spacing": pixel_spacing,
            "rescale_slope": metadata.get("rescale_slope", 1.0),
            "rescale_intercept": metadata.get("rescale_intercept", -1024.0),
            "window_center": metadata.get("window_center", -600.0),
            "window_width": metadata.get("window_width", 1500.0),
            "volume_shape": list(final_shape),
        }
        mongo.insert_document("dicom_metadata", mongo_doc)
        logger.info(f"Scan ID {scan_id} successfully reconstructed and processed.")

    except Exception as e:
        logger.error(f"Error processing scan {scan_id}: {e}")
        scan.status = ScanStatusEnum.FAILED.value
        scan.error_message = str(e)
        db.commit()
    finally:
        db.close()


@router.post("/upload", response_model=ScanUploadResponse, status_code=status.HTTP_201_CREATED, summary="Upload CT Scan (DICOM/ZIP/NIfTI)")
def upload_scan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CT Scan file (.dcm, .zip archive of DICOMs, .nii, .nii.gz, or .npy)"),
    patient_id: int = Form(..., description="Foreign key of patient"),
    modality: str = Form("CT", description="Imaging modality"),
    anatomical_region: str = Form("Chest / Thorax", description="Anatomical region"),
    auto_process: bool = Form(True, description="Automatically trigger 3D reconstruction pipeline"),
    db: Session = Depends(get_db)
):
    """
    Receives multi-part file upload, verifies patient, stores raw binary in storage,
    registers scan in PostgreSQL, and optionally triggers 3D reconstruction pipeline.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient ID {patient_id} not found.")

    # Save file
    scan_uid, file_path, file_size = storage_service.save_upload_file(file)

    scan = CTScan(
        scan_uid=scan_uid,
        patient_id=patient.id,
        modality=modality,
        anatomical_region=anatomical_region,
        original_filename=file.filename or "unknown_scan.dcm",
        file_path=file_path,
        file_size_bytes=file_size,
        status=ScanStatusEnum.UPLOADED.value
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    if auto_process:
        background_tasks.add_task(execute_scan_processing_task, scan.id)

    return {
        "scan_id": scan.id,
        "scan_uid": scan.scan_uid,
        "patient_id": scan.patient_id,
        "modality": scan.modality,
        "anatomical_region": scan.anatomical_region,
        "original_filename": scan.original_filename,
        "file_size_bytes": scan.file_size_bytes,
        "status": ScanStatusEnum.PROCESSING.value if auto_process else scan.status,
        "message": "Scan uploaded successfully. 3D reconstruction pipeline initiated." if auto_process else "Scan uploaded successfully.",
        "created_at": scan.created_at
    }


@router.post("/{scan_id}/process", response_model=ScanResponse, summary="Trigger 3D Reconstruction Pipeline")
def process_scan(scan_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Explicitly triggers the 3D volumetric reconstruction and HU normalization pipeline.
    """
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")

    scan.status = ScanStatusEnum.PROCESSING.value
    db.commit()
    background_tasks.add_task(execute_scan_processing_task, scan.id)
    return scan


@router.get("", response_model=ScanListResponse, summary="List All CT Scans")
def list_scans(
    patient_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(CTScan)
    if patient_id:
        query = query.filter(CTScan.patient_id == patient_id)
    if status_filter:
        query = query.filter(CTScan.status == status_filter)

    scans = query.order_by(CTScan.id.desc()).all()
    return {"total": len(scans), "items": scans}


@router.get("/{scan_id}", response_model=ScanDetailResponse, summary="Get Scan Details & DICOM Metadata")
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")

    mongo = get_mongo()
    mongo_docs = mongo.query_documents("dicom_metadata", {"scan_id": scan.id})
    dicom_meta = mongo_docs[0] if mongo_docs else None

    return {
        **scan.to_dict(),
        "patient_mrn": scan.patient.medical_record_number if scan.patient else None,
        "patient_name": scan.patient.full_name if scan.patient else None,
        "dicom_metadata": dicom_meta
    }


@router.get("/{scan_id}/download-volume", summary="Download Preprocessed 3D Volume (.npy)")
def download_processed_volume(scan_id: int, db: Session = Depends(get_db)):
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan or not scan.processed_volume_path or not os.path.exists(scan.processed_volume_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed 3D volume not found on disk.")

    return FileResponse(
        path=scan.processed_volume_path,
        media_type="application/octet-stream",
        filename=f"{scan.scan_uid}_volume.npy"
    )
