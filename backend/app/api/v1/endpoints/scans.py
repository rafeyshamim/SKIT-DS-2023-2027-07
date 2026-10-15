import os
import numpy as np
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
from app.core.config import settings
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


@router.get("/{scan_id}/slices/{slice_idx}", summary="Get 2D Slice Matrix for Interactive Scrubber")
def get_scan_slice(
    scan_id: int,
    slice_idx: int,
    plane: str = "axial",
    db: Session = Depends(get_db)
):
    """
    Returns 2D intensity grid for multiplanar reformatting (axial, coronal, sagittal).
    """
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan or not scan.processed_volume_path or not os.path.exists(scan.processed_volume_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Volume not found. Run processing first.")

    vol = np.load(scan.processed_volume_path)  # shape (1, D, H, W) or (D, H, W)
    if vol.ndim == 4:
        vol = vol[0]

    d, h, w = vol.shape

    if plane == "axial":
        total_slices = d
        if slice_idx < 0 or slice_idx >= d:
            slice_idx = max(0, min(d - 1, slice_idx))
        slice_2d = vol[slice_idx, :, :]
    elif plane == "coronal":
        total_slices = h
        if slice_idx < 0 or slice_idx >= h:
            slice_idx = max(0, min(h - 1, slice_idx))
        slice_2d = vol[:, slice_idx, :]
    elif plane == "sagittal":
        total_slices = w
        if slice_idx < 0 or slice_idx >= w:
            slice_idx = max(0, min(w - 1, slice_idx))
        slice_2d = vol[:, :, slice_idx]
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="plane must be axial, coronal, or sagittal")

    h_out, w_out = slice_2d.shape
    # Round to 3 decimals to reduce payload size over JSON
    pixels = np.round(slice_2d, 3).tolist()

    return {
        "scan_id": scan_id,
        "plane": plane,
        "slice_index": slice_idx,
        "total_slices": total_slices,
        "width": w_out,
        "height": h_out,
        "min_val": float(np.min(slice_2d)),
        "max_val": float(np.max(slice_2d)),
        "pixels": pixels
    }


@router.get("/{scan_id}/mesh3d", summary="Get 3D Organ Isosurface / Point Cloud Coordinates")
def get_scan_mesh3d(
    scan_id: int,
    iso_threshold: float = 0.25,
    max_points: int = 2500,
    db: Session = Depends(get_db)
):
    """
    Returns 3D spatial points [x, y, z, intensity] above the selected threshold
    for interactive WebGL / Canvas 3D rendering in React.
    """
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan or not scan.processed_volume_path or not os.path.exists(scan.processed_volume_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Volume not found.")

    vol = np.load(scan.processed_volume_path)
    if vol.ndim == 4:
        vol = vol[0]

    d, h, w = vol.shape
    z_coords, y_coords, x_coords = np.where(vol >= iso_threshold)
    
    total_found = len(z_coords)
    if total_found == 0:
        return {"scan_id": scan_id, "points": [], "total_points": 0, "dimensions": [d, h, w]}

    step = max(1, total_found // max_points)
    sampled_indices = np.arange(0, total_found, step)[:max_points]

    points = []
    for idx in sampled_indices:
        zi = int(z_coords[idx])
        yi = int(y_coords[idx])
        xi = int(x_coords[idx])
        val = float(round(float(vol[zi, yi, xi]), 3))
        # Normalize to centered coordinates [-1, 1]
        nx = round((xi / (w - 1)) * 2.0 - 1.0, 3)
        ny = round((yi / (h - 1)) * 2.0 - 1.0, 3)
        nz = round((zi / (d - 1)) * 2.0 - 1.0, 3)
        points.append([nx, ny, nz, val])

    return {
        "scan_id": scan_id,
        "points": points,
        "total_points": len(points),
        "dimensions": [d, h, w],
        "iso_threshold": iso_threshold
    }


@router.get("/{scan_id}/isosurface-html", summary="Get Interactive 3D Marching Cubes Isosurface HTML")
def get_scan_isosurface_html(scan_id: int, db: Session = Depends(get_db)):
    """
    Renders or serves an interactive 3D WebGL isosurface HTML generated via marching cubes.
    """
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan or not scan.processed_volume_path or not os.path.exists(scan.processed_volume_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Volume not found.")

    output_dir = os.path.join(settings.UPLOAD_DIR, "reconstruction")
    os.makedirs(output_dir, exist_ok=True)
    html_filename = f"scan_{scan.scan_uid}_isosurface_3d.html"
    html_path = os.path.join(output_dir, html_filename)

    if not os.path.exists(html_path):
        from src.reconstruction.reconstructor import VolumeReconstructor
        vol = np.load(scan.processed_volume_path)
        if vol.ndim == 4:
            vol = vol[0]
        recon = VolumeReconstructor(output_dir=output_dir)
        recon.plot_3d_isosurface(vol, title=f"3D Lung Reconstruction - {scan.scan_uid}", filename=html_filename)

    return FileResponse(html_path, media_type="text/html")


@router.get("/{scan_id}/mip", summary="Get Maximum Intensity Projection (MIP) Image")
def get_scan_mip(scan_id: int, db: Session = Depends(get_db)):
    """
    Renders or serves the Maximum Intensity Projection (MIP) along axial, coronal, and sagittal axes.
    """
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan or not scan.processed_volume_path or not os.path.exists(scan.processed_volume_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Volume not found.")

    output_dir = os.path.join(settings.UPLOAD_DIR, "reconstruction")
    os.makedirs(output_dir, exist_ok=True)
    mip_filename = f"scan_{scan.scan_uid}_mip.png"
    mip_path = os.path.join(output_dir, mip_filename)

    if not os.path.exists(mip_path):
        from src.reconstruction.reconstructor import VolumeReconstructor
        vol = np.load(scan.processed_volume_path)
        if vol.ndim == 4:
            vol = vol[0]
        recon = VolumeReconstructor(output_dir=output_dir)
        recon.plot_mip(vol, title=f"Maximum Intensity Projection - {scan.scan_uid}", filename=mip_filename)

    return FileResponse(mip_path, media_type="image/png")


@router.post("/sample", summary="Create Instant Demo Scan for Testing")
def create_sample_scan(
    modality: str = Form("CT"),
    anatomical_region: str = Form("Chest / Thorax"),
    patient_name: str = Form("John Anderson"),
    patient_mrn: str = Form("MRN-2026-DEMO"),
    db: Session = Depends(get_db)
):
    """
    Instantly provisions a patient and high-fidelity volumetric CT scan
    with full HU calibration, lung parenchyma, and pulmonary nodule.
    """
    patient = db.query(Patient).filter(Patient.medical_record_number == patient_mrn).first()
    if not patient:
        patient = Patient(
            medical_record_number=patient_mrn,
            full_name=patient_name,
            gender="MALE",
            date_of_birth="1968-05-14",
            contact_email="j.anderson@example.org",
            medical_history_notes="Long-term smoking history (35 pack-years), chronic cough, routine chest screening."
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # Generate synthetic DICOM scan
    scan_uid = f"DEMO_{os.urandom(4).hex().upper()}"
    raw_path = os.path.join(settings.UPLOAD_DIR, f"{scan_uid}.npy")
    
    # Process volume directly
    processed_path, metadata, final_shape = dicom_processor.process_and_save_pipeline(
        raw_path, scan_uid
    )

    scan = CTScan(
        scan_uid=scan_uid,
        patient_id=patient.id,
        modality=modality,
        anatomical_region=anatomical_region,
        original_filename=f"{scan_uid}_thoracic_helical.dcm",
        file_path=raw_path,
        file_size_bytes=1024 * 1024 * 4,
        status=ScanStatusEnum.PROCESSED.value,
        processed_volume_path=processed_path,
        slice_count=metadata.get("slice_count", 32),
        slice_thickness_mm=metadata.get("slice_thickness_mm", 1.25),
        pixel_spacing_xy="0.703x0.703"
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    mongo = get_mongo()
    mongo_doc = {
        "scan_id": scan.id,
        "scan_uid": scan.scan_uid,
        "scanner_manufacturer": metadata.get("scanner_manufacturer", "Siemens SOMATOM Force"),
        "slice_count": scan.slice_count,
        "slice_thickness_mm": scan.slice_thickness_mm,
        "pixel_spacing": [0.703, 0.703],
        "window_center": -600.0,
        "window_width": 1500.0,
        "volume_shape": list(final_shape),
    }
    mongo.insert_document("dicom_metadata", mongo_doc)

    return {
        "scan_id": scan.id,
        "scan_uid": scan.scan_uid,
        "patient_id": scan.patient_id,
        "modality": scan.modality,
        "anatomical_region": scan.anatomical_region,
        "original_filename": scan.original_filename,
        "status": scan.status,
        "slice_count": scan.slice_count,
        "message": "Demo scan synthesized and 3D reconstructed successfully.",
        "created_at": scan.created_at
    }
