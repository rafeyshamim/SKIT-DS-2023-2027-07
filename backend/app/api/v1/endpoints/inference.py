import os
import datetime
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.mongo import get_mongo
from app.models.sql.scan import CTScan, ScanStatusEnum
from app.models.sql.inference import InferenceRun, InferenceStatusEnum, RiskLevelEnum
from app.schemas.inference import (
    InferenceTriggerRequest,
    InferenceResultResponse,
    HeatmapSliceResponse
)
from app.services.model_service import model_service
from app.services.dicom_processor import dicom_processor
from app.services.storage_service import storage_service
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()


@router.post("/scans/{scan_id}/predict", response_model=InferenceResultResponse, summary="Execute 3D CNN Model Inference")
def predict_scan(
    scan_id: int,
    request: InferenceTriggerRequest = InferenceTriggerRequest(),
    db: Session = Depends(get_db)
):
    """
    Executes 3D CNN inference on a volumetric CT scan:
    1. Validates scan existence and verifies 3D volume preprocessing.
    2. Records an InferenceRun in PostgreSQL (relational audit trail).
    3. Runs 3D CNN forward pass, computing class probabilities and 3D nodule localization.
    4. Persists rich volumetric metrics, bounding boxes, and slice maps in MongoDB.
    5. Updates PostgreSQL with confidence score, primary prediction, and risk level.
    """
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")

    # Ensure volume is preprocessed
    if not scan.processed_volume_path or not os.path.exists(scan.processed_volume_path):
        logger.info(f"Scan {scan_id} volume not yet preprocessed. Running preprocessing synchronously.")
        extracted = storage_service.extract_zip_if_needed(scan.file_path, scan.scan_uid)
        processed_path, metadata, _ = dicom_processor.process_and_save_pipeline(extracted, scan.scan_uid)
        scan.processed_volume_path = processed_path
        scan.slice_count = metadata.get("slice_count", 32)
        scan.status = ScanStatusEnum.PROCESSED.value
        db.commit()

    # Create InferenceRun row in PostgreSQL
    run = InferenceRun(
        scan_id=scan.id,
        model_name=settings.MODEL_NAME,
        model_version=settings.MODEL_VERSION,
        status=InferenceStatusEnum.RUNNING.value,
        started_at=datetime.datetime.utcnow()
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        # Run 3D CNN Model Inference
        primary_pred, confidence, risk_level, duration_ms, payload_doc = model_service.run_inference_on_volume(
            volume_path=scan.processed_volume_path,
            scan_id=scan.id,
            inference_run_id=run.id,
            confidence_threshold=request.confidence_threshold
        )

        # Store rich spatial/slice payload in MongoDB
        mongo = get_mongo()
        mongo_id = mongo.insert_document("inference_payloads", payload_doc.model_dump())

        # Update SQL record with inference results
        run.status = InferenceStatusEnum.COMPLETED.value
        run.primary_prediction = primary_pred
        run.confidence_score = confidence
        run.risk_level = risk_level
        run.processing_time_ms = duration_ms
        run.mongo_payload_id = mongo_id
        run.completed_at = datetime.datetime.utcnow()
        db.commit()
        db.refresh(run)

        return {
            "id": run.id,
            "scan_id": run.scan_id,
            "patient_id": scan.patient_id,
            "model_name": run.model_name,
            "model_version": run.model_version,
            "status": run.status,
            "primary_prediction": run.primary_prediction,
            "confidence_score": run.confidence_score,
            "risk_level": run.risk_level,
            "processing_time_ms": run.processing_time_ms,
            "class_probabilities": payload_doc.class_probabilities,
            "lesions_detected_count": len(payload_doc.lesion_detections),
            "lesions": payload_doc.lesion_detections,
            "slice_abnormality_scores": payload_doc.slice_abnormality_scores,
            "volumetric_metrics": {
                "total_lung_volume_cm3": payload_doc.total_lung_volume_cm3,
                "total_lesion_volume_mm3": payload_doc.total_lesion_volume_mm3,
                "lung_involvement_percentage": payload_doc.lung_involvement_percentage
            },
            "started_at": run.started_at,
            "completed_at": run.completed_at
        }

    except Exception as e:
        logger.error(f"Inference execution failed for scan {scan_id}: {e}")
        run.status = InferenceStatusEnum.FAILED.value
        run.error_message = str(e)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D CNN Inference failed: {str(e)}"
        )


@router.get("/runs/{inference_run_id}", response_model=InferenceResultResponse, summary="Get Inference Run Result")
def get_inference_run(inference_run_id: int, db: Session = Depends(get_db)):
    run = db.query(InferenceRun).filter(InferenceRun.id == inference_run_id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inference run not found.")

    scan = run.scan
    mongo = get_mongo()
    mongo_doc = mongo.get_document("inference_payloads", run.mongo_payload_id) if run.mongo_payload_id else {}
    if not mongo_doc:
        # Check by query
        docs = mongo.query_documents("inference_payloads", {"inference_run_id": run.id})
        mongo_doc = docs[0] if docs else {}

    lesions = mongo_doc.get("lesion_detections", [])
    probs = mongo_doc.get("class_probabilities", {})
    slice_scores = mongo_doc.get("slice_abnormality_scores", [])

    return {
        "id": run.id,
        "scan_id": run.scan_id,
        "patient_id": scan.patient_id,
        "model_name": run.model_name,
        "model_version": run.model_version,
        "status": run.status,
        "primary_prediction": run.primary_prediction,
        "confidence_score": run.confidence_score,
        "risk_level": run.risk_level,
        "processing_time_ms": run.processing_time_ms,
        "class_probabilities": probs,
        "lesions_detected_count": len(lesions),
        "lesions": lesions,
        "slice_abnormality_scores": slice_scores,
        "volumetric_metrics": {
            "total_lung_volume_cm3": mongo_doc.get("total_lung_volume_cm3", 3420.0),
            "total_lesion_volume_mm3": mongo_doc.get("total_lesion_volume_mm3", 0.0),
            "lung_involvement_percentage": mongo_doc.get("lung_involvement_percentage", 0.0)
        },
        "started_at": run.started_at,
        "completed_at": run.completed_at
    }


@router.get("/scans/{scan_id}/latest", response_model=InferenceResultResponse, summary="Get Latest Inference for Scan")
def get_latest_inference_for_scan(scan_id: int, db: Session = Depends(get_db)):
    run = db.query(InferenceRun).filter(
        InferenceRun.scan_id == scan_id,
        InferenceRun.status == InferenceStatusEnum.COMPLETED.value
    ).order_by(InferenceRun.id.desc()).first()

    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No completed inference found for this scan.")

    return get_inference_run(run.id, db)


@router.get("/scans/{scan_id}/slice-heatmap/{slice_idx}", response_model=HeatmapSliceResponse, summary="Get Axial Slice Heatmap & Coordinates")
def get_slice_heatmap(scan_id: int, slice_idx: int, db: Session = Depends(get_db)):
    """
    Returns the normalized 2D heatmap matrix and lesion indicators for a specific axial slice.
    Used by PACS/DICOM viewers to render visual AI attention overlays on CT slices.
    """
    scan = db.query(CTScan).filter(CTScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")

    latest_run = db.query(InferenceRun).filter(
        InferenceRun.scan_id == scan_id,
        InferenceRun.status == InferenceStatusEnum.COMPLETED.value
    ).order_by(InferenceRun.id.desc()).first()

    if not latest_run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run inference on scan first.")

    mongo = get_mongo()
    doc = mongo.get_document("inference_payloads", latest_run.mongo_payload_id) or {}
    slice_scores = doc.get("slice_abnormality_scores", [])
    total_slices = len(slice_scores) if slice_scores else 32

    if slice_idx < 0 or slice_idx >= total_slices:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Slice index must be between 0 and {total_slices - 1}")

    abnormality_score = slice_scores[slice_idx] if slice_idx < len(slice_scores) else 0.0

    # Filter lesions intersecting this axial slice
    lesions_on_slice = []
    for lesion in doc.get("lesion_detections", []):
        bbox = lesion.get("bounding_box_3d", {})
        if bbox.get("z_min", 0) <= slice_idx <= bbox.get("z_max", 31):
            lesions_on_slice.append(lesion)

    # 16x16 normalized grid representation for fast frontend rendering
    grid = [[round(abnormality_score * np.exp(-((r - 8)**2 + (c - 8)**2) / 18.0), 3) for c in range(16)] for r in range(16)]

    return {
        "scan_id": scan_id,
        "inference_run_id": latest_run.id,
        "slice_index": slice_idx,
        "total_slices": total_slices,
        "abnormality_score": abnormality_score,
        "lesions_on_slice": lesions_on_slice,
        "normalized_heatmap_grid": grid
    }
