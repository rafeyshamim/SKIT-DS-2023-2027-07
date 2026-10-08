from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.mongo import get_mongo
from app.models.sql.report import Report, ReportStatusEnum
from app.models.sql.scan import CTScan
from app.models.sql.patient import Patient
from app.models.sql.inference import InferenceRun
from app.schemas.report import (
    ReportCreate,
    ReportUpdate,
    ReportResponse,
    ReportListResponse
)
from app.services.report_service import report_service
from app.core.logging import logger

router = APIRouter()


@router.post("/generate", response_model=ReportResponse, status_code=status.HTTP_201_CREATED, summary="Generate Structured Clinical Diagnostic Report")
def generate_report(report_in: ReportCreate, db: Session = Depends(get_db)):
    """
    Generates an automated, structured radiological diagnostic report
    synthesizing patient history, CT scan parameters, and 3D CNN inference outputs.
    """
    scan = db.query(CTScan).filter(CTScan.id == report_in.scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")

    patient = scan.patient

    # Fetch inference run
    inference_run = None
    if report_in.inference_run_id:
        inference_run = db.query(InferenceRun).filter(InferenceRun.id == report_in.inference_run_id).first()
    else:
        # Pick latest completed run
        inference_run = db.query(InferenceRun).filter(
            InferenceRun.scan_id == scan.id,
            InferenceRun.status == "COMPLETED"
        ).order_by(InferenceRun.id.desc()).first()

    # Fetch MongoDB payload if available
    mongo = get_mongo()
    inference_doc = None
    if inference_run and inference_run.mongo_payload_id:
        inference_doc = mongo.get_document("inference_payloads", inference_run.mongo_payload_id)

    # Generate content using report service if not provided
    generated = report_service.generate_clinical_report(
        patient=patient,
        scan=scan,
        inference_run=inference_run,
        inference_doc=inference_doc,
        radiologist_name=report_in.radiologist_name
    )

    report = Report(
        scan_id=scan.id,
        patient_id=patient.id,
        inference_run_id=inference_run.id if inference_run else None,
        radiologist_name=report_in.radiologist_name or generated["radiologist_name"],
        clinical_history=report_in.clinical_history or generated["clinical_history"],
        technique=report_in.technique or generated["technique"],
        findings=report_in.findings or generated["findings"],
        impression=report_in.impression or generated["impression"],
        recommendations=report_in.recommendations or generated["recommendations"],
        status=ReportStatusEnum.FINALIZED.value
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    logger.info(f"Generated clinical report ID {report.id} for Scan ID {scan.id}, Patient {patient.medical_record_number}")
    return report


@router.get("/{report_id}", response_model=ReportResponse, summary="Get Report by ID")
def get_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")
    return report


@router.get("/scan/{scan_id}", response_model=ReportListResponse, summary="Get Reports for CT Scan")
def get_reports_by_scan(scan_id: int, db: Session = Depends(get_db)):
    reports = db.query(Report).filter(Report.scan_id == scan_id).order_by(Report.id.desc()).all()
    return {"total": len(reports), "items": reports}


@router.put("/{report_id}", response_model=ReportResponse, summary="Update / Finalize Clinical Report")
def update_report(report_id: int, report_update: ReportUpdate, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    data = report_update.model_dump(exclude_unset=True)
    for field, val in data.items():
        setattr(report, field, val)

    db.commit()
    db.refresh(report)
    logger.info(f"Updated clinical report ID {report.id}")
    return report
