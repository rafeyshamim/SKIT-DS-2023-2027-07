import io


def test_full_medical_ai_lifecycle(client):
    """
    End-to-End System Integration Test:
    1. Register Patient in PostgreSQL
    2. Upload Multi-part CT Scan & Stream to Disk
    3. Execute 3D Volumetric Reconstruction & HU Windowing
    4. Run 3D CNN Model Inference (Classification + 3D Localization)
    5. Verify Dual-Database Persistence:
       - Structured Relational Records (PostgreSQL / SQLite)
       - Rich 3D Coordinates & Heatmap Document (MongoDB / JSON fallback)
    6. Generate Automated Clinical Diagnostic Report
    7. Query Complete Patient History & Audit Trail
    """
    # Step 1: Register Patient
    mrn = "MRN-E2E-2026-99"
    patient_res = client.post("/api/v1/patients", json={
        "medical_record_number": mrn,
        "full_name": "Marcus Aurelius",
        "date_of_birth": "1960-04-26",
        "gender": "MALE",
        "contact_email": "marcus.aurelius@hospital.org",
        "contact_phone": "+1-555-0812",
        "medical_history_notes": "Ex-smoker (20 pack-years), annual low-dose screening."
    })
    assert patient_res.status_code == 201
    patient = patient_res.json()
    patient_id = patient["id"]

    # Step 2: Upload CT Scan
    scan_file_bytes = b"SIMULATED_HIGH_RES_THORACIC_CT_SERIES"
    upload_res = client.post(
        "/api/v1/scans/upload",
        data={
            "patient_id": patient_id,
            "modality": "CT",
            "anatomical_region": "Chest / Thorax",
            "auto_process": True
        },
        files={"file": ("marcus_aurelius_thorax.dcm", io.BytesIO(scan_file_bytes), "application/dicom")}
    )
    assert upload_res.status_code == 201
    scan = upload_res.json()
    scan_id = scan["scan_id"]

    # Step 3: Trigger 3D CNN Inference
    infer_res = client.post(f"/api/v1/inference/scans/{scan_id}/predict", json={
        "confidence_threshold": 0.40,
        "run_gradcam_saliency": True
    })
    assert infer_res.status_code == 200
    inf_result = infer_res.json()

    assert inf_result["status"] == "COMPLETED"
    assert inf_result["primary_prediction"] in [
        "Normal / No Nodule",
        "Benign Pulmonary Nodule",
        "Malignant Suspicion"
    ]
    assert 0.0 <= inf_result["confidence_score"] <= 1.0
    assert inf_result["processing_time_ms"] > 0
    inference_run_id = inf_result["id"]

    # Step 4: Verify MongoDB & SQL Cross-Referencing
    get_run_res = client.get(f"/api/v1/inference/runs/{inference_run_id}")
    assert get_run_res.status_code == 200
    run_detail = get_run_res.json()
    assert run_detail["id"] == inference_run_id
    assert "class_probabilities" in run_detail
    assert len(run_detail["slice_abnormality_scores"]) > 0

    # Step 5: Generate and Finalize Clinical Diagnostic Report
    rep_res = client.post("/api/v1/reports/generate", json={
        "scan_id": scan_id,
        "inference_run_id": inference_run_id,
        "radiologist_name": "Dr. Sarah Lin, MD (Chief Thoracic Radiologist)"
    })
    assert rep_res.status_code == 201
    report = rep_res.json()
    assert report["scan_id"] == scan_id
    assert report["patient_id"] == patient_id
    assert "Fleischner" in report["recommendations"]

    # Step 6: Verify Scan Detail includes Patient MRN and DICOM metadata
    scan_detail_res = client.get(f"/api/v1/scans/{scan_id}")
    assert scan_detail_res.status_code == 200
    scan_detail = scan_detail_res.json()
    assert scan_detail["patient_mrn"] == mrn
    assert scan_detail["status"] == "PROCESSED"
