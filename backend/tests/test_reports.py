import io


def test_clinical_report_generation(client):
    # 1. Setup patient, scan, and run inference
    p_res = client.post("/api/v1/patients", json={
        "medical_record_number": "MRN-REP-TEST-4004",
        "full_name": "Sarah Connor",
        "gender": "FEMALE",
        "medical_history_notes": "Follow-up for solitary pulmonary nodule."
    })
    patient_id = p_res.json()["id"]

    upload_res = client.post(
        "/api/v1/scans/upload",
        data={"patient_id": patient_id, "auto_process": True},
        files={"file": ("sarah_chest.dcm", io.BytesIO(b"DICOM_SLICE_BYTES"), "application/dicom")}
    )
    scan_id = upload_res.json()["scan_id"]

    infer_res = client.post(f"/api/v1/inference/scans/{scan_id}/predict")
    inference_run_id = infer_res.json()["id"]

    # 2. Generate Automated Diagnostic Report
    rep_res = client.post("/api/v1/reports/generate", json={
        "scan_id": scan_id,
        "inference_run_id": inference_run_id,
        "radiologist_name": "Dr. Alice Morgan, MD (Thoracic Radiology)"
    })
    assert rep_res.status_code == 201
    report_data = rep_res.json()
    assert report_data["scan_id"] == scan_id
    assert report_data["patient_id"] == patient_id
    assert len(report_data["findings"]) > 0
    assert len(report_data["impression"]) > 0
    assert "Fleischner" in report_data["recommendations"]

    # 3. Update / Radiologist Sign-off
    rep_id = report_data["id"]
    update_res = client.put(f"/api/v1/reports/{rep_id}", json={
        "status": "FINALIZED",
        "impression": "FINAL SIGN-OFF: Confirmed suspicious nodule. Urgent referral advised."
    })
    assert update_res.status_code == 200
    assert update_res.json()["status"] == "FINALIZED"
