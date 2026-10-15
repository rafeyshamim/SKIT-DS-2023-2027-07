import io


def test_upload_and_process_scan(client):
    # 1. Create a patient first
    patient_res = client.post("/api/v1/patients", json={
        "medical_record_number": "MRN-SCAN-TEST-2002",
        "full_name": "Arthur Pendelton",
        "gender": "MALE"
    })
    patient_id = patient_res.json()["id"]

    # 2. Upload simulated CT scan file
    fake_scan_content = b"DICOM_HEADER_MOCK_STREAM_VOXEL_DATA_12345"
    file_tuple = ("patient_chest_scan.dcm", io.BytesIO(fake_scan_content), "application/dicom")

    upload_res = client.post(
        "/api/v1/scans/upload",
        data={
            "patient_id": patient_id,
            "modality": "CT",
            "anatomical_region": "Chest / Thorax",
            "auto_process": False
        },
        files={"file": file_tuple}
    )
    assert upload_res.status_code == 201
    scan_data = upload_res.json()
    assert "scan_id" in scan_data
    scan_id = scan_data["scan_id"]

    # 3. Explicitly trigger 3D reconstruction pipeline
    process_res = client.post(f"/api/v1/scans/{scan_id}/process")
    assert process_res.status_code == 200

    # 4. Query scan details
    detail_res = client.get(f"/api/v1/scans/{scan_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["patient_mrn"] == "MRN-SCAN-TEST-2002"
