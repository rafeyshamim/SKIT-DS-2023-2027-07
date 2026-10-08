import io


def test_3d_cnn_inference_pipeline(client):
    # 1. Create patient
    p_res = client.post("/api/v1/patients", json={
        "medical_record_number": "MRN-INF-TEST-3003",
        "full_name": "Gordon Freeman",
        "gender": "MALE",
        "medical_history_notes": "Heavy smoker, 30 pack-years. Hemoptysis."
    })
    patient_id = p_res.json()["id"]

    # 2. Upload scan
    fake_scan_content = b"SIMULATED_DICOM_DATA"
    upload_res = client.post(
        "/api/v1/scans/upload",
        data={
            "patient_id": patient_id,
            "modality": "CT",
            "anatomical_region": "Chest / Thorax",
            "auto_process": True
        },
        files={"file": ("gordon_chest.dcm", io.BytesIO(fake_scan_content), "application/dicom")}
    )
    scan_id = upload_res.json()["scan_id"]

    # 3. Trigger 3D CNN Inference
    infer_res = client.post(
        f"/api/v1/inference/scans/{scan_id}/predict",
        json={"confidence_threshold": 0.45, "run_gradcam_saliency": True}
    )
    assert infer_res.status_code == 200
    inf_data = infer_res.json()

    assert inf_data["status"] == "COMPLETED"
    assert inf_data["primary_prediction"] is not None
    assert inf_data["confidence_score"] > 0.0
    assert "class_probabilities" in inf_data
    assert len(inf_data["slice_abnormality_scores"]) > 0
    assert inf_data["processing_time_ms"] > 0.0

    run_id = inf_data["id"]

    # 4. Fetch specific inference run
    get_run_res = client.get(f"/api/v1/inference/runs/{run_id}")
    assert get_run_res.status_code == 200
    assert get_run_res.json()["primary_prediction"] == inf_data["primary_prediction"]

    # 5. Fetch axial slice heatmap overlay
    slice_heatmap_res = client.get(f"/api/v1/inference/scans/{scan_id}/slice-heatmap/16")
    assert slice_heatmap_res.status_code == 200
    slice_data = slice_heatmap_res.json()
    assert slice_data["slice_index"] == 16
    assert len(slice_data["normalized_heatmap_grid"]) == 16
