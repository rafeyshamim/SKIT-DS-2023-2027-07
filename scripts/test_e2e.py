import requests
import json
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_full_pipeline():
    print("1. Checking System Health...")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health = r.json()
    print(f"   Status: {health['status']} | Model: {health['ml_inference']['model_name']}")

    print("2. Provisioning Sample CT Scan...")
    r = requests.post(f"{BASE_URL}/scans/sample", data={
        "patient_name": "Eleanor Vance",
        "patient_mrn": "MRN-2026-VAL"
    })
    assert r.status_code == 200, f"Sample creation failed: {r.text}"
    scan = r.json()
    scan_id = scan["scan_id"]
    print(f"   Created Scan ID: {scan_id} (UID: {scan['scan_uid']}, Slices: {scan['slice_count']})")

    print("3. Fetching 2D Slice Data for Multiplanar Scrubber...")
    r = requests.get(f"{BASE_URL}/scans/{scan_id}/slices/16?plane=axial")
    assert r.status_code == 200, f"Slice fetch failed: {r.text}"
    slice_data = r.json()
    print(f"   Axial Slice 16 Grid: {slice_data['width']}x{slice_data['height']} pixels")

    print("4. Fetching 3D Organ Mesh Points for WebGL...")
    r = requests.get(f"{BASE_URL}/scans/{scan_id}/mesh3d?iso_threshold=0.25&max_points=1500")
    assert r.status_code == 200, f"3D mesh fetch failed: {r.text}"
    mesh_data = r.json()
    print(f"   Extracted 3D Points: {mesh_data['total_points']} points")

    print("5. Executing 3D CNN Volumetric Inference...")
    r = requests.post(f"{BASE_URL}/inference/scans/{scan_id}/predict", json={
        "confidence_threshold": 0.45
    })
    assert r.status_code == 200, f"Inference failed: {r.text}"
    result = r.json()
    print(f"   Diagnosis: {result['primary_prediction']}")
    print(f"   Confidence: {result['confidence_score'] * 100:.1f}%")
    print(f"   Risk Category: {result['risk_level']}")
    print(f"   Lesions Detected: {result['lesions_detected_count']}")

    print("6. Generating Fleischner Society Diagnostic Report...")
    r = requests.post(f"{BASE_URL}/reports/generate", json={
        "scan_id": scan_id,
        "inference_run_id": result["id"],
        "radiologist_name": "Dr. Elena Rostova, MD"
    })
    assert r.status_code in (200, 201), f"Report generation failed: {r.text}"
    rep = r.json()
    print(f"   Report Status: {rep['status']}")

    print("7. Checking Model Training Dashboard Status...")
    r = requests.get(f"{BASE_URL}/training/status")
    assert r.status_code == 200, f"Training status failed: {r.text}"
    train_status = r.json()
    print(f"   Dataset: {train_status['dataset_name']} ({train_status['dataset_size_mb']} MB)")
    print(f"   Saved Checkpoint: {train_status['checkpoint']['checkpoint_exists']} ({train_status['checkpoint']['size_mb']} MB)")

    print("\n>>> ALL 7 END-TO-END PIPELINE CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    try:
        test_full_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
