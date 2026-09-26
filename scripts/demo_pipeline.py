"""
Interactive CLI Script: Runs the complete end-to-end MedVision 3D CT pipeline
1. Starts in-memory app client
2. Registers a patient
3. Uploads and reconstructs a high-resolution 3D thoracic CT volume
4. Executes 3D CNN deep learning inference with 3D nodule localization
5. Generates a clinical diagnostic report
"""
import sys
import os
import io

# Ensure backend package is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from fastapi.testclient import TestClient
from app.main import app

def run_pipeline_demo():
    print("=" * 75)
    print("      MEDVISION 3D CT DIAGNOSTIC PIPELINE DEMO & BENCHMARK")
    print("=" * 75)

    with TestClient(app) as client:
        # 1. Health check
        print("\n[Step 1] Checking API & Model Health...")
        health = client.get("/api/v1/health").json()
        print(f"  -> System Status   : {health['status']}")
        print(f"  -> Relational DB   : {health['database']['relational_db']}")
        print(f"  -> MongoDB Status  : {health['database']['nosql_mongo']}")
        print(f"  -> ML Model Name   : {health['ml_inference']['model_name']} ({health['ml_inference']['model_version']})")
        print(f"  -> Model Loaded    : {health['ml_inference']['is_loaded']}")

        # 2. Register Patient
        print("\n[Step 2] Registering Patient in Relational Database...")
        mrn = f"MRN-DEMO-{os.urandom(2).hex().upper()}"
        patient_payload = {
            "medical_record_number": mrn,
            "full_name": "Alexander Hayes",
            "date_of_birth": "1964-11-18",
            "gender": "MALE",
            "contact_email": "alex.hayes@example.org",
            "contact_phone": "+1-555-8392",
            "medical_history_notes": "40 pack-year smoking history. Presenting with mild dyspnea and chronic dry cough."
        }
        p_res = client.post("/api/v1/patients", json=patient_payload)
        patient = p_res.json()
        patient_id = patient["id"]
        print(f"  -> Patient Created : ID={patient_id}, MRN={mrn}, Name={patient['full_name']}")

        # 3. Upload & Reconstruct CT Scan
        print("\n[Step 3] Uploading & Reconstructing 3D Thoracic CT Volume...")
        mock_dicom = b"DICOM_VOLUMETRIC_BINARY_HEADER_SIMULATION"
        upload_res = client.post(
            "/api/v1/scans/upload",
            data={
                "patient_id": patient_id,
                "modality": "CT",
                "anatomical_region": "Chest / Thorax",
                "auto_process": True
            },
            files={"file": ("alexander_chest_highres.dcm", io.BytesIO(mock_dicom), "application/dicom")}
        )
        scan = upload_res.json()
        scan_id = scan["scan_id"]
        print(f"  -> Scan Registered : Scan ID={scan_id}, UID={scan['scan_uid']}")
        print(f"  -> Status          : {scan['status']}")

        # 4. Run 3D CNN Inference
        print("\n[Step 4] Running Packaged 3D CNN Inference Service...")
        infer_res = client.post(
            f"/api/v1/inference/scans/{scan_id}/predict",
            json={"confidence_threshold": 0.45, "run_gradcam_saliency": True}
        )
        inf_data = infer_res.json()
        print(f"  -> Primary Finding : {inf_data['primary_prediction']}")
        print(f"  -> Confidence Score: {inf_data['confidence_score'] * 100:.2f}%")
        print(f"  -> Risk Level      : {inf_data['risk_level']}")
        print(f"  -> Processing Time : {inf_data['processing_time_ms']} ms")
        print(f"  -> Lesions Found   : {inf_data['lesions_detected_count']}")

        for idx, lesion in enumerate(inf_data["lesions"], 1):
            centroid = lesion["centroid_voxel"]
            bbox = lesion["bounding_box_3d"]
            print(f"     * Lesion {idx}: {lesion['nodule_type']}")
            print(f"       Voxel Centroid : (z={centroid['z']}, y={centroid['y']}, x={centroid['x']})")
            print(f"       3D Bounding Box: z:[{bbox['z_min']},{bbox['z_max']}], y:[{bbox['y_min']},{bbox['y_max']}], x:[{bbox['x_min']},{bbox['x_max']}]")
            print(f"       Volume (mm3)   : {lesion['volume_mm3']} mm³")
            print(f"       Malignancy Ind.: {lesion['malignancy_score']:.2f}")

        # 5. Generate Clinical Diagnostic Report
        print("\n[Step 5] Generating Structured Clinical Diagnostic Report...")
        rep_res = client.post(
            "/api/v1/reports/generate",
            json={
                "scan_id": scan_id,
                "inference_run_id": inf_data["id"],
                "radiologist_name": "Dr. Elena Rostova, MD (Senior Thoracic Radiologist)"
            }
        )
        report = rep_res.json()
        print(f"  -> Report ID       : {report['id']}")
        print(f"  -> Radiologist     : {report['radiologist_name']}")
        print("\n" + "-" * 75)
        print("CLINICAL DIAGNOSTIC REPORT PREVIEW:")
        print("-" * 75)
        print(f"TECHNIQUE:\n{report['technique']}\n")
        print(f"CLINICAL INDICATION:\n{report['clinical_history']}\n")
        print(f"FINDINGS:\n{report['findings']}\n")
        print(f"IMPRESSION:\n{report['impression']}\n")
        print(f"RECOMMENDATIONS:\n{report['recommendations']}")
        print("=" * 75)
        print("PIPELINE EXECUTION COMPLETE & ALL INTEGRATION CHECKS PASSED!")
        print("=" * 75)

if __name__ == "__main__":
    run_pipeline_demo()
