# MedVision 3D CT Diagnostic Backend - REST API Contract Specification

**Base URL**: `http://localhost:8000/api/v1`  
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc` (ReDoc)

---

## 1. System Health & Diagnostics

### `GET /health`
Returns the runtime health of all dependent services and ML model readiness.

**Response `200 OK`**:
```json
{
  "status": "ONLINE",
  "service": "MedVision 3D CT Backend",
  "environment": "production",
  "database": {
    "relational_db": "HEALTHY",
    "nosql_mongo": "CONNECTED"
  },
  "ml_inference": {
    "model_name": "MedNet-3D-CNN",
    "model_version": "v1.2.0",
    "is_loaded": true,
    "target_tensor_shape": [1, 32, 64, 64]
  }
}
```

---

## 2. Patient Records Management

### `POST /patients`
Registers a new patient record with a unique Medical Record Number (MRN).

**Request Body**:
```json
{
  "medical_record_number": "MRN-2026-9042",
  "full_name": "Jane Doe",
  "date_of_birth": "1968-04-12",
  "gender": "FEMALE",
  "contact_email": "jane.doe@example.com",
  "contact_phone": "+1-555-0199",
  "medical_history_notes": "40 pack-year smoking history, persistent cough."
}
```

**Response `201 Created`**:
```json
{
  "id": 1,
  "medical_record_number": "MRN-2026-9042",
  "full_name": "Jane Doe",
  "date_of_birth": "1968-04-12",
  "gender": "FEMALE",
  "contact_email": "jane.doe@example.com",
  "contact_phone": "+1-555-0199",
  "medical_history_notes": "40 pack-year smoking history, persistent cough.",
  "created_at": "2026-09-26T10:45:00.000Z",
  "updated_at": "2026-09-26T10:45:00.000Z"
}
```

### `GET /patients`
Lists patients with pagination and optional search filter.

**Query Parameters**:
- `page` (integer, default 1)
- `limit` (integer, default 20)
- `search` (string, optional: filter by name or MRN)

---

## 3. CT Scan Upload & 3D Reconstruction

### `POST /scans/upload`
Uploads a CT scan in multi-part format (`.dcm`, `.zip` slice archive, `.nii`, or `.npy`).

**Request (Multipart Form-Data)**:
- `file`: Binary file stream
- `patient_id`: `1` (integer)
- `modality`: `"CT"` (string, default: `"CT"`)
- `anatomical_region`: `"Chest / Thorax"` (string)
- `auto_process`: `true` (boolean, default: `true`)

**Response `201 Created`**:
```json
{
  "scan_id": 1,
  "scan_uid": "SCAN_A1091C9F24D8",
  "patient_id": 1,
  "modality": "CT",
  "anatomical_region": "Chest / Thorax",
  "original_filename": "chest_scan_series.zip",
  "file_size_bytes": 14589200,
  "status": "PROCESSING",
  "message": "Scan uploaded successfully. 3D reconstruction pipeline initiated.",
  "created_at": "2026-09-26T10:46:00.000Z"
}
```

### `POST /scans/{scan_id}/process`
Triggers or re-runs 3D volumetric reconstruction and Hounsfield Unit calibration.

### `GET /scans/{scan_id}`
Returns scan status along with DICOM acquisition parameters stored in MongoDB.

---

## 4. 3D CNN Model Inference

### `POST /inference/scans/{scan_id}/predict`
Executes volumetric deep learning inference on the 3D CT scan.

**Request Body**:
```json
{
  "confidence_threshold": 0.45,
  "run_gradcam_saliency": true
}
```

**Response `200 OK`**:
```json
{
  "id": 1,
  "scan_id": 1,
  "patient_id": 1,
  "model_name": "MedNet-3D-CNN",
  "model_version": "v1.2.0",
  "status": "COMPLETED",
  "primary_prediction": "Malignant Suspicion",
  "confidence_score": 0.8978,
  "risk_level": "CRITICAL",
  "processing_time_ms": 138.4,
  "class_probabilities": {
    "Normal / No Nodule": 0.0215,
    "Benign Pulmonary Nodule": 0.0807,
    "Malignant Suspicion": 0.8978
  },
  "lesions_detected_count": 1,
  "lesions": [
    {
      "lesion_id": "LESION-15_33_31",
      "nodule_type": "Spiculated Subpleural Pulmonary Nodule",
      "centroid_voxel": { "z": 15.5, "y": 33.1, "x": 31.8 },
      "centroid_mm": { "z": 19.4, "y": 23.3, "x": 22.4 },
      "bounding_box_3d": {
        "z_min": 0, "z_max": 31,
        "y_min": 5, "y_max": 58,
        "x_min": 4, "x_max": 60
      },
      "volume_mm3": 15290.21,
      "malignancy_score": 0.92,
      "calcification_pattern": "Non-calcified (Soft Tissue Attenuation)",
      "spiculation_score": 0.88,
      "lobulation_score": 0.74
    }
  ],
  "slice_abnormality_scores": [0.0, 0.05, 0.12, 0.45, 0.89, 0.92, ...],
  "volumetric_metrics": {
    "total_lung_volume_cm3": 3420.0,
    "total_lesion_volume_mm3": 15290.21,
    "lung_involvement_percentage": 0.4471
  },
  "started_at": "2026-09-26T10:46:10.000Z",
  "completed_at": "2026-09-26T10:46:10.140Z"
}
```

### `GET /inference/scans/{scan_id}/slice-heatmap/{slice_idx}`
Returns normalized 2D attention grid and lesion coordinates for an individual axial slice.

---

## 5. Clinical Diagnostic Reports

### `POST /reports/generate`
Generates a structured clinical report adhering to Fleischner Society guidelines.

**Request Body**:
```json
{
  "scan_id": 1,
  "inference_run_id": 1,
  "radiologist_name": "Dr. Elena Rostova, MD (Senior Thoracic Radiologist)"
}
```

**Response `201 Created`**:
```json
{
  "id": 1,
  "scan_id": 1,
  "patient_id": 1,
  "inference_run_id": 1,
  "radiologist_name": "Dr. Elena Rostova, MD (Senior Thoracic Radiologist)",
  "clinical_history": "40 pack-year smoking history, persistent cough.",
  "technique": "Helical high-resolution volumetric chest CT acquisition without IV contrast.",
  "findings": "1. LUNGS AND AIRWAYS: The 3D CNN deep learning analysis identified 1 focal pulmonary lesion...",
  "impression": "1. MALIGNANT SUSPICION (Model Confidence: 89.8%, Risk Category: CRITICAL)...",
  "recommendations": "Per Fleischner Society Guidelines for pulmonary nodules: Recommend contrast-enhanced CT or 18F-FDG PET/CT...",
  "status": "FINALIZED",
  "created_at": "2026-09-26T10:47:00.000Z",
  "updated_at": "2026-09-26T10:47:00.000Z"
}
```

### `PUT /reports/{report_id}`
Updates, amends, or adds final radiologist sign-off to a report.
