const API_BASE = '/api/v1';

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Health fetch error:', err);
    return null;
  }
}

export async function fetchPatients() {
  const res = await fetch(`${API_BASE}/patients`);
  if (!res.ok) throw new Error('Failed to fetch patients');
  return await res.json();
}

export async function createPatient(data) {
  const res = await fetch(`${API_BASE}/patients`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to create patient');
  return await res.json();
}

export async function fetchScans(patientId = null) {
  const url = patientId ? `${API_BASE}/scans?patient_id=${patientId}` : `${API_BASE}/scans`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch scans');
  return await res.json();
}

export async function fetchScanDetails(scanId) {
  const res = await fetch(`${API_BASE}/scans/${scanId}`);
  if (!res.ok) throw new Error(`Failed to fetch scan ${scanId}`);
  return await res.json();
}

export async function uploadScanFile(formData) {
  const res = await fetch(`${API_BASE}/scans/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Upload failed');
  }
  return await res.json();
}

export async function createDemoScan(patientName = 'John Anderson', patientMrn = 'MRN-2026-DEMO') {
  const formData = new FormData();
  formData.append('modality', 'CT');
  formData.append('anatomical_region', 'Chest / Thorax');
  formData.append('patient_name', patientName);
  formData.append('patient_mrn', patientMrn);

  const res = await fetch(`${API_BASE}/scans/sample`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Failed to generate demo scan');
  return await res.json();
}

export async function fetchScanSlice(scanId, sliceIdx = 16, plane = 'axial') {
  const res = await fetch(`${API_BASE}/scans/${scanId}/slices/${sliceIdx}?plane=${plane}`);
  if (!res.ok) throw new Error('Failed to fetch scan slice');
  return await res.json();
}

export async function fetchScanMesh3d(scanId, isoThreshold = 0.25, maxPoints = 2500) {
  const res = await fetch(`${API_BASE}/scans/${scanId}/mesh3d?iso_threshold=${isoThreshold}&max_points=${maxPoints}`);
  if (!res.ok) throw new Error('Failed to fetch 3D mesh');
  return await res.json();
}

export async function triggerInference(scanId, confidenceThreshold = 0.45) {
  const res = await fetch(`${API_BASE}/inference/scans/${scanId}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      confidence_threshold: confidenceThreshold,
      run_gradcam_saliency: true
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Inference failed');
  }
  return await res.json();
}

export async function fetchLatestInference(scanId) {
  const res = await fetch(`${API_BASE}/inference/scans/${scanId}/latest`);
  if (!res.ok) return null;
  return await res.json();
}

export async function fetchSliceHeatmap(scanId, sliceIdx = 16) {
  const res = await fetch(`${API_BASE}/inference/scans/${scanId}/slice-heatmap/${sliceIdx}`);
  if (!res.ok) return null;
  return await res.json();
}

export async function generateClinicalReport(scanId, runId, doctorName = 'Dr. Elena Rostova, MD') {
  const res = await fetch(`${API_BASE}/reports/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scan_id: scanId,
      inference_run_id: runId,
      radiologist_name: doctorName
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to generate report');
  }
  return await res.json();
}

export async function fetchTrainingStatus() {
  const res = await fetch(`${API_BASE}/training/status`);
  if (!res.ok) throw new Error('Failed to fetch training status');
  return await res.json();
}

export async function triggerTraining(epochs = 5) {
  const res = await fetch(`${API_BASE}/training/start?epochs=${epochs}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to start training');
  return await res.json();
}

export function getIsosurfaceHtmlUrl(scanId) {
  return `${API_BASE}/scans/${scanId}/isosurface-html`;
}

export function getMipImageUrl(scanId) {
  return `${API_BASE}/scans/${scanId}/mip`;
}

