// Backend contract (FastAPI), used when VITE_USE_MOCK=false:
//   POST /api/scans   multipart: file, patient_id, scan_type, scan_date
//     -> { scan_id, filename, slices?, resolution?, thickness?,
//          mesh?: { vertices: number[][], faces: number[][] } }
// If "mesh" is returned, the 3D viewer renders it. Until then the viewer stays empty.
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hash = (s) => [...s].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7);

export function uploadScan(file, meta, onProgress) {
  if (USE_MOCK) return mockUpload(file, onProgress);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${BASE}/api/scans`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100));
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve(JSON.parse(xhr.responseText))
        : reject(new Error(`Upload failed (server returned ${xhr.status}).`));
    xhr.onerror = () => reject(new Error('Cannot reach the server. Check that the backend is running.'));
    const fd = new FormData();
    fd.append('file', file);
    if (meta.patientId) fd.append('patient_id', meta.patientId);
    fd.append('scan_type', meta.scanType);
    fd.append('scan_date', meta.scanDate);
    xhr.send(fd);
  });
}

/* ---------- demo mode: simulated upload, no model ---------- */
async function mockUpload(file, onProgress) {
  for (let p = 0; p <= 100; p += 5) {
    onProgress(p);
    await sleep(70);
  }
  return { scan_id: `demo-${hash(file.name).toString(16)}`, filename: file.name };
}
