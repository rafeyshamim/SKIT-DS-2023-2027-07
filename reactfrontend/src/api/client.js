// Backend contract (FastAPI), used when VITE_USE_MOCK=false:
//   POST /api/scans   multipart: file, patient_id, scan_type, scan_date
//     -> { scan_id, filename, slices?, resolution?, thickness?,
//          mesh?: { vertices: number[][], faces: number[][] }            (single model)
//              or { parts: [{ name, color, solid?, vertices, faces }] }   (several layers) }
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

/* ---------- demo mode: sample lungs + airways (NOT built from your scan) ---------- */

// side: -1 = right lung (shown on the left), +1 = left lung (has the cardiac notch)
function lungPart(side, name, color) {
  const LAT = 90;
  const LON = 120;
  const vertices = [];
  const faces = [];
  const m = -side;
  const sc = side > 0 ? 0.94 : 1;

  for (let i = 0; i <= LAT; i++) {
    const t = (i * Math.PI) / LAT;
    for (let j = 0; j < LON; j++) {
      const p = (j * 2 * Math.PI) / LON;
      const x0 = Math.sin(t) * Math.cos(p);
      const y0 = Math.cos(t);
      const z0 = Math.sin(t) * Math.sin(p);

      const taper = 0.6 + 0.4 * Math.pow((1 - y0) / 2, 0.8);
      const med = x0 * m > 0 ? Math.min(1, x0 * m) : 0;
      const xx = x0 * (x0 * m > 0 ? 0.5 : 1);
      const yy = y0 < -0.7 ? -0.7 + (y0 + 0.7) * 0.6 : y0;

      const notch =
        side > 0 ? 0.5 * Math.exp(-(((y0 + 0.3) / 0.4) ** 2)) * Math.max(0, z0) * med : 0;
      const g1 = Math.exp(-(((y0 + 0.7 * z0 - 0.1) / 0.06) ** 2)) * (1 - med * 0.3);
      const g2 =
        side < 0 ? Math.exp(-(((y0 - 0.38) / 0.05) ** 2)) * Math.max(0, z0) * (1 - med) : 0;
      const bump = 1 + 0.012 * Math.sin(7 * p + 4 * t) + 0.008 * Math.sin(13 * t + 3 * p);
      const k = (1 - 0.07 * g1 - 0.07 * g2) * bump * sc;

      vertices.push([
        side * 1.2 + xx * taper * (1 - notch) * k,
        yy * 2.15 * k,
        z0 * 0.88 * taper * (1 - notch) * k,
      ]);
    }
  }

  for (let a = 0; a < LAT; a++) {
    for (let b = 0; b < LON; b++) {
      const A = a * LON + b;
      const B = a * LON + ((b + 1) % LON);
      const C = (a + 1) * LON + b;
      const D = (a + 1) * LON + ((b + 1) % LON);
      faces.push([A, C, B], [B, C, D]);
    }
  }
  return { name, color, solid: false, vertices, faces };
}

// Adds one tube (airway branch) to the shared vertices/faces lists
function addTube(vertices, faces, pts, radius, sides = 10, segs = 18) {
  const P0 = pts[0];
  const P1 = pts[1];
  const P2 = pts[2];
  const C = P2 ? [2 * P1[0] - (P0[0] + P2[0]) / 2, 2 * P1[1] - (P0[1] + P2[1]) / 2, 2 * P1[2] - (P0[2] + P2[2]) / 2] : null;
  const base = vertices.length;

  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    let pt;
    let tan;
    if (P2) {
      const u = 1 - t;
      pt = [0, 1, 2].map((k) => u * u * P0[k] + 2 * u * t * C[k] + t * t * P2[k]);
      tan = [0, 1, 2].map((k) => 2 * u * (C[k] - P0[k]) + 2 * t * (P2[k] - C[k]));
    } else {
      pt = [0, 1, 2].map((k) => P0[k] + t * (P1[k] - P0[k]));
      tan = [0, 1, 2].map((k) => P1[k] - P0[k]);
    }
    const tl = Math.hypot(...tan) || 1;
    const T = tan.map((v) => v / tl);
    const ref = Math.abs(T[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    let N = [ref[1] * T[2] - ref[2] * T[1], ref[2] * T[0] - ref[0] * T[2], ref[0] * T[1] - ref[1] * T[0]];
    const nl = Math.hypot(...N) || 1;
    N = N.map((v) => v / nl);
    const B = [T[1] * N[2] - T[2] * N[1], T[2] * N[0] - T[0] * N[2], T[0] * N[1] - T[1] * N[0]];

    for (let j = 0; j < sides; j++) {
      const a = (j * 2 * Math.PI) / sides;
      const ca = Math.cos(a) * radius;
      const sa = Math.sin(a) * radius;
      vertices.push([
        pt[0] + ca * N[0] + sa * B[0],
        pt[1] + ca * N[1] + sa * B[1],
        pt[2] + ca * N[2] + sa * B[2],
      ]);
    }
  }

  for (let i = 0; i < segs; i++) {
    for (let j = 0; j < sides; j++) {
      const a = base + i * sides + j;
      const b = base + i * sides + ((j + 1) % sides);
      const c = base + (i + 1) * sides + j;
      const d = base + (i + 1) * sides + ((j + 1) % sides);
      faces.push([a, c, b], [b, c, d]);
    }
  }
}

function airwayPart() {
  const vertices = [];
  const faces = [];
  const tube = (pts, r) => addTube(vertices, faces, pts, r);

  tube([[0, 2.9, 0], [0, 2.2, -0.02], [0, 1.5, -0.05]], 0.17);
  [-1, 1].forEach((s) => {
    tube([[0, 1.5, -0.05], [s * 0.5, 1.15, -0.05], [s * 1, 0.75, 0]], 0.11);
    tube([[s * 1, 0.75, 0], [s * 1.35, 1.15, 0.05], [s * 1.6, 1.6, 0.05]], 0.065);
    tube([[s * 1, 0.75, 0], [s * 1.45, 0.55, 0.1], [s * 1.85, 0.7, 0.15]], 0.06);
    tube([[s * 1, 0.75, 0], [s * 1.3, 0.1, -0.05], [s * 1.6, -0.45, 0]], 0.065);
    tube([[s * 1.3, 0.1, -0.05], [s * 1, -0.5, 0.05], [s * 1.15, -1.2, 0.1]], 0.05);
    tube([[s * 1.6, -0.45, 0], [s * 1.9, -0.9, 0.1]], 0.035);
    tube([[s * 1.6, 1.6, 0.05], [s * 1.4, 1.95, 0.1]], 0.035);
    tube([[s * 1.35, 1.15, 0.05], [s * 1.85, 1.2, 0.1]], 0.035);
  });
  return { name: 'Airways', color: '#f3dcc6', solid: true, vertices, faces };
}

function sampleLungMesh() {
  return {
    parts: [
      lungPart(-1, 'Right lung', '#e0928c'),
      lungPart(1, 'Left lung', '#d98a95'),
      airwayPart(),
    ],
  };
}

async function mockUpload(file, onProgress) {
  for (let p = 0; p <= 100; p += 5) {
    onProgress(p);
    await sleep(70);
  }
  return {
    scan_id: `demo-${hash(file.name).toString(16)}`,
    filename: file.name,
    slices: 120,
    resolution: '512 × 512',
    thickness: '1.25 mm',
    mesh: sampleLungMesh(),
  };
}