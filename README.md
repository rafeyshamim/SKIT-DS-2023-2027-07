# CT Vision - Frontend (React)

Frontend for **AI-Powered 3D Medical Image Reconstruction and Disease Analysis** (DS-07).
Owner: Naman Verma - Frontend, Visualization & Report Generation.

## Run

```bash
npm install
npm run dev        # opens http://localhost:5173
npm test           # unit tests
npm run build      # production build
```
Requires Node.js 18+.

## Pages
1. **Upload** - drag and drop or browse. DICOM, NIfTI, ZIP series, PNG/JPG, up to 500 MB.
2. **3D view** - empty 3D stage for now. When the backend returns a model, it renders here (rotate, zoom, Front/Side/Top, opacity, wireframe, screenshot).
3. **Results** - empty for now ("No results yet").

## Demo mode vs. real backend
`.env` controls this:
- `VITE_USE_MOCK=true`  -> simulated upload, no backend needed.
- `VITE_USE_MOCK=false` -> uploads to FastAPI at `VITE_API_URL` (default `http://localhost:8000`).

Backend contract (see `src/api/client.js`):
`POST /api/scans` (multipart `file`, `patient_id`, `scan_type`, `scan_date`)
-> `{ scan_id, filename, slices?, resolution?, thickness?, mesh?: { vertices, faces } }`
If `mesh` is returned, the 3D viewer renders it.

## Files
| Area | File |
|---|---|
| Upload page | `src/components/UploadPage.jsx` |
| 3D stage | `src/components/ModelViewer.jsx`, `ViewerPage.jsx` |
| Results (empty) | `src/components/ResultsPage.jsx` |
| API layer | `src/api/client.js` |
| File checks + tests | `src/utils/files.js`, `files.test.js` |
