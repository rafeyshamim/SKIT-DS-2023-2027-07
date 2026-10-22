import { useRef, useState } from "react";
import { uploadScan } from "../api/client.js";
import { ACCEPTED, formatBytes, validateFile } from "../utils/files.js";

const RECENT = [
  { name: "chest_ct_0412.nii", when: "Yesterday" },
  { name: "P-10388_series.zip", when: "7 Oct" },
  { name: "slice_042.png", when: "5 Oct" },
];

const todayISO = () => new Date().toISOString().slice(0, 10);

export default function UploadPage({ onUploaded }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [patientId, setPatientId] = useState("");
  const [scanType, setScanType] = useState("Chest CT");
  const [scanDate, setScanDate] = useState(todayISO());

  const pick = (f) => {
    setError("");
    if (!f) return;
    const err = validateFile(f);
    if (err) return setError(err);
    setFile(f);
    setProgress(0);
  };

  const clearFile = () => {
    if (uploading) return;
    setFile(null);
    setProgress(0);
    if (inputRef.current) inputRef.current.value = "";
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (!uploading) pick(e.dataTransfer.files?.[0]);
  };

  const startUpload = async () => {
    const err = validateFile(file);
    if (err) return setError(err);
    if (!patientId.trim()) return setError("Enter a patient ID.");
    setError("");
    setUploading(true);
    setProgress(0);
    try {
      const data = await uploadScan(file, { patientId: patientId.trim(), scanType, scanDate }, setProgress);
      setProgress(100);
      onUploaded({
        id: data.scan_id || "demo",
        mesh: data.mesh,
        fileName: file.name,
        patientId: patientId.trim(),
        scanType,
        scanDate,
        slices: data.slices ?? "—",
        resolution: data.resolution ?? "—",
        thickness: data.thickness ?? "—",
        uploadedAt: new Date().toLocaleString("en-GB", {
          day: "2-digit",
          month: "short",
          hour: "numeric",
          minute: "2-digit",
          hour12: true,
        }),
      });
    } catch (e) {
      setProgress(0);
      setError(e.message || "Something went wrong. Try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="grid-upload">
      <section className="card">
        <h2 className="card-title">Upload a CT scan</h2>
        <p className="muted">Add a scan to reconstruct it in 3D.</p>

        <div
          className={`dropzone ${dragging ? "dropzone-on" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <div className="drop-icon">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M7 18a4.5 4.5 0 0 1-.5-8.97A6 6 0 0 1 18 10a4 4 0 0 1 0 8" />
              <path d="M12 12v8M9 15l3-3 3 3" />
            </svg>
          </div>
          <strong>Drag a scan here</strong>
          <span className="muted">or choose a file from your computer</span>
          <button className="btn btn-outline" disabled={uploading} onClick={() => inputRef.current.click()}>
            Browse files
          </button>
          <input
            ref={inputRef}
            type="file"
            hidden
            accept={ACCEPTED.join(",")}
            onChange={(e) => pick(e.target.files?.[0])}
          />
        </div>

        <div className="tags">
          {["DICOM", "NIfTI", "ZIP series", "PNG / JPG"].map((t) => (
            <span className="tag" key={t}>{t}</span>
          ))}
          <span className="muted small">Up to 500 MB</span>
        </div>

        {file && (
          <div className="file-row">
            <div className="file-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z" />
                <path d="M14 3v5h5" />
              </svg>
            </div>
            <div className="file-info">
              <div className="file-name">
                <strong>{file.name}</strong> <span className="muted">· {formatBytes(file.size)}</span>
              </div>
              {(uploading || progress > 0) && (
                <div className="bar"><div className="bar-fill" style={{ width: `${progress}%` }} /></div>
              )}
            </div>
            {uploading || progress > 0 ? (
              <span className="file-pct">{progress}%</span>
            ) : (
              <button className="x-btn" onClick={clearFile} aria-label="Remove file">×</button>
            )}
          </div>
        )}

        {error && <div className="error" role="alert">{error}</div>}

        <div className="form-row">
          <label>
            Patient ID
            <input
              value={patientId}
              maxLength={40}
              disabled={uploading}
              placeholder="e.g. P-10492"
              onChange={(e) => setPatientId(e.target.value)}
            />
          </label>
          <label>
            Scan type
            <select value={scanType} disabled={uploading} onChange={(e) => setScanType(e.target.value)}>
              <option>Chest CT</option>
              <option>Abdomen CT</option>
              <option>Head CT</option>
              <option>Pelvis CT</option>
            </select>
          </label>
          <label>
            Scan date
            <input type="date" value={scanDate} disabled={uploading} onChange={(e) => setScanDate(e.target.value)} />
          </label>
        </div>

        <button className="btn btn-primary btn-wide" onClick={startUpload} disabled={!file || uploading}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />
          </svg>
          {uploading ? "Uploading…" : "Upload scan"}
        </button>
      </section>

      <aside className="side">
        <section className="card">
          <h3 className="card-title">Before you upload</h3>
          <ul className="checks">
            <li>Use the full slice series, not one image</li>
            <li>Remove patient names from the file</li>
            <li>Zip DICOM folders before uploading</li>
          </ul>
        </section>
        <section className="card">
          <h3 className="card-title">Recent uploads</h3>
          <ul className="recent">
            {RECENT.map((r) => (
              <li key={r.name}>
                <span>{r.name}</span>
                <strong>{r.when}</strong>
              </li>
            ))}
          </ul>
        </section>
      </aside>
    </div>
  );
}
