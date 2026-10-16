import React, { useState, useEffect } from 'react';
import { 
  UploadCloud, 
  FileText, 
  CheckCircle2, 
  Clock, 
  Sparkles, 
  ArrowRight, 
  User, 
  AlertTriangle,
  Play
} from 'lucide-react';
import { uploadScanFile, createDemoScan, fetchScans, fetchPatients, createPatient } from '../api';

export default function UploadView({ onScanSelected, currentScan }) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  
  // Patient details state
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [newPatientName, setNewPatientName] = useState('Jane Doe');
  const [newPatientMrn, setNewPatientMrn] = useState(`MRN-${Math.floor(1000 + Math.random() * 9000)}`);
  
  // Scans history list
  const [scansList, setScansList] = useState([]);
  const [loadingScans, setLoadingScans] = useState(false);

  useEffect(() => {
    loadPatients();
    loadScans();
  }, []);

  async function loadPatients() {
    try {
      const data = await fetchPatients();
      if (data && data.items && data.items.length > 0) {
        setPatients(data.items);
        setSelectedPatientId(data.items[0].id);
      }
    } catch (err) {
      console.warn('Could not load patients list:', err);
    }
  }

  async function loadScans() {
    setLoadingScans(true);
    try {
      const data = await fetchScans();
      if (data && data.items) {
        setScansList(data.items);
      }
    } catch (err) {
      console.warn('Could not load scans:', err);
    } finally {
      setLoadingScans(false);
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  async function handleUpload() {
    if (!selectedFile) return;
    setUploading(true);
    setErrorMessage('');
    setUploadProgress(15);
    setStatusMessage('Uploading CT series stream...');

    try {
      let patientId = selectedPatientId;
      if (!patientId) {
        // Create patient first
        const p = await createPatient({
          medical_record_number: newPatientMrn,
          full_name: newPatientName,
          gender: 'FEMALE',
          contact_email: 'patient@clinic.org'
        });
        patientId = p.id;
      }

      setUploadProgress(45);
      setStatusMessage('Streaming file to persistent volume storage...');

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('patient_id', patientId);
      formData.append('modality', 'CT');
      formData.append('anatomical_region', 'Chest / Thorax');
      formData.append('auto_process', 'true');

      const result = await uploadScanFile(formData);
      setUploadProgress(85);
      setStatusMessage('Reconstructing 3D volume & calibrating Hounsfield Units...');

      await new Promise(r => setTimeout(r, 600));
      setUploadProgress(100);
      setStatusMessage('Reconstruction complete!');

      await loadScans();
      onScanSelected(result);
    } catch (err) {
      setErrorMessage(err.message || 'Scan upload failed');
    } finally {
      setUploading(false);
    }
  }

  async function handleQuickDemo() {
    setUploading(true);
    setErrorMessage('');
    setUploadProgress(30);
    setStatusMessage('Synthesizing high-resolution Thoracic CT phantom with injected lesion...');

    try {
      const demoScan = await createDemoScan(
        newPatientName || 'John Anderson',
        `MRN-DEMO-${Math.floor(1000 + Math.random() * 9000)}`
      );

      setUploadProgress(80);
      setStatusMessage('Applying Lung Window (-600 HU / 1500 WW) and resampling to 32x64x64...');
      await new Promise(r => setTimeout(r, 500));

      setUploadProgress(100);
      setStatusMessage('3D Volumetric Scan Ready!');
      await loadScans();
      onScanSelected(demoScan);
    } catch (err) {
      setErrorMessage(err.message || 'Demo generation failed');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="view-header">
        <div className="view-title-group">
          <h1>CT Scan Ingestion & 3D Preprocessing</h1>
          <p>Multi-part DICOM, NIfTI, and slice archives stream ingestion with automatic Hounsfield Unit windowing.</p>
        </div>
      </div>

      {/* Quick Demo Hero Banner */}
      <div className="demo-banner">
        <div className="demo-banner-content">
          <h3>
            <Sparkles size={18} color="#06b6d4" />
            Evaluation Mode — High-Fidelity Thoracic CT Scan
          </h3>
          <p>
            Quick-load a pre-calibrated thoracic CT volume (32 axial slices with pulmonary parenchyma and lung nodule) for instant diagnosis and 3D visualization.
          </p>
        </div>
        <button
          id="btn-quick-demo"
          className="btn btn-primary"
          onClick={handleQuickDemo}
          disabled={uploading}
        >
          <Play size={15} />
          <span>Quick-Load Demo Scan</span>
        </button>
      </div>

      {errorMessage && (
        <div className="glass-panel" style={{ padding: '1rem 1.25rem', marginBottom: '1.5rem', borderColor: 'var(--danger)', color: 'var(--danger)', display: 'flex', gap: '0.75rem', alignItems: 'center', background: 'var(--danger-bg)' }}>
          <AlertTriangle size={18} color="#e11d48" />
          <span>{errorMessage}</span>
        </div>
      )}

      <div className="grid-sidebar">
        {/* Patient Profile Box */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', marginBottom: '1.25rem' }}>
            <User size={18} color="#06b6d4" />
            <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)' }}>Patient Demographics</h3>
          </div>

          {patients.length > 0 && (
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ display: 'block', fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                Select Existing Patient:
              </label>
              <select
                id="select-patient"
                className="form-select"
                value={selectedPatientId}
                onChange={(e) => setSelectedPatientId(e.target.value)}
              >
                {patients.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.full_name} ({p.medical_record_number})
                  </option>
                ))}
              </select>
            </div>
          )}

          <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '1.25rem' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              Or Register New Patient:
            </div>
            
            <div style={{ marginBottom: '0.85rem' }}>
              <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>Full Name</label>
              <input
                id="input-patient-name"
                type="text"
                className="form-input"
                value={newPatientName}
                onChange={(e) => setNewPatientName(e.target.value)}
                placeholder="e.g. Jane Doe"
              />
            </div>

            <div style={{ marginBottom: '0.85rem' }}>
              <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>Medical Record Number (MRN)</label>
              <input
                id="input-patient-mrn"
                type="text"
                className="form-input font-mono"
                value={newPatientMrn}
                onChange={(e) => setNewPatientMrn(e.target.value)}
                placeholder="MRN-2026-XXXX"
              />
            </div>
          </div>
        </div>

        {/* Upload Zone & Scan List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div
            id="dropzone"
            className={`dropzone-card ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input-hidden').click()}
          >
            <input
              type="file"
              id="file-input-hidden"
              style={{ display: 'none' }}
              accept=".dcm,.nii,.gz,.zip,.npy"
              onChange={handleFileChange}
            />

            <div className="dropzone-icon-circle">
              <UploadCloud size={28} />
            </div>

            <h3 style={{ fontSize: '1.15rem', marginBottom: '0.35rem', color: 'var(--text-main)' }}>
              {selectedFile ? selectedFile.name : 'Select or Drag & Drop CT Series'}
            </h3>

            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '440px', margin: '0 auto 1.15rem' }}>
              Accepts DICOM (.dcm), NIfTI (.nii, .nii.gz), multi-slice ZIP archives, or NumPy arrays (.npy).
            </p>

            {selectedFile ? (
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.35rem 0.85rem', background: 'var(--primary-tint)', border: '1px solid var(--primary)', borderRadius: '9999px', color: 'var(--primary-hover)' }}>
                <FileText size={15} />
                <span className="font-mono">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
            ) : (
              <button className="btn btn-secondary" style={{ pointerEvents: 'none' }}>
                Browse Local Files
              </button>
            )}

            {uploading && (
              <div style={{ marginTop: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.4rem', color: 'var(--primary-hover)' }}>
                  <span>{statusMessage}</span>
                  <span className="font-mono">{uploadProgress}%</span>
                </div>
                <div style={{ height: '5px', background: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ width: `${uploadProgress}%`, height: '100%', background: 'var(--primary)', transition: 'width 0.3s ease' }} />
                </div>
              </div>
            )}
          </div>

          {selectedFile && !uploading && (
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button className="btn btn-secondary" onClick={() => setSelectedFile(null)}>
                Clear Selection
              </button>
              <button
                id="btn-upload-submit"
                className="btn btn-primary"
                onClick={handleUpload}
              >
                <span>Upload & Process 3D Volume</span>
                <ArrowRight size={15} />
              </button>
            </div>
          )}

          {/* Scans Repository Table */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)' }}>Processed CT Scans Repository</h3>
              <button className="btn btn-secondary" style={{ fontSize: '0.76rem', padding: '0.3rem 0.65rem' }} onClick={loadScans}>
                Refresh
              </button>
            </div>

            {scansList.length === 0 ? (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                No scans uploaded yet. Use the upload box or Quick-Load Demo Scan above.
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Scan UID</th>
                      <th>Modality / Region</th>
                      <th>Slices</th>
                      <th>Status</th>
                      <th>Created</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scansList.map((scan) => {
                      const isCurrent = currentScan && currentScan.id === scan.id;
                      return (
                        <tr key={scan.id} style={{ background: isCurrent ? 'var(--primary-tint)' : 'transparent' }}>
                          <td className="font-mono" style={{ color: isCurrent ? 'var(--primary-hover)' : 'var(--text-main)', fontWeight: 600 }}>
                            {scan.scan_uid}
                          </td>
                          <td>{scan.modality} · {scan.anatomical_region}</td>
                          <td className="font-mono">{scan.slice_count || 32} slices</td>
                          <td>
                            <span className={`badge ${scan.status === 'PROCESSED' ? 'badge-cyan' : 'badge-moderate'}`}>
                              {scan.status}
                            </span>
                          </td>
                          <td style={{ color: 'var(--text-muted)', fontSize: '0.76rem' }}>
                            {new Date(scan.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </td>
                          <td>
                            <button
                              id={`btn-select-scan-${scan.id}`}
                              className="btn btn-outline-primary"
                              style={{ padding: '0.25rem 0.65rem', fontSize: '0.78rem' }}
                              onClick={() => onScanSelected(scan)}
                            >
                              Open in 3D Viewer →
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
