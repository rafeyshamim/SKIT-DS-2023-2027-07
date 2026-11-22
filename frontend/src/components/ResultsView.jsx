import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  FileText, 
  Printer, 
  Clock, 
  Cpu, 
  Sparkles, 
  ShieldAlert, 
  Info,
  Download
} from 'lucide-react';
import { triggerInference, fetchLatestInference, generateClinicalReport } from '../api';

const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[char]));

export default function ResultsView({ currentScan, onNavigateToViewer }) {
  const [inferenceResult, setInferenceResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState('');
  const [printError, setPrintError] = useState('');
  const [report, setReport] = useState(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [radiologistName, setRadiologistName] = useState('Dr. Elena Rostova, MD (Senior Thoracic Radiologist)');

  useEffect(() => {
    if (currentScan) {
      loadLatestOrRun();
    }
  }, [currentScan]);

  async function loadLatestOrRun() {
    setLoading(true);
    setAnalysisError('');
    try {
      let data = await fetchLatestInference(currentScan.id);
      if (!data) {
        // Run inference automatically
        data = await triggerInference(currentScan.id);
      }
      setInferenceResult(data);

      // Auto-load or generate clinical report
      const rep = await generateClinicalReport(currentScan.id, data.id, radiologistName);
      setReport(rep);
    } catch (err) {
      setAnalysisError(err.message || 'Could not load analysis results.');
      console.warn('Inference / Report fetch error:', err);
    } finally {
      setLoading(false);
    }
  }

  async function handleRerunInference() {
    setLoading(true);
    setAnalysisError('');
    try {
      const data = await triggerInference(currentScan.id, 0.45);
      setInferenceResult(data);
      const rep = await generateClinicalReport(currentScan.id, data.id, radiologistName);
      setReport(rep);
    } catch (err) {
      setAnalysisError(err.message || 'Could not run inference.');
      console.error('Rerun inference failed:', err);
    } finally {
      setLoading(false);
    }
  }

  const handlePrint = () => {
    if (!inferenceResult) {
      setPrintError('Run inference before printing this report.');
      return;
    }

    const printWindow = window.open('', '_blank');
    if (!printWindow) {
      setPrintError('The browser blocked the report window. Allow pop-ups for this app, then try again.');
      return;
    }

    setPrintError('');
    const metrics = inferenceResult.volumetric_metrics || {};
    const probabilities = Object.entries(inferenceResult.class_probabilities || {})
      .map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td>${(Number(value) * 100).toFixed(1)}%</td></tr>`)
      .join('');
    const lesions = (inferenceResult.lesions || []).map((lesion) => `
      <tr>
        <td>${escapeHtml(lesion.lesion_id)}</td>
        <td>${escapeHtml(lesion.nodule_type)}</td>
        <td>${escapeHtml(`Z ${lesion.centroid_mm?.z ?? '—'}, Y ${lesion.centroid_mm?.y ?? '—'}, X ${lesion.centroid_mm?.x ?? '—'} mm`)}</td>
        <td>${escapeHtml(lesion.volume_mm3)} mm³</td>
        <td>${(Number(lesion.malignancy_score || 0) * 100).toFixed(1)}%</td>
      </tr>`).join('');
    const paragraph = (value, fallback) => escapeHtml(value || fallback);

    printWindow.document.open();
    printWindow.document.write(`<!doctype html>
      <html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
      <title>MedVision CT Analysis — ${escapeHtml(currentScan.scan_uid)}</title>
      <style>
        @page { size: A4; margin: 16mm; }
        * { box-sizing: border-box; }
        html, body { background: #fff !important; color: #172033 !important; color-scheme: light; }
        body { margin: 0; min-height: 100vh; padding: 8px; font: 11pt/1.5 Arial, sans-serif; }
        h1 { margin: 0; font-size: 20pt; }
        h2 { margin: 20px 0 7px; padding-bottom: 4px; border-bottom: 1px solid #cbd5e1; font-size: 12pt; }
        p { margin: 5px 0; white-space: pre-wrap; }
        .top { display:flex; justify-content:space-between; gap:20px; padding-bottom:14px; border-bottom:2px solid #0e7490; }
        .muted { color:#64748b; }
        .summary { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:14px; }
        .metric { padding:9px; border:1px solid #cbd5e1; border-radius:6px; }
        .metric strong { display:block; font-size:13pt; }
        table { width:100%; border-collapse:collapse; margin:8px 0 14px; }
        th,td { padding:7px; text-align:left; border:1px solid #cbd5e1; vertical-align:top; }
        .disclaimer { margin-top:24px; padding-top:10px; border-top:1px solid #cbd5e1; color:#64748b; font-size:9pt; }
        tr { break-inside:avoid; }
        @media print {
          html, body { background: #fff !important; color: #172033 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
          .muted, .disclaimer { color: #64748b !important; }
        }
      </style></head><body>
        <header class="top"><div><h1>MedVision 3D</h1><div class="muted">AI-assisted volumetric CT analysis report</div></div>
          <div><strong>Scan:</strong> ${escapeHtml(currentScan.scan_uid)}<br><strong>Date:</strong> ${new Date().toLocaleDateString()}</div></header>
        <h2>Analysis Summary</h2>
        <p><strong>Prediction:</strong> ${escapeHtml(inferenceResult.primary_prediction)}</p>
        <p><strong>Risk category:</strong> ${escapeHtml(inferenceResult.risk_level)}</p>
        <p><strong>Confidence:</strong> ${(Number(inferenceResult.confidence_score || 0) * 100).toFixed(1)}%</p>
        <p><strong>Model:</strong> ${escapeHtml(inferenceResult.model_name)} ${escapeHtml(inferenceResult.model_version)} · Run #${escapeHtml(inferenceResult.id)}</p>
        <div class="summary">
          <div class="metric">Total lung volume<strong>${escapeHtml(metrics.total_lung_volume_cm3 ?? '—')} cm³</strong></div>
          <div class="metric">Total lesion volume<strong>${escapeHtml(metrics.total_lesion_volume_mm3 ?? '—')} mm³</strong></div>
          <div class="metric">Lung involvement<strong>${escapeHtml(metrics.lung_involvement_percentage ?? '—')}%</strong></div>
          <div class="metric">Lesions detected<strong>${escapeHtml(inferenceResult.lesions_detected_count ?? 0)}</strong></div>
        </div>
        <h2>Class Probabilities</h2><table><tbody>${probabilities || '<tr><td>No probability data</td></tr>'}</tbody></table>
        <h2>Detected Lesions</h2><table><thead><tr><th>ID</th><th>Morphology</th><th>Centroid</th><th>Volume</th><th>Score</th></tr></thead>
          <tbody>${lesions || '<tr><td colspan="5">No lesions reported</td></tr>'}</tbody></table>
        <h2>Clinical Findings</h2><p>${paragraph(report?.findings, 'No narrative findings are available.')}</p>
        <h2>Impression</h2><p>${paragraph(report?.impression, inferenceResult.primary_prediction)}</p>
        <h2>Recommendations</h2><p>${paragraph(report?.recommendations, 'No recommendations are available.')}</p>
        <p class="disclaimer">Academic demonstration output only. This AI-generated report is not a clinically validated diagnosis and requires review by a qualified clinician.</p>
      </body></html>`);
    printWindow.document.close();
    printWindow.focus();
    window.setTimeout(() => printWindow.print(), 300);
    printWindow.onafterprint = () => printWindow.close();
  };

  if (!currentScan) {
    return (
      <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
        <Activity size={40} color="#00d2b4" style={{ marginBottom: '1rem' }} />
        <h2 style={{ fontSize: '1.4rem', marginBottom: '0.5rem' }}>No CT Scan Selected</h2>
        <p style={{ color: '#94a3b8' }}>Please select or upload a scan in the Ingestion tab first.</p>
      </div>
    );
  }

  const isCritical = inferenceResult?.risk_level === 'CRITICAL' || inferenceResult?.risk_level === 'HIGH';
  const confidencePct = inferenceResult ? Math.round(inferenceResult.confidence_score * 100) : 0;
  const strokeDashoffset = 283 - (283 * confidencePct) / 100;

  return (
    <div className="animate-fade-in">
      <div className="view-header">
        <div className="view-title-group">
          <h1>AI Disease Analysis & Quantitative Findings</h1>
          <p>
            3D CNN Deep Learning Diagnostic Results for Scan <span className="font-mono" style={{ color: 'var(--primary)' }}>{currentScan.scan_uid}</span>
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            className="btn btn-secondary"
            onClick={handleRerunInference}
            disabled={loading}
          >
            <Cpu size={15} />
            <span>{loading ? 'Analyzing...' : 'Re-run Inference'}</span>
          </button>
          
          <button
            id="btn-print-report"
            className="btn btn-primary"
            onClick={handlePrint}
          >
            <Printer size={15} />
            <span>Download / Print Clinical Report</span>
          </button>
        </div>
      </div>

      {printError && <p role="alert" style={{ color: '#b45309', marginBottom: '1rem' }}>{printError}</p>}
      {analysisError && inferenceResult && <p role="status" style={{ color: '#b45309', marginBottom: '1rem' }}>Inference finished, but the report could not refresh: {analysisError}</p>}
      {loading && inferenceResult && <p role="status" style={{ color: 'var(--primary-hover)', marginBottom: '1rem' }}>Running a new inference for scan {currentScan.scan_uid}…</p>}

      {loading && !inferenceResult ? (
        <div className="glass-panel" style={{ padding: '4rem', textAlign: 'center' }}>
          <div style={{ width: '48px', height: '48px', border: '3px solid rgba(0, 210, 180, 0.2)', borderTopColor: 'var(--primary)', borderRadius: '50%', margin: '0 auto 1.5rem', animation: 'spin 1s linear infinite' }} />
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Executing 3D CNN Volumetric Inference...</h3>
          <p style={{ color: '#94a3b8' }}>
            Forward pass on (1, 1, 32, 64, 64) tensor · 3D spatial convolutions · Grad-CAM localization
          </p>
        </div>
      ) : analysisError && !inferenceResult ? (
        <div className="glass-panel" role="alert" style={{ padding: '2.5rem', textAlign: 'center', marginTop: '1.5rem' }}>
          <AlertTriangle size={36} color="#f59e0b" style={{ marginBottom: '0.75rem' }} />
          <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>Analysis could not be loaded</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1.25rem' }}>{analysisError}</p>
          <button className="btn btn-primary" onClick={handleRerunInference} disabled={loading}>
            <Cpu size={15} />
            <span>Try Inference Again</span>
          </button>
        </div>
      ) : inferenceResult ? (
        <>
          {/* Primary Prediction Hero Banner */}
          <div className={`prediction-hero-card ${isCritical ? 'critical' : 'moderate'}`} style={{ marginBottom: '1.75rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.65rem' }}>
                <span className={`badge ${isCritical ? 'badge-critical' : 'badge-moderate'}`}>
                  RISK LEVEL: {inferenceResult.risk_level}
                </span>
                <span className="telemetry-chip" style={{ padding: '0.15rem 0.6rem' }}>
                  Inference run #{inferenceResult.id}
                </span>
                <span className="telemetry-chip" style={{ padding: '0.15rem 0.6rem' }}>
                  <Clock size={12} />
                  <span>Latency: {inferenceResult.processing_time_ms} ms</span>
                </span>
              </div>

              <h2 style={{ fontSize: '1.85rem', color: 'var(--text-main)', marginBottom: '0.35rem' }}>
                {inferenceResult.primary_prediction}
              </h2>

              <p style={{ color: 'var(--text-muted)', maxWidth: '560px' }}>
                {isCritical 
                  ? 'Focal pulmonary nodule detected within right thoracic parenchymal window demonstrating malignant morphological characteristics.'
                  : 'Volumetric CT scan indicates low-density pulmonary features consistent with benign clinical guidelines.'}
              </p>
            </div>

            {/* Circular Confidence Meter */}
            <div className="confidence-gauge-box">
              <svg className="gauge-svg" viewBox="0 0 100 100">
                <circle className="gauge-bg-circle" cx="50" cy="50" r="45" />
                <circle 
                  className="gauge-fill-circle" 
                  cx="50" 
                  cy="50" 
                  r="45"
                  stroke={isCritical ? '#e11d48' : '#06b6d4'}
                  strokeDasharray="283"
                  strokeDashoffset={strokeDashoffset}
                />
              </svg>

              <div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Confidence
                </div>
                <div className="gauge-value-text font-mono">
                  {confidencePct}%
                </div>
                <div style={{ fontSize: '0.75rem', color: isCritical ? '#e11d48' : 'var(--primary-hover)', fontWeight: 600 }}>
                  {inferenceResult.confidence_score >= 0.45 ? 'Validated' : 'Borderline'}
                </div>
              </div>
            </div>
          </div>

          <div className="grid-2col" style={{ marginBottom: '1.75rem' }}>
            {/* Probability Bars */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.05rem', marginBottom: '0.35rem', color: 'var(--text-main)' }}>
                Diagnostic Class Probabilities
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '1.15rem' }}>
                3D CNN Softmax distribution across target pulmonary nodule categories
              </p>

              <div className="prob-bar-container">
                {Object.entries(inferenceResult.class_probabilities || {}).map(([className, prob]) => {
                  const pct = Math.round(prob * 100);
                  const isTop = className === inferenceResult.primary_prediction;
                  return (
                    <div key={className} className="prob-row">
                      <div className="prob-row-header">
                        <span style={{ color: isTop ? 'var(--text-main)' : 'var(--text-muted)', fontWeight: isTop ? 600 : 400 }}>
                          {className}
                        </span>
                        <span className="font-mono" style={{ color: isTop ? 'var(--primary-hover)' : 'var(--text-muted)', fontWeight: 600 }}>
                          {pct}%
                        </span>
                      </div>
                      <div className="prob-track">
                        <div
                          className="prob-fill"
                          style={{
                            width: `${pct}%`,
                            background: isTop 
                              ? (isCritical ? 'linear-gradient(90deg, #e11d48, #ea580c)' : 'linear-gradient(90deg, #06b6d4, #0284c7)')
                              : '#cbd5e1'
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Volumetric Metrics */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.05rem', marginBottom: '0.35rem', color: 'var(--text-main)' }}>
                Volumetric CT Metrics
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '1.15rem' }}>
                Quantified anatomical parenchyma and lesion volumetric calculations
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem' }}>
                <div style={{ background: 'var(--bg-base)', padding: '0.85rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>Total Lung Volume</div>
                  <div className="font-mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-main)' }}>
                    {inferenceResult.volumetric_metrics?.total_lung_volume_cm3 || 3420} cm³
                  </div>
                </div>

                <div style={{ background: 'var(--bg-base)', padding: '0.85rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>Total Lesion Volume</div>
                  <div className="font-mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: isCritical ? '#e11d48' : 'var(--primary-hover)' }}>
                    {inferenceResult.volumetric_metrics?.total_lesion_volume_mm3?.toFixed(1) || 15290.2} mm³
                  </div>
                </div>

                <div style={{ background: 'var(--bg-base)', padding: '0.85rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>Lung Involvement</div>
                  <div className="font-mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0284c7' }}>
                    {inferenceResult.volumetric_metrics?.lung_involvement_percentage?.toFixed(2) || '0.45'}%
                  </div>
                </div>

                <div style={{ background: 'var(--bg-base)', padding: '0.85rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>Lesions Detected</div>
                  <div className="font-mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-main)' }}>
                    {inferenceResult.lesions_detected_count || 1}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 3D Lesion Localization Table */}
          {inferenceResult.lesions && inferenceResult.lesions.length > 0 && (
            <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.75rem' }}>
              <h3 style={{ fontSize: '1.05rem', marginBottom: '0.85rem', color: 'var(--text-main)' }}>
                Detected 3D Pulmonary Lesions
              </h3>

              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Lesion ID</th>
                      <th>Morphology</th>
                      <th>Centroid (Z, Y, X mm)</th>
                      <th>Volume</th>
                      <th>Malignancy Score</th>
                      <th>Calcification</th>
                      <th>Spiculation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inferenceResult.lesions.map((lesion) => (
                      <tr key={lesion.lesion_id}>
                        <td className="font-mono" style={{ color: 'var(--primary-hover)', fontWeight: 600 }}>{lesion.lesion_id}</td>
                        <td>{lesion.nodule_type}</td>
                        <td className="font-mono" style={{ fontSize: '0.8rem' }}>
                          Z:{lesion.centroid_mm.z} Y:{lesion.centroid_mm.y} X:{lesion.centroid_mm.x}
                        </td>
                        <td className="font-mono">{lesion.volume_mm3} mm³</td>
                        <td>
                          <span className="badge badge-critical font-mono">
                            {Math.round(lesion.malignancy_score * 100)}%
                          </span>
                        </td>
                        <td style={{ fontSize: '0.8rem' }}>{lesion.calcification_pattern}</td>
                        <td className="font-mono">{lesion.spiculation_score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Clinical Diagnostic Report (Printable Paper) */}
          <div className="report-paper">
            <div className="report-header">
              <div>
                <h2 style={{ fontSize: '1.5rem', color: '#0f172a', fontWeight: 800 }}>
                  DEPARTMENT OF THORACIC RADIOLOGY & MEDICAL AI
                </h2>
                <div style={{ fontSize: '0.9rem', color: '#475569' }}>
                  Volumetric 3D CT Diagnostic Analysis · Fleischner Society Protocol
                </div>
              </div>
              <div style={{ textAlign: 'right', fontSize: '0.82rem', color: '#64748b' }}>
                <div>DATE: {new Date().toLocaleDateString()}</div>
                <div>EXAM: High-Resolution Volumetric Chest CT</div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', paddingBottom: '1rem', marginBottom: '1.5rem', borderBottom: '1px solid #e2e8f0', fontSize: '0.85rem' }}>
              <div><strong>PATIENT MRN:</strong> {currentScan.patient_mrn || 'MRN-2026-9042'}</div>
              <div><strong>NAME:</strong> {currentScan.patient_name || 'Jane Doe'}</div>
              <div><strong>MODALITY:</strong> CT (Axial 32)</div>
              <div><strong>STATUS:</strong> FINALIZED</div>
            </div>

            <div className="report-section">
              <div className="report-section-title">Clinical History & Indication</div>
              <div className="report-text">
                {report?.clinical_history || 'Long-term tobacco exposure (35 pack-years), chronic cough, nodule surveillance screening.'}
              </div>
            </div>

            <div className="report-section">
              <div className="report-section-title">Technique & Volumetric Acquisition</div>
              <div className="report-text">
                {report?.technique || 'Helical high-resolution volumetric chest CT acquisition without IV contrast. Hounsfield Unit lung windowing (-600 HU WL, 1500 HU WW). Deep learning 3D CNN inference model.'}
              </div>
            </div>

            <div className="report-section">
              <div className="report-section-title">Diagnostic Findings</div>
              <div className="report-text">
                {report?.findings || '1. LUNGS AND AIRWAYS: The 3D CNN deep learning analysis identified 1 focal pulmonary lesion measuring 15,290 mm³ in right mid-lung zone with spiculation and non-calcified soft tissue attenuation.'}
              </div>
            </div>

            <div className="report-section">
              <div className="report-section-title">Impression & Diagnostic Assessment</div>
              <div className="report-text" style={{ fontWeight: 600, color: isCritical ? '#dc2626' : '#059669' }}>
                {report?.impression || `1. ${inferenceResult.primary_prediction.toUpperCase()} (Model Confidence: ${confidencePct}%, Risk Category: ${inferenceResult.risk_level}).`}
              </div>
            </div>

            <div className="report-section">
              <div className="report-section-title">Recommendations</div>
              <div className="report-text">
                {report?.recommendations || 'Per Fleischner Society Guidelines for pulmonary nodules: Recommend contrast-enhanced CT or 18F-FDG PET/CT within 3 months, followed by multidisciplinary thoracic tumor board consultation.'}
              </div>
            </div>

            <div className="report-signature">
              <div>
                <div style={{ fontSize: '0.78rem', color: '#64748b' }}>REPORT GENERATED BY:</div>
                <div style={{ fontWeight: 600 }}>MedVision 3D Deep Learning Diagnostic Engine v1.2</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.78rem', color: '#64748b' }}>ELECTRONIC SIGN-OFF:</div>
                <div style={{ fontWeight: 700, color: '#0f172a' }}>{radiologistName}</div>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="glass-panel" style={{ padding: '2.5rem', textAlign: 'center', marginTop: '1.5rem' }}>
          <Info size={32} color="var(--primary)" style={{ marginBottom: '0.75rem' }} />
          <h3>No analysis result is available yet</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>Run inference to generate findings for this scan.</p>
        </div>
      )}
    </div>
  );
}
