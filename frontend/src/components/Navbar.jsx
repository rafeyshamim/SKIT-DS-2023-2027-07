import React from 'react';
import { 
  Box, 
  Layers, 
  Activity, 
  Cpu, 
  Database
} from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, health, currentScan }) {
  const isHealthy = health && health.status === 'ONLINE';

  return (
    <>
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo-cube">
            <Box size={20} color="#06b6d4" />
          </div>
          <div>
            <div className="brand-title">
              MedVision 3D
              <span className="brand-tag">Clinical AI</span>
            </div>
            <div style={{ fontSize: '0.74rem', color: '#64748b' }}>
              Volumetric CT Diagnostic Intelligence
            </div>
          </div>
        </div>

        <div className="header-telemetry">
          {currentScan && (
            <div className="telemetry-chip" style={{ borderColor: 'rgba(6, 182, 212, 0.4)' }}>
              <span style={{ color: '#0891b2', fontWeight: 600 }}>Active Scan:</span>
              <span className="font-mono">{currentScan.scan_uid}</span>
              <span className="badge badge-cyan" style={{ fontSize: '0.65rem', padding: '0.1rem 0.4rem' }}>
                {currentScan.slice_count || 32} SLICES
              </span>
            </div>
          )}

          <div className="telemetry-chip">
            <div className={`telemetry-dot ${isHealthy ? 'online pulse' : ''}`} />
            <span>FastAPI:</span>
            <span style={{ color: isHealthy ? '#0891b2' : '#e11d48', fontWeight: 600 }}>
              {isHealthy ? 'ONLINE' : 'CONNECTING...'}
            </span>
          </div>

          <div className="telemetry-chip">
            <Cpu size={13} color="#06b6d4" />
            <span>Model:</span>
            <span style={{ color: '#0891b2', fontWeight: 600 }}>
              {health?.ml_inference?.model_name || 'MedNet-3D'}
            </span>
          </div>

          <div className="telemetry-chip">
            <Database size={13} color="#64748b" />
            <span>Storage:</span>
            <span style={{ color: '#475569', fontWeight: 600 }}>
              Connected
            </span>
          </div>
        </div>
      </header>

      <nav className="nav-tabs-wrapper" aria-label="Main Navigation">
        <button
          id="nav-tab-upload"
          className={`nav-tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          <span className="tab-badge-num">1</span>
          <Box size={15} />
          <span>CT Ingestion</span>
        </button>

        <button
          id="nav-tab-viewer"
          className={`nav-tab-btn ${activeTab === 'viewer' ? 'active' : ''}`}
          onClick={() => setActiveTab('viewer')}
        >
          <span className="tab-badge-num">2</span>
          <Layers size={15} />
          <span>3D Reconstruction & Scrubber</span>
          {currentScan && (
            <span className="badge badge-cyan" style={{ fontSize: '0.65rem', padding: '0.05rem 0.35rem' }}>
              READY
            </span>
          )}
        </button>

        <button
          id="nav-tab-results"
          className={`nav-tab-btn ${activeTab === 'results' ? 'active' : ''}`}
          onClick={() => setActiveTab('results')}
        >
          <span className="tab-badge-num">3</span>
          <Activity size={15} />
          <span>Disease Analysis & Report</span>
        </button>

        <button
          id="nav-tab-training"
          className={`nav-tab-btn ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          <span className="tab-badge-num">4</span>
          <Cpu size={15} />
          <span>Model Training & Datasets</span>
          <span className="badge badge-cyan" style={{ fontSize: '0.65rem', padding: '0.05rem 0.35rem' }}>
            NoduleMNIST3D
          </span>
        </button>
      </nav>
    </>
  );
}
