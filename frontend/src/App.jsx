import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import UploadView from './components/UploadView';
import Viewer3DView from './components/Viewer3DView';
import ResultsView from './components/ResultsView';
import TrainingView from './components/TrainingView';
import { fetchHealth, fetchScans } from './api';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [currentScan, setCurrentScan] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    // Fetch initial health and latest scan
    async function init() {
      try {
        const h = await fetchHealth();
        setHealth(h);

        const scans = await fetchScans();
        if (scans && scans.items && scans.items.length > 0) {
          setCurrentScan(scans.items[0]);
        }
      } catch (err) {
        console.warn('Init error:', err);
      }
    }
    init();

    // Health poll
    const interval = setInterval(async () => {
      try {
        const h = await fetchHealth();
        setHealth(h);
      } catch {}
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const handleScanSelected = (scan) => {
    setCurrentScan(scan);
    setActiveTab('viewer');
  };

  const handleNavigateToResults = () => {
    setActiveTab('results');
  };

  return (
    <div className="app-container">
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        health={health} 
        currentScan={currentScan}
      />

      <main className="main-content">
        {activeTab === 'upload' && (
          <UploadView 
            onScanSelected={handleScanSelected} 
            currentScan={currentScan}
          />
        )}

        {activeTab === 'viewer' && (
          <Viewer3DView 
            currentScan={currentScan} 
            onNavigateToResults={handleNavigateToResults}
          />
        )}

        {activeTab === 'results' && (
          <ResultsView 
            currentScan={currentScan}
            onNavigateToViewer={() => setActiveTab('viewer')}
          />
        )}

        {activeTab === 'training' && (
          <TrainingView />
        )}
      </main>

      <footer style={{ borderTop: '1px solid var(--border-subtle)', background: '#ffffff', padding: '1.25rem 2rem', textAlign: 'center', color: '#64748b', fontSize: '0.8rem', marginTop: 'auto' }}>
        <div>
          <strong>MedVision 3D</strong> · AI-Powered Volumetric CT Diagnostic System
        </div>
        <div style={{ marginTop: '0.25rem', color: '#94a3b8', fontSize: '0.74rem' }}>
          Clinical Intelligence Platform for Volumetric Medical Imaging Analysis
        </div>
      </footer>
    </div>
  );
}
