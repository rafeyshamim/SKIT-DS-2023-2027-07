import React, { useState, useEffect, useRef } from 'react';
import { 
  Layers, 
  RotateCw, 
  ZoomIn, 
  ZoomOut, 
  Play, 
  Pause, 
  Eye, 
  Sparkles, 
  ArrowRight,
  Maximize2,
  Crosshair,
  Sliders,
  Compass,
  Box,
  ExternalLink
} from 'lucide-react';
import { 
  fetchScanSlice, 
  fetchScanMesh3d, 
  fetchSliceHeatmap,
  getIsosurfaceHtmlUrl,
  getMipImageUrl
} from '../api';

export default function Viewer3DView({ currentScan, onNavigateToResults }) {
  // 2D slice state
  const [sliceIndex, setSliceIndex] = useState(16);
  const [totalSlices, setTotalSlices] = useState(32);
  const [plane, setPlane] = useState('axial'); // axial, coronal, sagittal
  const [isPlaying, setIsPlaying] = useState(false);
  const [windowPreset, setWindowPreset] = useState('lung'); // lung, bone, soft
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.45);
  const [sliceData, setSliceData] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [loadingSlice, setLoadingSlice] = useState(false);

  // 3D Canvas state
  const [view3DMode, setView3DMode] = useState('cloud'); // 'cloud', 'surface', 'mip'
  const [meshPoints, setMeshPoints] = useState([]);
  const [isoThreshold, setIsoThreshold] = useState(0.28);
  const [rotX, setRotX] = useState(25);
  const [rotY, setRotY] = useState(45);
  const [zoom3D, setZoom3D] = useState(1.0);
  const [isDragging3D, setIsDragging3D] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [showBoundingBox, setShowBoundingBox] = useState(true);

  const canvas2DRef = useRef(null);
  const canvas3DRef = useRef(null);

  useEffect(() => {
    if (currentScan) {
      loadSlice(sliceIndex, plane);
      loadMesh(isoThreshold);
    }
  }, [currentScan, sliceIndex, plane]);

  useEffect(() => {
    if (currentScan && showHeatmap) {
      loadHeatmap(sliceIndex);
    }
  }, [currentScan, sliceIndex, showHeatmap]);

  // Autoplay scrubber carousel
  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setSliceIndex((prev) => (prev + 1) % totalSlices);
      }, 140);
    }
    return () => clearInterval(interval);
  }, [isPlaying, totalSlices]);

  async function loadSlice(idx, currentPlane) {
    if (!currentScan) return;
    setLoadingSlice(true);
    try {
      const data = await fetchScanSlice(currentScan.id, idx, currentPlane);
      setSliceData(data);
      setTotalSlices(data.total_slices);
    } catch (err) {
      console.warn('Error loading slice:', err);
    } finally {
      setLoadingSlice(false);
    }
  }

  async function loadHeatmap(idx) {
    if (!currentScan) return;
    try {
      const data = await fetchSliceHeatmap(currentScan.id, idx);
      setHeatmapData(data);
    } catch (err) {
      console.warn('Error loading heatmap:', err);
    }
  }

  async function loadMesh(thresh) {
    if (!currentScan) return;
    try {
      const data = await fetchScanMesh3d(currentScan.id, thresh, 2200);
      if (data && data.points) {
        setMeshPoints(data.points);
      }
    } catch (err) {
      console.warn('Error loading 3D mesh points:', err);
    }
  }

  // Draw 2D CT Slice on HTML5 Canvas
  useEffect(() => {
    const canvas = canvas2DRef.current;
    if (!canvas || !sliceData || !sliceData.pixels) return;

    const ctx = canvas.getContext('2d');
    const width = sliceData.width;
    const height = sliceData.height;
    canvas.width = width;
    canvas.height = height;

    const imgData = ctx.createImageData(width, height);
    const pixels = sliceData.pixels;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = pixels[y][x];

        // Apply window preset modulation
        if (windowPreset === 'bone') {
          val = Math.pow(val, 1.8);
        } else if (windowPreset === 'soft') {
          val = Math.min(1.0, val * 1.5);
        }

        const idx = (y * width + x) * 4;
        const gray = Math.floor(val * 255);
        imgData.data[idx] = gray;
        imgData.data[idx + 1] = gray;
        imgData.data[idx + 2] = gray;
        imgData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // Overlay AI Saliency Heatmap if enabled
    if (showHeatmap && heatmapData && heatmapData.normalized_heatmap_grid) {
      const grid = heatmapData.normalized_heatmap_grid;
      const gRows = grid.length;
      const gCols = grid[0].length;
      const cellW = width / gCols;
      const cellH = height / gRows;

      for (let r = 0; r < gRows; r++) {
        for (let c = 0; c < gCols; c++) {
          const heat = grid[r][c];
          if (heat > 0.15) {
            ctx.fillStyle = `rgba(${Math.floor(heat * 255)}, ${Math.floor((1 - heat) * 120)}, 50, ${heat * heatmapOpacity})`;
            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
          }
        }
      }

      // Draw bounding box if lesion detected on this slice
      if (heatmapData.lesions_on_slice && heatmapData.lesions_on_slice.length > 0) {
        heatmapData.lesions_on_slice.forEach((lesion) => {
          const bbox = lesion.bounding_box_3d;
          if (bbox) {
            const scaleX = width / 64;
            const scaleY = height / 64;
            const bx = bbox.x_min * scaleX;
            const by = bbox.y_min * scaleY;
            const bw = (bbox.x_max - bbox.x_min) * scaleX;
            const bh = (bbox.y_max - bbox.y_min) * scaleY;

            ctx.strokeStyle = '#f43f5e';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(bx, by, bw, bh);

            ctx.fillStyle = '#f43f5e';
            ctx.font = '9px monospace';
            ctx.fillText(`NODULE: ${Math.round(lesion.malignancy_score * 100)}%`, bx, Math.max(10, by - 4));
          }
        });
      }
    }
  }, [sliceData, windowPreset, showHeatmap, heatmapData, heatmapOpacity]);

  // Draw 3D Interactive Point Cloud / Mesh on 3D Canvas
  useEffect(() => {
    const canvas = canvas3DRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#050810';
    ctx.fillRect(0, 0, width, height);

    // Subtle 3D grid cage
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.strokeRect(width * 0.15, height * 0.15, width * 0.7, height * 0.7);

    const radX = (rotX * Math.PI) / 180;
    const radY = (rotY * Math.PI) / 180;
    const cosX = Math.cos(radX);
    const sinX = Math.sin(radX);
    const cosY = Math.cos(radY);
    const sinY = Math.sin(radY);

    const centerX = width / 2;
    const centerY = height / 2;
    const scale = (Math.min(width, height) * 0.42) * zoom3D;

    // Render 3D points
    if (meshPoints && meshPoints.length > 0) {
      meshPoints.forEach(([nx, ny, nz, val]) => {
        // Rotation around Y then X
        const x1 = nx * cosY + nz * sinY;
        const z1 = -nx * sinY + nz * cosY;
        const y1 = ny * cosX - z1 * sinX;
        const z2 = ny * sinX + z1 * cosX;

        // Orthographic/Perspective projection
        const depthFactor = (z2 + 2.5) / 2.5;
        const screenX = centerX + x1 * scale * depthFactor;
        const screenY = centerY + y1 * scale * depthFactor;

        // Color mapped to tissue density
        let r = 0, g = 210, b = 180;
        if (val > 0.70) {
          // Bone / Ribs (White/Yellow)
          r = 255; g = 240; b = 180;
        } else if (val > 0.50) {
          // Soft tissue / Focal lesion
          r = 244; g = 63; b = 94;
        }

        const alpha = Math.max(0.15, Math.min(0.9, (val - 0.1) * 1.5));
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        const dotSize = Math.max(1.2, val * 2.8 * zoom3D);
        ctx.fillRect(screenX, screenY, dotSize, dotSize);
      });
    }

    // 3D Nodule Highlight Beacon
    if (showBoundingBox) {
      const noduleNorm = [-0.35, 0.05, 0.1]; // Right upper thoracic lobe
      const x1 = noduleNorm[0] * cosY + noduleNorm[2] * sinY;
      const z1 = -noduleNorm[0] * sinY + noduleNorm[2] * cosY;
      const y1 = noduleNorm[1] * cosX - z1 * sinX;
      const z2 = noduleNorm[1] * sinX + z1 * cosX;

      const scrX = centerX + x1 * scale * ((z2 + 2.5) / 2.5);
      const scrY = centerY + y1 * scale * ((z2 + 2.5) / 2.5);

      // Pulsing target ring
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(scrX, scrY, 14, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#f43f5e';
      ctx.beginPath();
      ctx.arc(scrX, scrY, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = '10px monospace';
      ctx.fillStyle = '#ff6b81';
      ctx.fillText('SUSPECTED NODULE', scrX + 18, scrY + 4);
    }
  }, [meshPoints, rotX, rotY, zoom3D, showBoundingBox]);

  // 3D Mouse Drag handlers
  const handleMouseDown = (e) => {
    setIsDragging3D(true);
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
    if (!isDragging3D) return;
    const dx = e.clientX - lastMousePos.x;
    const dy = e.clientY - lastMousePos.y;
    setRotY((prev) => prev + dx * 0.7);
    setRotX((prev) => Math.max(-85, Math.min(85, prev - dy * 0.7)));
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging3D(false);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    setZoom3D((prev) => Math.max(0.5, Math.min(2.5, prev - e.deltaY * 0.0015)));
  };

  if (!currentScan) {
    return (
      <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
        <Layers size={40} color="#00d2b4" style={{ marginBottom: '1rem' }} />
        <h2 style={{ fontSize: '1.4rem', marginBottom: '0.5rem' }}>No CT Scan Selected</h2>
        <p style={{ color: '#94a3b8', marginBottom: '1.5rem' }}>
          Please upload a scan or click Quick-Load Demo in the Ingestion tab first.
        </p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <div className="view-header">
        <div className="view-title-group">
          <h1>3D Volumetric Reconstruction & Multiplanar Scrubber</h1>
          <p>Multiplanar reformatting (Axial, Coronal, Sagittal) and 3D organ mesh isosurface viewer.</p>
        </div>

        <button
          id="btn-goto-analysis"
          className="btn btn-primary"
          onClick={onNavigateToResults}
        >
          <span>Run 3D CNN Disease Analysis</span>
          <ArrowRight size={16} />
        </button>
      </div>

      <div className="grid-2col" style={{ alignItems: 'start' }}>
        {/* 2D Slice Scrubber Panel */}
        <div className="viewer-panel">
          <div className="viewer-toolbar">
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              {['axial', 'coronal', 'sagittal'].map((p) => (
                <button
                  key={p}
                  className={`btn ${plane === p ? 'btn-outline-primary' : 'btn-secondary'}`}
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', textTransform: 'capitalize' }}
                  onClick={() => { setPlane(p); setSliceIndex(16); }}
                >
                  {p}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <select
                value={windowPreset}
                onChange={(e) => setWindowPreset(e.target.value)}
                className="form-select"
                style={{
                  padding: '0.3rem 0.5rem',
                  fontSize: '0.78rem',
                  width: 'auto'
                }}
              >
                <option value="lung">Lung Window (-600 HU)</option>
                <option value="bone">Bone Window (+400 HU)</option>
                <option value="soft">Soft Tissue (+40 HU)</option>
              </select>

              <button
                className={`btn ${showHeatmap ? 'btn-outline-primary' : 'btn-secondary'}`}
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
                onClick={() => setShowHeatmap(!showHeatmap)}
                title="Toggle AI Attention Heatmap Overlay"
              >
                <Eye size={13} />
                <span>AI Saliency</span>
              </button>
            </div>
          </div>

          <div className="canvas-viewport">
            <canvas ref={canvas2DRef} className="slice-canvas" />

            <div className="canvas-hud-overlay">
              <div>SCAN: {currentScan.scan_uid}</div>
              <div>PLANE: {plane.toUpperCase()}</div>
              <div>THICKNESS: {currentScan.slice_thickness_mm || 1.25} mm</div>
            </div>

            <div className="canvas-crosshair-hud">
              SLICE {sliceIndex + 1} / {totalSlices}
            </div>
          </div>

          <div className="scrubber-bar">
            <button
              className="btn btn-secondary"
              style={{ padding: '0.4rem 0.65rem' }}
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            </button>

            <input
              type="range"
              min="0"
              max={Math.max(0, totalSlices - 1)}
              value={sliceIndex}
              onChange={(e) => setSliceIndex(parseInt(e.target.value, 10))}
              className="slider-custom"
            />

            <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--primary-hover)', minWidth: '85px', textAlign: 'right', fontWeight: 600 }}>
              {sliceIndex + 1} / {totalSlices}
            </span>
          </div>
        </div>

        {/* 3D Volumetric Reconstruction & Mesh Panel */}
        <div className="viewer-panel">
          <div className="viewer-toolbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Compass size={16} color="#06b6d4" />
              <span style={{ fontSize: '0.86rem', fontWeight: 600, color: 'var(--text-main)', marginRight: '0.4rem' }}>
                3D Reconstruction
              </span>
              <div style={{ display: 'flex', background: '#f1f5f9', borderRadius: '6px', padding: '2px', border: '1px solid var(--border-subtle)' }}>
                <button
                  className={`btn ${view3DMode === 'cloud' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '0.2rem 0.45rem', fontSize: '0.72rem', border: 'none', background: view3DMode === 'cloud' ? 'var(--primary)' : 'transparent', color: view3DMode === 'cloud' ? '#ffffff' : 'var(--text-muted)' }}
                  onClick={() => setView3DMode('cloud')}
                >
                  Point Cloud
                </button>
                <button
                  className={`btn ${view3DMode === 'surface' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '0.2rem 0.45rem', fontSize: '0.72rem', border: 'none', background: view3DMode === 'surface' ? 'var(--primary)' : 'transparent', color: view3DMode === 'surface' ? '#ffffff' : 'var(--text-muted)' }}
                  onClick={() => setView3DMode('surface')}
                >
                  Marching Cubes 3D
                </button>
                <button
                  className={`btn ${view3DMode === 'mip' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '0.2rem 0.45rem', fontSize: '0.72rem', border: 'none', background: view3DMode === 'mip' ? 'var(--primary)' : 'transparent', color: view3DMode === 'mip' ? '#ffffff' : 'var(--text-muted)' }}
                  onClick={() => setView3DMode('mip')}
                >
                  MIP Projection
                </button>
              </div>
            </div>

            {view3DMode === 'cloud' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <button
                  className="btn btn-secondary"
                  style={{ padding: '0.3rem 0.55rem' }}
                  onClick={() => setZoom3D((z) => Math.min(2.5, z + 0.15))}
                  title="Zoom In"
                >
                  <ZoomIn size={14} />
                </button>
                <button
                  className="btn btn-secondary"
                  style={{ padding: '0.3rem 0.55rem' }}
                  onClick={() => setZoom3D((z) => Math.max(0.5, z - 0.15))}
                  title="Zoom Out"
                >
                  <ZoomOut size={14} />
                </button>
                <button
                  className="btn btn-secondary"
                  style={{ padding: '0.3rem 0.55rem' }}
                  onClick={() => { setRotX(25); setRotY(45); setZoom3D(1.0); }}
                  title="Reset Camera"
                >
                  <RotateCw size={14} />
                </button>
              </div>
            )}

            {view3DMode === 'surface' && (
              <a
                href={getIsosurfaceHtmlUrl(currentScan.id)}
                target="_blank"
                rel="noreferrer"
                className="btn btn-secondary"
                style={{ padding: '0.3rem 0.55rem', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                title="Open 3D Mesh in New Tab"
              >
                <ExternalLink size={12} />
                <span>Fullscreen</span>
              </a>
            )}
          </div>

          {view3DMode === 'cloud' && (
            <>
              <div
                className="canvas-viewport"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={handleWheel}
                style={{ cursor: isDragging3D ? 'grabbing' : 'grab' }}
              >
                <canvas
                  ref={canvas3DRef}
                  width={600}
                  height={440}
                  style={{ width: '100%', height: '440px', display: 'block' }}
                />

                <div className="canvas-hud-overlay">
                  <div>POINTS: {meshPoints.length}</div>
                  <div>ROT: X:{Math.round(rotX)}° Y:{Math.round(rotY)}°</div>
                  <div>ZOOM: {zoom3D.toFixed(2)}x</div>
                </div>

                <div className="canvas-crosshair-hud">
                  Drag to Orbit 360° · Scroll to Zoom
                </div>
              </div>

              <div className="scrubber-bar" style={{ justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', flex: 1 }}>
                  <Sliders size={14} color="#06b6d4" />
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Tissue Density:</span>
                  <input
                    type="range"
                    min="0.10"
                    max="0.65"
                    step="0.02"
                    value={isoThreshold}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setIsoThreshold(val);
                      loadMesh(val);
                    }}
                    className="slider-custom"
                    style={{ maxWidth: '160px' }}
                  />
                  <span className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--primary-hover)', fontWeight: 600 }}>
                    {isoThreshold.toFixed(2)}
                  </span>
                </div>

                <button
                  className={`btn ${showBoundingBox ? 'btn-outline-primary' : 'btn-secondary'}`}
                  style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
                  onClick={() => setShowBoundingBox(!showBoundingBox)}
                >
                  <Crosshair size={12} />
                  <span>Target Nodule</span>
                </button>
              </div>
            </>
          )}

          {view3DMode === 'surface' && (
            <div className="canvas-viewport" style={{ height: '495px', position: 'relative', background: '#0b1120', overflow: 'hidden' }}>
              <iframe
                src={getIsosurfaceHtmlUrl(currentScan.id)}
                title="3D Marching Cubes Isosurface"
                style={{ width: '100%', height: '100%', border: 'none' }}
              />
            </div>
          )}

          {view3DMode === 'mip' && (
            <div className="canvas-viewport" style={{ height: '495px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#0b1120', padding: '1rem' }}>
              <img
                src={getMipImageUrl(currentScan.id)}
                alt="Maximum Intensity Projection"
                style={{ maxHeight: '420px', maxWidth: '100%', objectFit: 'contain', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)' }}
              />
              <div style={{ marginTop: '0.5rem', fontSize: '0.78rem', color: '#94a3b8', display: 'flex', gap: '1rem' }}>
                <span>Axial (XY)</span>
                <span>•</span>
                <span>Coronal (XZ)</span>
                <span>•</span>
                <span>Sagittal (YZ)</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
