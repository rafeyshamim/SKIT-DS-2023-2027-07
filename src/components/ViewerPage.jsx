import { useRef, useState } from "react";
import ModelViewer from "./ModelViewer.jsx";

export default function ViewerPage({ scan, onUploadAnother, onResults }) {
  const viewer = useRef(null);
  const hasModel = !!scan.mesh;
  const [opacity, setOpacity] = useState(0.85);
  const [autoRotate, setAutoRotate] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [showModel, setShowModel] = useState(true);
  const [view, setView] = useState("front");

  const changeView = (v) => {
    setView(v);
    viewer.current.setView(v);
  };

  const screenshot = () => {
    const url = viewer.current.screenshot();
    const a = document.createElement("a");
    a.href = url;
    a.download = `${scan.patientId}_3d_view.png`;
    a.click();
  };

  return (
    <div className="grid-view">
      <section className="card">
        <div className="card-head">
          <div>
            <h2 className="card-title">3D reconstruction</h2>
            <p className="muted">
              {scan.fileName} · Patient {scan.patientId}
            </p>
          </div>
          <span className={`status ${hasModel ? "status-ok" : ""}`}>
            {hasModel ? "Model ready" : "No model yet"}
          </span>
        </div>

        <div className="view-tabs" role="tablist">
          {["front", "side", "top"].map((v) => (
            <button
              key={v}
              role="tab"
              aria-selected={view === v}
              className={`vtab ${view === v ? "vtab-on" : ""}`}
              onClick={() => changeView(v)}
            >
              {v[0].toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>

        <div className="stage">
          <ModelViewer
            ref={viewer}
            mesh={scan.mesh}
            opacity={opacity}
            autoRotate={autoRotate}
            wireframe={wireframe}
            showModel={showModel}
          />

          {!hasModel && (
            <div className="empty-overlay">
              <div className="empty-icon">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round">
                  <path d="M12 2l8 4.5v9L12 20l-8-4.5v-9L12 2z" />
                  <path d="M12 11v9M4 6.5l8 4.5 8-4.5" />
                </svg>
              </div>
              <strong>3D model will appear here</strong>
              <span>Reconstruction isn't connected yet. Once the backend returns a model, it shows up on this stage.</span>
            </div>
          )}

          <div className="zoom-col">
            <button className="icon-btn" onClick={() => viewer.current.zoomIn()} aria-label="Zoom in">+</button>
            <button className="icon-btn" onClick={() => viewer.current.zoomOut()} aria-label="Zoom out">−</button>
            <button
              className="icon-btn"
              onClick={() => {
                viewer.current.reset();
                setView("front");
              }}
              aria-label="Reset view"
            >
              ⟳
            </button>
          </div>
          <div className="hint">Drag to rotate · scroll to zoom</div>
        </div>

        <div className={`controls-row ${hasModel ? "" : "controls-off"}`}>
          <label className="opacity">
            Opacity
            <input
              type="range"
              min="0.15"
              max="1"
              step="0.05"
              value={opacity}
              disabled={!hasModel}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
            />
          </label>
          <label className="check">
            <input type="checkbox" checked={autoRotate} disabled={!hasModel} onChange={(e) => setAutoRotate(e.target.checked)} />
            Auto-rotate
          </label>
          <label className="check">
            <input type="checkbox" checked={wireframe} disabled={!hasModel} onChange={(e) => setWireframe(e.target.checked)} />
            Wireframe
          </label>
        </div>
      </section>

      <aside className="side">
        <section className="card">
          <h3 className="card-title">Scan details</h3>
          <dl className="details">
            <div><dt>Slices</dt><dd>{scan.slices}</dd></div>
            <div><dt>Resolution</dt><dd>{scan.resolution}</dd></div>
            <div><dt>Slice thickness</dt><dd>{scan.thickness}</dd></div>
            <div><dt>Uploaded</dt><dd>{scan.uploadedAt}</dd></div>
          </dl>
        </section>

        <section className="card">
          <h3 className="card-title">Layers</h3>
          {hasModel ? (
            <ul className="layers">
              <li>
                <span>Reconstructed model</span>
                <button className={`toggle ${showModel ? "toggle-on" : ""}`} onClick={() => setShowModel((v) => !v)}>
                  {showModel ? "Shown" : "Hidden"}
                </button>
              </li>
            </ul>
          ) : (
            <p className="muted small">No layers yet. They appear once a 3D model is available.</p>
          )}
        </section>

        <button className="btn btn-primary" onClick={screenshot} disabled={!hasModel}>
          Save screenshot
        </button>
        <button className="btn btn-outline" onClick={onResults}>
          Results
        </button>
        <button className="btn btn-outline" onClick={onUploadAnother}>
          Upload another scan
        </button>
      </aside>
    </div>
  );
}
