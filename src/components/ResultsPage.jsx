export default function ResultsPage({ scan, onBack }) {
  return (
    <div className="grid-view">
      <section className="card">
        <div className="card-head">
          <div>
            <h2 className="card-title">Results</h2>
            <p className="muted">
              {scan.fileName} · Patient {scan.patientId}
            </p>
          </div>
          <span className="status">No results</span>
        </div>

        <div className="empty-block">
          <div className="empty-icon dark-text">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z" />
              <path d="M14 3v5h5M9 13h6M9 17h4" />
            </svg>
          </div>
          <strong>No results yet</strong>
          <span>Analysis results for this scan will show up here.</span>
        </div>
      </section>

      <aside className="side">
        <button className="btn btn-outline" onClick={onBack}>
          Back to 3D view
        </button>
      </aside>
    </div>
  );
}
