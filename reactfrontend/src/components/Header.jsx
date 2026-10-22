const STEPS = [
  { key: "upload", label: "Upload" },
  { key: "view", label: "3D view" },
  { key: "results", label: "Results" },
];

export default function Header({ step, setStep, hasScan }) {
  const current = STEPS.findIndex((s) => s.key === step);

  return (
    <header className="header">
      <div className="brand">
        <div className="logo">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinejoin="round">
            <path d="M12 2l8 4.5v9L12 20l-8-4.5v-9L12 2z" />
            <path d="M12 11v9M4 6.5l8 4.5 8-4.5" />
          </svg>
        </div>
        <div>
          <div className="brand-name">CT Vision</div>
          <div className="brand-sub">Project DS-07</div>
        </div>
      </div>

      <nav className="steps" aria-label="Progress">
        {STEPS.map((s, i) => {
          const locked = s.key !== "upload" && !hasScan;
          const state = i === current ? "step-on" : i < current ? "step-done" : "";
          return (
            <button
              key={s.key}
              className={`step ${state}`}
              disabled={locked}
              aria-current={i === current ? "step" : undefined}
              onClick={() => setStep(s.key)}
            >
              <span className="step-num">{i < current ? "✓" : i + 1}</span>
              {s.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
