import { useState } from "react";
import Header from "./components/Header.jsx";
import UploadPage from "./components/UploadPage.jsx";
import ViewerPage from "./components/ViewerPage.jsx";
import ResultsPage from "./components/ResultsPage.jsx";
import { USE_MOCK } from "./api/client.js";

export default function App() {
  const [step, setStep] = useState("upload"); // upload | view | results
  const [scan, setScan] = useState(null);

  const handleUploaded = (info) => {
    setScan(info);
    setStep("view");
  };

  return (
    <div className="app">
      <div className="shell">
        <Header step={step} setStep={setStep} hasScan={!!scan} />
        {USE_MOCK && (
          <div className="banner">
            Demo mode: the upload is simulated. Set <code>VITE_USE_MOCK=false</code> in <code>.env</code> to use the backend.
          </div>
        )}
        <main className="main" key={step}>
          {step === "upload" && <UploadPage onUploaded={handleUploaded} />}
          {step === "view" && scan && (
            <ViewerPage
              scan={scan}
              onUploadAnother={() => setStep("upload")}
              onResults={() => setStep("results")}
            />
          )}
          {step === "results" && scan && (
            <ResultsPage scan={scan} onBack={() => setStep("view")} />
          )}
        </main>
      </div>
      <footer className="foot">AI-Powered 3D Medical Image Reconstruction and Disease Analysis</footer>
    </div>
  );
}
