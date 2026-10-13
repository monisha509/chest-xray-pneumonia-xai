import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, CheckCircle2, AlertCircle, ArrowLeft, RefreshCw } from 'lucide-react';
import Navbar from '../components/Navbar';
import { useAnalysis } from '../context/AnalysisContext';
import { predictXRay } from '../services/api';

const STAGES = [
  { id: 1, label: 'Image uploaded' },
  { id: 2, label: 'Image preprocessing' },
  { id: 3, label: 'Running MobileNetV2' },
  { id: 4, label: 'Generating Grad-CAM' },
  { id: 5, label: 'Preparing results' },
];

export default function ProcessingPage() {
  const { selectedImage, caseMetadata, setPredictionResult } = useAnalysis();
  const [currentStage, setCurrentStage] = useState(1);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!selectedImage) {
      navigate('/upload');
      return;
    }

    const abortController = new AbortController();
    let isCurrentEffect = true;

    async function executeWorkflow() {
      try {
        // Stage 1: Image uploaded
        if (!isCurrentEffect) return;
        setCurrentStage(1);
        await new Promise((r) => setTimeout(r, 300));

        // Stage 2: Image preprocessing
        if (!isCurrentEffect) return;
        setCurrentStage(2);
        await new Promise((r) => setTimeout(r, 400));

        // Stage 3: Running MobileNetV2
        if (!isCurrentEffect) return;
        setCurrentStage(3);

        // Start real inference call with timeout and abort signal
        const apiPromise = predictXRay(selectedImage, caseMetadata, {
          timeoutMs: 60000,
          signal: abortController.signal,
        });

        // Stage 4: Advance to Grad-CAM progress state after brief inference indication
        const gradCamTimer = setTimeout(() => {
          if (isCurrentEffect) {
            setCurrentStage(4);
          }
        }, 1200);

        const result = await apiPromise;
        clearTimeout(gradCamTimer);

        if (!isCurrentEffect) return;

        // Stage 5: Preparing results
        setCurrentStage(5);
        await new Promise((r) => setTimeout(r, 300));

        if (!isCurrentEffect) return;
        setPredictionResult(result);
        navigate('/results');
      } catch (err) {
        // Do not set error if component was unmounted by user navigation
        if (isCurrentEffect && err.name !== 'AbortError') {
          console.error('ChestXR prediction error:', err);
          setError(err.message || 'An unexpected error occurred during image processing.');
        }
      }
    }

    executeWorkflow();

    return () => {
      isCurrentEffect = false;
      // Abort in-flight network request if user navigates away
      abortController.abort();
    };
  }, [selectedImage, caseMetadata, navigate, setPredictionResult]);

  const handleRetry = () => {
    setError(null);
    setCurrentStage(1);
    // Trigger re-execution by navigating or re-invoking
    window.location.reload();
  };

  return (
    <div className="app-container">
      <Navbar />

      <main className="main-content">
        <div style={{ maxWidth: '560px', width: '100%' }}>
          <div className="card" style={{ padding: '2.5rem 2rem' }}>
            <h2 className="card-title" style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
              {error ? 'Analysis Failed' : 'Analyzing X-Ray'}
            </h2>
            <p className="card-subtitle" style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
              {error
                ? 'Unable to complete inference and explainability pipeline.'
                : 'Please wait while MobileNetV2 inference and Grad-CAM are executed.'}
            </p>

            {error ? (
              <div>
                <div className="alert-banner alert-danger">
                  <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
                  <div>
                    <strong>Backend Error:</strong>
                    <div style={{ marginTop: '4px' }}>{error}</div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: '2rem' }}>
                  <button
                    onClick={() => navigate('/upload')}
                    className="btn btn-secondary"
                  >
                    <ArrowLeft size={16} />
                    Back to Upload
                  </button>
                  <button
                    onClick={handleRetry}
                    className="btn btn-primary"
                  >
                    <RefreshCw size={16} />
                    Retry Analysis
                  </button>
                </div>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  {STAGES.map((s) => {
                    const isDone = currentStage > s.id;
                    const isActive = currentStage === s.id;

                    return (
                      <div
                        key={s.id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '1rem',
                          padding: '0.85rem 1.25rem',
                          borderRadius: '8px',
                          backgroundColor: isActive ? 'var(--medical-teal-light)' : '#f8fafc',
                          border: `1px solid ${isActive ? 'var(--medical-teal)' : 'var(--border-color)'}`,
                          transition: 'all 0.2s ease',
                        }}
                      >
                        <div style={{ width: '24px', display: 'flex', justifyContent: 'center' }}>
                          {isDone ? (
                            <CheckCircle2 size={20} style={{ color: 'var(--success-green)' }} />
                          ) : isActive ? (
                            <Loader2 size={20} className="spinning" style={{ color: 'var(--medical-teal)' }} />
                          ) : (
                            <div
                              style={{
                                width: '10px',
                                height: '10px',
                                borderRadius: '50%',
                                backgroundColor: '#cbd5e1',
                              }}
                            />
                          )}
                        </div>
                        <span
                          style={{
                            fontSize: '0.95rem',
                            fontWeight: isActive || isDone ? 600 : 500,
                            color: isActive
                              ? 'var(--medical-teal-dark)'
                              : isDone
                              ? 'var(--primary-navy)'
                              : 'var(--text-muted)',
                          }}
                        >
                          {s.label}
                        </span>
                      </div>
                    );
                  })}
                </div>

                <div style={{ textAlign: 'center', marginTop: '2.5rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  Running MobileNetV2 with Grad-CAM layer <code>out_relu</code>...
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        ChestXR Research Application &copy; 2026.
      </footer>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spinning {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
}

