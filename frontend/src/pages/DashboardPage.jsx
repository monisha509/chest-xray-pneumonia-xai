import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Cpu, Eye, ShieldAlert, CheckCircle2 } from 'lucide-react';
import Navbar from '../components/Navbar';
import { useAuth } from '../context/AuthContext';
import { useAnalysis } from '../context/AnalysisContext';
import { checkHealth } from '../services/api';

export default function DashboardPage() {
  const { user } = useAuth();
  const { resetAnalysis } = useAnalysis();
  const navigate = useNavigate();
  const [modelStatus, setModelStatus] = useState({ loaded: false, checked: false });

  useEffect(() => {
    checkHealth()
      .then((res) => {
        setModelStatus({ loaded: res.model_loaded, checked: true });
      })
      .catch(() => {
        setModelStatus({ loaded: false, checked: true });
      });
  }, []);

  const handleStartAnalysis = () => {
    resetAnalysis();
    navigate('/case-info');
  };

  return (
    <div className="app-container">
      <Navbar />

      <main className="main-content">
        <div style={{ width: '100%', marginBottom: '2.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h1 style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--primary-navy)' }}>
                Research Workspace
              </h1>
              <p style={{ color: 'var(--text-muted)', fontSize: '1.05rem', marginTop: '0.25rem' }}>
                Welcome, {user || 'Researcher'}. Pneumonia detection inference and visual explainability platform.
              </p>
            </div>

            <button
              onClick={handleStartAnalysis}
              className="btn btn-primary"
              style={{ padding: '0.8rem 1.75rem', fontSize: '1rem' }}
            >
              Start New Analysis
              <ArrowRight size={18} />
            </button>
          </div>
        </div>

        {modelStatus.checked && (
          <div className={`alert-banner ${modelStatus.loaded ? 'alert-info' : 'alert-danger'}`} style={{ marginBottom: '2rem' }}>
            <CheckCircle2 size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <strong>Model Backend Status: {modelStatus.loaded ? 'Ready' : 'Offline / Error'}</strong>
              <div style={{ fontSize: '0.85rem' }}>
                {modelStatus.loaded
                  ? 'Frozen MobileNetV2 architecture loaded with classification threshold fixed at 0.50.'
                  : 'Backend service or model weights not accessible. Verify backend is running on port 8000.'}
              </div>
            </div>
          </div>
        )}

        <div className="grid-cards">
          <div className="feature-card">
            <div className="feature-card-icon">
              <Cpu size={24} />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--primary-navy)', marginBottom: '0.5rem' }}>
              MobileNetV2 CNN Architecture
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.5 }}>
              Transfer learning binary classification backbone pretrained on ImageNet with integrated model-boundary scaling mapping [0, 255] to [-1.0, 1.0].
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-card-icon">
              <Eye size={24} />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--primary-navy)', marginBottom: '0.5rem' }}>
              Grad-CAM Explainable AI
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.5 }}>
              Gradient-weighted Class Activation Mapping computed from the final convolutional layer (<code>out_relu</code>) highlighting spatial features influencing output scores.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-card-icon">
              <ShieldAlert size={24} />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--primary-navy)', marginBottom: '0.5rem' }}>
              Educational / Research Mode
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.5 }}>
              Designed strictly for experimental validation and academic research. No clinical diagnostic decisions or permanent storage of patient data.
            </p>
          </div>
        </div>

        <div className="card" style={{ marginTop: '2.5rem', backgroundColor: '#f8fafc' }}>
          <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--primary-navy)', marginBottom: '0.5rem' }}>
            Analysis Workflow Overview
          </h4>
          <ol style={{ paddingLeft: '1.25rem', color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.7 }}>
            <li>Optional Case Information (de-identified research metadata).</li>
            <li>Chest X-ray image upload (PNG, JPG, or JPEG format).</li>
            <li>Inference & Grad-CAM visual heatmap generation.</li>
            <li>Side-by-side inspection of original X-ray alongside explainability overlay.</li>
          </ol>
        </div>
      </main>

      <footer className="footer">
        ChestXR — Chest X-Ray Pneumonia Detection Using CNN and Explainable AI &copy; 2026.
      </footer>
    </div>
  );
}

