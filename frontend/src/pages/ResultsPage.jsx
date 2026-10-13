import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  RotateCcw,
  LayoutDashboard,
  LogOut,
  Info,
  Calendar,
  Layers,
  Percent,
  CheckCircle,
  AlertTriangle,
} from 'lucide-react';
import Navbar from '../components/Navbar';
import { useAnalysis } from '../context/AnalysisContext';
import { useAuth } from '../context/AuthContext';

export default function ResultsPage() {
  const { predictionResult, resetAnalysis } = useAnalysis();
  const { logout } = useAuth();
  const navigate = useNavigate();

  if (!predictionResult) {
    return (
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <div className="card" style={{ maxWidth: '500px', textAlign: 'center', margin: '2rem auto' }}>
            <h3 style={{ marginBottom: '1rem' }}>No Active Result</h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              No inference data is available in the current session.
            </p>
            <button onClick={() => navigate('/upload')} className="btn btn-primary">
              Go to Upload
            </button>
          </div>
        </main>
      </div>
    );
  }

  const {
    prediction,
    model_score,
    model_score_percentage,
    raw_score,
    threshold,
    model_name,
    timestamp,
    original_image,
    gradcam_image,
    disclaimer,
    case_metadata,
  } = predictionResult;

  const isPneumonia = prediction === 'Pneumonia';

  const handleAnalyzeAnother = () => {
    resetAnalysis();
    navigate('/upload');
  };

  const handleBackToDashboard = () => {
    resetAnalysis();
    navigate('/dashboard');
  };

  const handleLogout = () => {
    resetAnalysis();
    logout();
    navigate('/login');
  };

  return (
    <div className="app-container">
      <Navbar />

      <main className="main-content">
        <div style={{ width: '100%', maxWidth: '1100px' }}>
          {/* Header & Controls */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '1rem',
              marginBottom: '2rem',
            }}
          >
            <div>
              <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--primary-navy)' }}>
                Analysis Workspace
              </h1>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                Completed MobileNetV2 inference and Grad-CAM explainability generation.
              </p>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <button onClick={handleAnalyzeAnother} className="btn btn-primary">
                <RotateCcw size={16} />
                Analyze Another X-ray
              </button>
              <button onClick={handleBackToDashboard} className="btn btn-secondary">
                <LayoutDashboard size={16} />
                Dashboard
              </button>
              <button onClick={handleLogout} className="btn btn-secondary" style={{ color: 'var(--text-muted)' }}>
                <LogOut size={16} />
                Sign Out
              </button>
            </div>
          </div>

          {/* Model Prediction & Score Hero */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '1.5rem',
              marginBottom: '2rem',
            }}
          >
            {/* Prediction Card */}
            <div
              className="card"
              style={{
                borderLeft: `5px solid ${isPneumonia ? 'var(--warning-amber)' : 'var(--medical-teal)'}`,
                padding: '1.75rem',
              }}
            >
              <div style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', fontWeight: 700 }}>
                Model Prediction
              </div>
              <div
                style={{
                  fontSize: '2rem',
                  fontWeight: 800,
                  marginTop: '0.35rem',
                  color: isPneumonia ? '#b45309' : 'var(--medical-teal-dark)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}
              >
                {isPneumonia ? <AlertTriangle size={28} /> : <CheckCircle size={28} />}
                {prediction}
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                Decision rule: Score {raw_score >= threshold ? '≥' : '<'} frozen threshold of {threshold.toFixed(2)}
              </div>
            </div>

            {/* Model Score Card */}
            <div className="card" style={{ padding: '1.75rem' }}>
              <div style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', fontWeight: 700 }}>
                Model Output Score
              </div>
              <div
                style={{
                  fontSize: '2rem',
                  fontWeight: 800,
                  marginTop: '0.35rem',
                  color: 'var(--primary-navy)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                }}
              >
                <Percent size={24} style={{ color: 'var(--medical-teal)' }} />
                {model_score_percentage.toFixed(1)}%
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                Normalized model score: {model_score.toFixed(4)} (Raw sigmoid: {raw_score.toFixed(4)})
              </div>
            </div>
          </div>

          {/* Side-by-Side X-ray Comparison */}
          <div className="card" style={{ marginBottom: '2rem' }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--primary-navy)', marginBottom: '0.25rem' }}>
              X-ray Comparison & Spatial Attribution
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Side-by-side display of the preprocessed grayscale input alongside the Grad-CAM activation heatmap overlay.
            </p>

            <div className="xray-comparison-grid">
              {/* Original Panel */}
              <div className="xray-panel">
                <div className="xray-panel-header">
                  <span>Preprocessed Input X-Ray</span>
                  <span>224 × 224 px</span>
                </div>
                <div className="xray-img-container">
                  <img src={original_image} alt="Preprocessed input X-ray" />
                </div>
              </div>

              {/* Grad-CAM Panel */}
              <div className="xray-panel">
                <div className="xray-panel-header">
                  <span>Grad-CAM Activation Overlay</span>
                  <span>Target: out_relu</span>
                </div>
                <div className="xray-img-container">
                  <img src={gradcam_image} alt="Grad-CAM visualization overlay" />
                </div>
              </div>
            </div>

            {/* Explainability Note */}
            <div
              style={{
                marginTop: '1.5rem',
                padding: '1rem 1.25rem',
                backgroundColor: 'var(--bg-light)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                display: 'flex',
                gap: '0.75rem',
                alignItems: 'flex-start',
              }}
            >
              <Info size={18} style={{ color: 'var(--medical-blue)', flexShrink: 0, marginTop: '2px' }} />
              <div style={{ fontSize: '0.875rem', color: 'var(--secondary-navy)', lineHeight: 1.5 }}>
                <strong>Explainability Note: </strong>
                Grad-CAM highlights image regions that contributed to the model output. This visualization is an explainability aid and is not medical proof.
              </div>
            </div>
          </div>

          {/* Analysis Information & Case Metadata */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '1.5rem',
              marginBottom: '2rem',
            }}
          >
            {/* System Info */}
            <div className="card" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--primary-navy)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Layers size={18} style={{ color: 'var(--medical-teal)' }} />
                Model Specifications
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                <div>Model Architecture: <strong style={{ color: 'var(--primary-navy)' }}>{model_name}</strong></div>
                <div>Backbone Weights: <strong style={{ color: 'var(--primary-navy)' }}>ImageNet (Transfer Learning)</strong></div>
                <div>Decision Cutoff: <strong style={{ color: 'var(--primary-navy)' }}>{threshold.toFixed(2)}</strong></div>
                <div>Target Convolutional Layer: <strong style={{ color: 'var(--primary-navy)' }}>out_relu (7×7×1280)</strong></div>
                <div>Timestamp (UTC): <strong style={{ color: 'var(--primary-navy)' }}>{timestamp}</strong></div>
              </div>
            </div>

            {/* Case Metadata (if provided) */}
            <div className="card" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--primary-navy)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Calendar size={18} style={{ color: 'var(--medical-teal)' }} />
                Session Metadata
              </h3>
              {case_metadata && (case_metadata.case_id || case_metadata.age || case_metadata.sex || case_metadata.examination_date || case_metadata.notes) ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                  <div>Case ID: <strong style={{ color: 'var(--primary-navy)' }}>{case_metadata.case_id || 'Not specified'}</strong></div>
                  <div>Subject Age: <strong style={{ color: 'var(--primary-navy)' }}>{case_metadata.age || 'Not specified'}</strong></div>
                  <div>Subject Biological Sex: <strong style={{ color: 'var(--primary-navy)' }}>{case_metadata.sex || 'Not specified'}</strong></div>
                  <div>Examination Date: <strong style={{ color: 'var(--primary-navy)' }}>{case_metadata.examination_date || 'Not specified'}</strong></div>
                  {case_metadata.notes && (
                    <div style={{ marginTop: '0.25rem' }}>Notes: <em>{case_metadata.notes}</em></div>
                  )}
                </div>
              ) : (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  No case metadata was entered for this analysis session.
                </p>
              )}
            </div>
          </div>

          {/* Mandatory Disclaimer */}
          <div className="alert-banner alert-warning" style={{ marginBottom: '2.5rem' }}>
            <AlertTriangle size={22} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <strong>Mandatory Regulatory & Research Disclaimer:</strong>
              <div style={{ marginTop: '4px', fontSize: '0.875rem' }}>{disclaimer}</div>
            </div>
          </div>
        </div>
      </main>

      <footer className="footer">
        ChestXR Research Application &copy; 2026.
      </footer>
    </div>
  );
}

