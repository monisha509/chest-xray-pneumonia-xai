import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, FileImage, Trash2, RefreshCw, AlertCircle, ArrowRight } from 'lucide-react';
import Navbar from '../components/Navbar';
import { useAnalysis } from '../context/AnalysisContext';

const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15MB
const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png'];

export default function UploadPage() {
  const { selectedImage, imagePreviewUrl, setImage } = useAnalysis();
  const [imageMeta, setImageMeta] = useState({ dimensions: null, sizeStr: null });
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState('');
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const processFile = (file) => {
    setValidationError('');
    if (!file) return;

    // Check extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setValidationError(`Invalid file type (.${ext}). Allowed formats: JPG, JPEG, PNG.`);
      return;
    }

    // Check size
    if (file.size === 0) {
      setValidationError('The selected file is empty (0 bytes).');
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setValidationError(`File exceeds maximum size limit (15 MB). File size: ${(file.size / (1024 * 1024)).toFixed(1)} MB.`);
      return;
    }

    // Inspect image dimensions via browser Image object
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImageMeta({
        dimensions: `${img.naturalWidth} × ${img.naturalHeight} px`,
        sizeStr: `${(file.size / 1024).toFixed(1)} KB`,
      });
      URL.revokeObjectURL(objectUrl);
      setImage(file);
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setValidationError('Failed to decode image data. The file may be corrupted.');
    };
    img.src = objectUrl;
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleRemove = () => {
    setImage(null);
    setImageMeta({ dimensions: null, sizeStr: null });
    setValidationError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyze = () => {
    if (!selectedImage) {
      setValidationError('Please select or upload a chest X-ray image first.');
      return;
    }
    navigate('/processing');
  };

  return (
    <div className="app-container">
      <Navbar showBack backTo="/case-info" />

      <main className="main-content">
        <div style={{ maxWidth: '750px', width: '100%' }}>
          <div style={{ marginBottom: '1.5rem' }}>
            <h1 className="card-title">X-Ray Upload</h1>
            <p className="card-subtitle">
              Upload a standard chest X-ray image for pneumonia analysis and Grad-CAM explainability.
            </p>
          </div>

          {validationError && (
            <div className="alert-banner alert-danger">
              <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>{validationError}</div>
            </div>
          )}

          <div className="card">
            {!imagePreviewUrl ? (
              <div
                className={`dropzone ${isDragging ? 'active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div style={{ display: 'inline-flex', padding: '1rem', backgroundColor: 'var(--medical-teal-light)', color: 'var(--medical-teal)', borderRadius: '50%', marginBottom: '1rem' }}>
                  <UploadCloud size={32} />
                </div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--primary-navy)', marginBottom: '0.5rem' }}>
                  Drag and drop chest X-ray here
                </h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
                  Supports PNG, JPG, or JPEG format up to 15 MB.
                </p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                >
                  <FileImage size={16} />
                  Choose Image File
                </button>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
                  <div
                    style={{
                      width: '260px',
                      height: '260px',
                      backgroundColor: '#0b1120',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: '1px solid var(--border-color)',
                    }}
                  >
                    <img
                      src={imagePreviewUrl}
                      alt="Uploaded X-ray preview"
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                  </div>

                  <div style={{ flex: 1, minWidth: '240px' }}>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary-navy)', wordBreak: 'break-all' }}>
                      {selectedImage?.name}
                    </h3>
                    <div style={{ marginTop: '0.75rem', fontSize: '0.9rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                      <div>Size: <strong style={{ color: 'var(--primary-navy)' }}>{imageMeta.sizeStr || `${(selectedImage.size / 1024).toFixed(1)} KB`}</strong></div>
                      <div>Dimensions: <strong style={{ color: 'var(--primary-navy)' }}>{imageMeta.dimensions || 'Loaded'}</strong></div>
                      <div>Pipeline Target: <strong style={{ color: 'var(--medical-teal)' }}>224 × 224 px (Bilinear Grayscale)</strong></div>
                    </div>

                    <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="btn btn-secondary"
                        style={{ fontSize: '0.85rem', padding: '0.5rem 0.9rem' }}
                      >
                        <RefreshCw size={14} />
                        Replace Image
                      </button>
                      <button
                        type="button"
                        onClick={handleRemove}
                        className="btn btn-secondary"
                        style={{ fontSize: '0.85rem', padding: '0.5rem 0.9rem', color: 'var(--danger-red)' }}
                      >
                        <Trash2 size={14} />
                        Remove
                      </button>
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    type="button"
                    onClick={handleAnalyze}
                    className="btn btn-primary"
                    style={{ padding: '0.75rem 1.75rem', fontSize: '1rem' }}
                  >
                    Analyze X-ray
                    <ArrowRight size={18} />
                  </button>
                </div>
              </div>
            )}

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              style={{ display: 'none' }}
            />
          </div>
        </div>
      </main>

      <footer className="footer">
        ChestXR Research Application &copy; 2026.
      </footer>
    </div>
  );
}

