import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Lock, User, Eye, EyeOff, AlertTriangle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!username.trim()) {
      setError('Please enter your email or username.');
      return;
    }
    if (!password) {
      setError('Please enter your password.');
      return;
    }
    setError('');
    // Demo-only login flow
    login(username.trim());
    navigate('/dashboard');
  };

  return (
    <div className="app-container" style={{ justifyContent: 'center', alignItems: 'center' }}>
      <div style={{ maxWidth: '440px', width: '100%', padding: '1rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div
            style={{
              display: 'inline-flex',
              padding: '0.85rem',
              backgroundColor: 'var(--medical-teal)',
              borderRadius: '12px',
              color: '#ffffff',
              marginBottom: '1rem',
            }}
          >
            <Activity size={36} />
          </div>
          <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--primary-navy)' }}>
            ChestXR Portal
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', marginTop: '0.25rem' }}>
            Chest X-Ray Pneumonia Detection & Explainable AI
          </p>
        </div>

        <div className="card">
          <div className="alert-banner alert-info" style={{ marginBottom: '1.5rem' }}>
            <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <strong>Educational / Research Access</strong>
              <div style={{ fontSize: '0.8rem', marginTop: '2px' }}>
                This is a research prototype. Demo access is enabled—enter any username/password to proceed.
              </div>
            </div>
          </div>

          {error && (
            <div className="alert-banner alert-danger" style={{ marginBottom: '1.25rem' }}>
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label" htmlFor="username">
                Username / Email
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  id="username"
                  type="text"
                  className="form-input"
                  style={{ width: '100%', paddingLeft: '2.5rem' }}
                  placeholder="e.g. researcher@hospital.edu"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                />
                <User
                  size={16}
                  style={{
                    position: 'absolute',
                    left: '0.85rem',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    color: 'var(--text-muted)',
                  }}
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="password">
                Password
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  className="form-input"
                  style={{ width: '100%', paddingLeft: '2.5rem', paddingRight: '2.5rem' }}
                  placeholder="Enter password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                />
                <Lock
                  size={16}
                  style={{
                    position: 'absolute',
                    left: '0.85rem',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    color: 'var(--text-muted)',
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  style={{
                    position: 'absolute',
                    right: '0.75rem',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: 'var(--text-muted)',
                  }}
                  title={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: '100%', marginTop: '1rem', padding: '0.75rem' }}
            >
              Sign In to ChestXR
            </button>
          </form>
        </div>

        <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          ChestXR Evaluation Prototype &copy; 2026. For research and demonstration purposes only.
        </div>
      </div>
    </div>
  );
}

