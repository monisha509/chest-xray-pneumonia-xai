import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Activity, LogOut, ArrowLeft } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useAnalysis } from '../context/AnalysisContext';

export default function Navbar({ showBack = false, backTo = '/dashboard' }) {
  const { user, logout } = useAuth();
  const { resetAnalysis } = useAnalysis();
  const navigate = useNavigate();

  const handleLogout = () => {
    resetAnalysis();
    logout();
    navigate('/login');
  };

  return (
    <header className="navbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {showBack && (
          <button
            onClick={() => navigate(backTo)}
            className="nav-link-btn"
            title="Go Back"
          >
            <ArrowLeft size={16} />
            Back
          </button>
        )}
        <Link to="/dashboard" className="nav-brand">
          <div className="brand-icon">
            <Activity size={20} />
          </div>
          <span className="brand-title">ChestXR</span>
          <span className="brand-badge">Research v1.0</span>
        </Link>
      </div>

      {user && (
        <div className="nav-user-actions">
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            User: <strong style={{ color: 'var(--primary-navy)' }}>{user}</strong>
          </span>
          <button onClick={handleLogout} className="nav-link-btn" title="Sign out">
            <LogOut size={15} />
            Sign Out
          </button>
        </div>
      )}
    </header>
  );
}

