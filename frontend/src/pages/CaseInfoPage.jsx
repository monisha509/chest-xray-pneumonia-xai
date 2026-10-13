import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldAlert, ArrowRight } from 'lucide-react';
import Navbar from '../components/Navbar';
import { useAnalysis } from '../context/AnalysisContext';

export default function CaseInfoPage() {
  const { caseMetadata, setCaseMetadata } = useAnalysis();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    case_id: caseMetadata?.case_id || '',
    age: caseMetadata?.age || '',
    sex: caseMetadata?.sex || '',
    examination_date: caseMetadata?.examination_date || '',
    notes: caseMetadata?.notes || '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleContinue = (e) => {
    e.preventDefault();
    // Only store in temporary memory if at least one field has data
    const hasData = Object.values(form).some((v) => v.trim() !== '');
    if (hasData) {
      setCaseMetadata(form);
    } else {
      setCaseMetadata(null);
    }
    navigate('/upload');
  };

  const handleSkip = () => {
    setCaseMetadata(null);
    navigate('/upload');
  };

  return (
    <div className="app-container">
      <Navbar showBack backTo="/dashboard" />

      <main className="main-content">
        <div style={{ maxWidth: '650px', width: '100%' }}>
          <div style={{ marginBottom: '1.5rem' }}>
            <h1 className="card-title">Optional Case Information</h1>
            <p className="card-subtitle">
              You can provide optional research metadata for this examination session. This data is kept strictly in temporary memory and is never passed to the ML model.
            </p>
          </div>

          <div className="alert-banner alert-warning">
            <ShieldAlert size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <strong>Privacy Notice</strong>
              <div>Do not enter real patient identifiers or sensitive personal information.</div>
            </div>
          </div>

          <div className="card">
            <form onSubmit={handleContinue}>
              <div className="form-group">
                <label className="form-label" htmlFor="case_id">
                  Case / Specimen ID (Optional)
                </label>
                <input
                  id="case_id"
                  name="case_id"
                  type="text"
                  className="form-input"
                  placeholder="e.g. DEMO-VAL-01"
                  value={form.case_id}
                  onChange={handleChange}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label className="form-label" htmlFor="age">
                    Subject Age (Optional)
                  </label>
                  <input
                    id="age"
                    name="age"
                    type="text"
                    className="form-input"
                    placeholder="e.g. 52"
                    value={form.age}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="sex">
                    Subject Biological Sex (Optional)
                  </label>
                  <select
                    id="sex"
                    name="sex"
                    className="form-select"
                    value={form.sex}
                    onChange={handleChange}
                  >
                    <option value="">Select (Optional)</option>
                    <option value="Female">Female</option>
                    <option value="Male">Male</option>
                    <option value="Other / Unspecified">Other / Unspecified</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="examination_date">
                  Examination Date (Optional)
                </label>
                <input
                  id="examination_date"
                  name="examination_date"
                  type="date"
                  className="form-input"
                  value={form.examination_date}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="notes">
                  Clinical / Research Notes (Optional)
                </label>
                <textarea
                  id="notes"
                  name="notes"
                  rows={3}
                  className="form-textarea"
                  placeholder="e.g. Anteroposterior projection from validation split"
                  value={form.notes}
                  onChange={handleChange}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2rem' }}>
                <button
                  type="button"
                  onClick={handleSkip}
                  className="btn btn-secondary"
                >
                  Skip Step
                </button>

                <button
                  type="submit"
                  className="btn btn-primary"
                >
                  Proceed to Upload
                  <ArrowRight size={16} />
                </button>
              </div>
            </form>
          </div>
        </div>
      </main>

      <footer className="footer">
        ChestXR Research Application &copy; 2026.
      </footer>
    </div>
  );
}
