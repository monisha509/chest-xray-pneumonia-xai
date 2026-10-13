import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { AnalysisProvider } from './context/AnalysisContext';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import CaseInfoPage from './pages/CaseInfoPage';
import UploadPage from './pages/UploadPage';
import ProcessingPage from './pages/ProcessingPage';
import ResultsPage from './pages/ResultsPage';

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AnalysisProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <DashboardPage />
                </ProtectedRoute>
              }
            />
            
            <Route
              path="/case-info"
              element={
                <ProtectedRoute>
                  <CaseInfoPage />
                </ProtectedRoute>
              }
            />
            
            <Route
              path="/upload"
              element={
                <ProtectedRoute>
                  <UploadPage />
                </ProtectedRoute>
              }
            />
            
            <Route
              path="/processing"
              element={
                <ProtectedRoute>
                  <ProcessingPage />
                </ProtectedRoute>
              }
            />
            
            <Route
              path="/results"
              element={
                <ProtectedRoute>
                  <ResultsPage />
                </ProtectedRoute>
              }
            />
            
            {/* Catch-all redirect to login or dashboard */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </AnalysisProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
