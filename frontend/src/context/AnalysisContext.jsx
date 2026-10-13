import React, { createContext, useContext, useState } from 'react';

const AnalysisContext = createContext(null);

export function AnalysisProvider({ children }) {
  // Case metadata: strictly kept in memory, not localStorage
  const [caseMetadata, setCaseMetadata] = useState(null);
  
  // Selected/uploaded image file and preview URL
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  
  // Prediction result from backend
  const [predictionResult, setPredictionResult] = useState(null);

  const resetAnalysis = () => {
    setCaseMetadata(null);
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setSelectedImage(null);
    setImagePreviewUrl(null);
    setPredictionResult(null);
  };

  const setImage = (file) => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setSelectedImage(file);
    if (file) {
      setImagePreviewUrl(URL.createObjectURL(file));
    } else {
      setImagePreviewUrl(null);
    }
  };

  return (
    <AnalysisContext.Provider
      value={{
        caseMetadata,
        setCaseMetadata,
        selectedImage,
        imagePreviewUrl,
        setImage,
        predictionResult,
        setPredictionResult,
        resetAnalysis,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
}

