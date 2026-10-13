/**
 * ChestXR Centralized API Client
 */
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }
  return response.json();
}

export async function predictXRay(imageFile, caseMetadata = null, { timeoutMs = 60000, signal } = {}) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  if (caseMetadata && Object.keys(caseMetadata).length > 0) {
    formData.append('metadata', JSON.stringify(caseMetadata));
  }

  // Set up timeout controller
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort(new Error(`Inference request timed out after ${timeoutMs / 1000} seconds. Please verify the backend server status.`));
  }, timeoutMs);

  // If external signal is provided, link it
  if (signal) {
    signal.addEventListener('abort', () => controller.abort(signal.reason));
  }

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorDetail = 'Inference request failed';
      try {
        const errJson = await response.json();
        if (errJson.detail) {
          errorDetail = errJson.detail;
        }
      } catch {
        errorDetail = `Request failed with status ${response.status} (${response.statusText})`;
      }
      throw new Error(errorDetail);
    }

    return await response.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error(err.message || 'The request timed out or was cancelled.');
    }
    if (err.message && err.message.includes('Failed to fetch')) {
      throw new Error('Cannot connect to ChestXR backend at http://localhost:8000. Ensure the FastAPI service is running.');
    }
    throw err;
  }
}


