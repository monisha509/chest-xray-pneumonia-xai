"""
Backend Automated Test Suite for ChestXR
Tests coverage:
  - /health endpoint & model readiness
  - /predict with valid Normal demo image
  - /predict with valid Pneumonia demo image
  - /predict with different image dimensions (handling arbitrary inputs)
  - /predict with invalid image content (corrupt file)
  - /predict with unsupported file extensions
  - /predict with empty file upload
  - Frozen preprocessing verification
  - Response schema integrity
  - Grad-CAM heatmap generation and overlay
  - Decision threshold logic (0.50 cutoff verification)
"""
import io
import json
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.model_service import model_service
from backend.services.xai_service import generate_gradcam_heatmap, overlay_gradcam_on_image
from backend.config import settings
from ml.preprocessing import preprocess_image, preprocess_image_bytes

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def init_model():
    """Ensure model is loaded before running tests."""
    if not model_service.is_ready:
        model_service.load_model()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    """Test /health endpoint returns 200 and model_loaded: true."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "MobileNetV2"
    assert data["target_layer"] == "out_relu"
    assert data["threshold"] == 0.50


def test_predict_normal_sample(client):
    """Test /predict with demo_samples/normal_01.jpg produces Normal prediction."""
    sample_path = PROJECT_ROOT / "demo_samples" / "normal_01.jpg"
    assert sample_path.exists()
    
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict",
        files={"file": ("normal_01.jpg", file_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Normal"
    assert data["raw_score"] < 0.50
    assert data["model_score"] > 0.50  # symmetric: 1.0 - raw_score
    assert data["model_name"] == "MobileNetV2"
    assert data["original_image"].startswith("data:image/png;base64,")
    assert data["gradcam_image"].startswith("data:image/png;base64,")
    assert "Educational / Research Use Only" in data["disclaimer"]


def test_predict_pneumonia_sample(client):
    """Test /predict with demo_samples/pneumonia_01.jpg produces Pneumonia prediction."""
    sample_path = PROJECT_ROOT / "demo_samples" / "pneumonia_01.jpg"
    assert sample_path.exists()
    
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    metadata = json.dumps({"case_id": "CASE-101", "age": "45", "sex": "M"})

    response = client.post(
        "/predict",
        files={"file": ("pneumonia_01.jpg", file_bytes, "image/jpeg")},
        data={"metadata": metadata}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Pneumonia"
    assert data["raw_score"] >= 0.50
    assert data["model_score"] == pytest.approx(data["raw_score"], abs=1e-4)
    assert data["case_metadata"]["case_id"] == "CASE-101"
    assert data["case_metadata"]["age"] == "45"


def test_predict_different_dimensions(client):
    """Test that arbitrary image dimensions are accepted, resized cleanly by preprocessing."""
    # Create an arbitrary 512x384 image in memory
    img = Image.new("RGB", (512, 384), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("custom_size.png", buf.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Normal", "Pneumonia"]


def test_predict_empty_file_rejected(client):
    """Test empty file upload is rejected with 400 Bad Request."""
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_invalid_corrupted_file_rejected(client):
    """Test invalid non-image file is rejected with 400 Bad Request."""
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", b"NOT_AN_IMAGE_DATA_12345", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "valid image" in response.json()["detail"].lower() or "corrupted" in response.json()["detail"].lower()


def test_predict_unsupported_extension_rejected(client):
    """Test unsupported extension (.txt, .pdf) is rejected."""
    response = client.post(
        "/predict",
        files={"file": ("report.pdf", b"%PDF-1.4...", "application/pdf")}
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_frozen_preprocessing_reuse():
    """Verify that preprocess_image produces (224, 224, 3) float32 in [0, 255]."""
    sample_path = PROJECT_ROOT / "demo_samples" / "normal_01.jpg"
    arr = preprocess_image(sample_path)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype.name == "float32"
    assert 0.0 <= arr.min() <= 255.0
    assert 0.0 <= arr.max() <= 255.0
    # Confirm 3 channels are identical copies of grayscale
    assert (arr[:, :, 0] == arr[:, :, 1]).all()
    assert (arr[:, :, 1] == arr[:, :, 2]).all()


def test_gradcam_generation():
    """Verify Grad-CAM produces a valid 2D normalized heatmap and base64 overlay."""
    sample_path = PROJECT_ROOT / "demo_samples" / "pneumonia_01.jpg"
    arr = preprocess_image(sample_path)
    heatmap = generate_gradcam_heatmap(arr)
    
    assert heatmap.shape == (7, 7)
    assert 0.0 <= heatmap.min() <= 1.0
    assert 0.0 <= heatmap.max() <= 1.0
    
    orig_b64, overlay_b64 = overlay_gradcam_on_image(arr, heatmap)
    assert orig_b64.startswith("data:image/png;base64,")
    assert overlay_b64.startswith("data:image/png;base64,")


def test_threshold_decision_logic():
    """Unit test decision logic at exactly 0.50 cutoff."""
    # Test >= 0.50 -> Pneumonia
    raw_p = 0.50
    pred_label_p = "Pneumonia" if raw_p >= settings.CLASSIFICATION_THRESHOLD else "Normal"
    assert pred_label_p == "Pneumonia"

    # Test < 0.50 -> Normal
    raw_n = 0.4999
    pred_label_n = "Pneumonia" if raw_n >= settings.CLASSIFICATION_THRESHOLD else "Normal"
    assert pred_label_n == "Normal"

