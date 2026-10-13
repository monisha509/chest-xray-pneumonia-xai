"""
Pydantic Schemas for API Requests and Responses
"""
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('ok' or 'degraded')")
    model_loaded: bool = Field(..., description="Whether MobileNetV2 is loaded and ready")
    model_name: str = Field("MobileNetV2", description="Name of loaded model")
    target_layer: str = Field("out_relu", description="Grad-CAM target layer")
    threshold: float = Field(0.50, description="Frozen classification threshold")
    version: str = Field("1.0.0", description="API version")


class CaseMetadata(BaseModel):
    case_id: Optional[str] = None
    age: Optional[str] = None
    sex: Optional[str] = None
    examination_date: Optional[str] = None
    notes: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Class name: 'Normal' or 'Pneumonia'")
    raw_score: float = Field(..., description="Raw sigmoid output value [0.0, 1.0]")
    model_score: float = Field(..., description="Model score formatted symmetrically for the predicted class [0.0, 1.0]")
    model_score_percentage: float = Field(..., description="Model score expressed as percentage (0-100%)")
    threshold: float = Field(0.50, description="Operating threshold used for decision")
    model_name: str = Field("MobileNetV2", description="Architecture identifier")
    timestamp: str = Field(..., description="ISO 8601 prediction timestamp")
    original_image: str = Field(..., description="Base64-encoded preprocessed/displayable original X-ray image (PNG)")
    gradcam_image: str = Field(..., description="Base64-encoded Grad-CAM heatmap overlay (PNG)")
    heatmap_dimensions: str = Field(..., description="Dimensions of feature map and overlay")
    disclaimer: str = Field(
        "Educational / Research Use Only. This system does not provide a medical diagnosis or clinical decision. "
        "Model outputs and Grad-CAM visualizations must not be used as a substitute for professional medical evaluation."
    )
    case_metadata: Optional[CaseMetadata] = None

