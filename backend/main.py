"""
FastAPI Application Entrypoint for ChestXR
Provides endpoints:
  - GET  /health: System readiness and model status
  - POST /predict: Multipart file upload -> Preprocessing -> MobileNetV2 -> Grad-CAM -> Results
"""
import io
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.config import settings
from backend.schemas import HealthResponse, PredictResponse, CaseMetadata
from backend.services.model_service import model_service
from backend.services.xai_service import generate_gradcam_heatmap, overlay_gradcam_on_image
from ml.preprocessing import preprocess_image

# Configure server logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("chestxr.api")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: loads model once on startup."""
    try:
        model_service.load_model()
        logger.info("Application lifespan startup: MobileNetV2 model successfully loaded and ready.")
    except Exception as exc:
        logger.error(f"Failed to load MobileNetV2 model on startup: {exc}", exc_info=True)
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ChestXR API — Pneumonia classification with MobileNetV2 and Grad-CAM explainability",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint: verifies service status and model readiness."""
    is_ready = model_service.is_ready
    return HealthResponse(
        status="ok" if is_ready else "degraded",
        model_loaded=is_ready,
        model_name="MobileNetV2",
        target_layer=settings.GRAD_CAM_LAYER,
        threshold=settings.CLASSIFICATION_THRESHOLD,
        version=settings.VERSION
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_xray(
    file: UploadFile = File(..., description="Uploaded chest X-ray image file (JPG, JPEG, PNG)"),
    metadata: Optional[str] = Form(None, description="Optional JSON-encoded case metadata")
):
    """
    Accepts uploaded chest X-ray image, runs frozen preprocessing,
    executes MobileNetV2 inference, generates Grad-CAM explainability overlay,
    and returns symmetric scores and visualizations.
    """
    # 1. Model readiness validation
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MobileNetV2 model is not loaded or not ready for inference."
        )

    # 2. Filename and extension validation
    filename = file.filename or ""
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
        )

    # 3. Read image bytes into memory (never saved to disk)
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error(f"Error reading uploaded file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded image data."
        )

    # Validate file size
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
    if len(contents) > settings.MAX_FILE_SIZE_BYTES:
        max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed size of {max_mb} MB."
        )

    # 4. Image integrity and dimension validation
    try:
        with Image.open(io.BytesIO(contents)) as pil_img:
            pil_img.verify()
        # Re-open after verify() to inspect dimensions
        with Image.open(io.BytesIO(contents)) as pil_img:
            width, height = pil_img.size
            if width < settings.MIN_IMAGE_DIMENSION or height < settings.MIN_IMAGE_DIMENSION:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image dimensions ({width}x{height}) are too small. Minimum dimension is {settings.MIN_IMAGE_DIMENSION}px."
                )
            if width > settings.MAX_IMAGE_DIMENSION or height > settings.MAX_IMAGE_DIMENSION:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {settings.MAX_IMAGE_DIMENSION}px."
                )
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content could not be decoded as a valid image."
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"Image validation issue: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or malformed image data."
        )

    # 5. Exact reuse of frozen preprocessing pipeline
    t_start = datetime.now()
    try:
        t0 = datetime.now()
        preprocessed_arr = preprocess_image(contents)
        dur_prep = (datetime.now() - t0).total_seconds()
        logger.info(f"[{filename}] Step 1/4 - Preprocessing completed in {dur_prep:.3f}s (shape: {preprocessed_arr.shape})")
    except Exception as exc:
        logger.error(f"Preprocessing error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to preprocess X-ray image using pipeline."
        )

    # 6. Model prediction using MobileNetV2
    try:
        t0 = datetime.now()
        raw_score, prediction, model_score = model_service.predict_tensor(preprocessed_arr)
        dur_infer = (datetime.now() - t0).total_seconds()
        logger.info(f"[{filename}] Step 2/4 - MobileNetV2 inference completed in {dur_infer:.3f}s (prediction: {prediction}, score: {model_score:.4f})")
    except Exception as exc:
        logger.error(f"Model prediction error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed."
        )

    # 7. Grad-CAM generation targeting 'out_relu'
    try:
        t0 = datetime.now()
        heatmap = generate_gradcam_heatmap(preprocessed_arr)
        dur_cam = (datetime.now() - t0).total_seconds()
        logger.info(f"[{filename}] Step 3/4 - Grad-CAM heatmap generated in {dur_cam:.3f}s")
        
        t0 = datetime.now()
        orig_base64, gradcam_base64 = overlay_gradcam_on_image(preprocessed_arr, heatmap)
        dur_overlay = (datetime.now() - t0).total_seconds()
        logger.info(f"[{filename}] Step 4/4 - Visualization overlay generated in {dur_overlay:.3f}s")
    except Exception as exc:
        logger.error(f"Grad-CAM generation error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute Grad-CAM explanation overlay."
        )

    # 8. Parse optional metadata (in-memory only, never persisted)
    case_meta_obj = None
    if metadata:
        try:
            meta_dict = json.loads(metadata)
            case_meta_obj = CaseMetadata(**meta_dict)
        except Exception:
            # Metadata is strictly non-essential, skip if unparseable
            case_meta_obj = None

    timestamp_str = datetime.now(timezone.utc).isoformat()
    total_dur = (datetime.now() - t_start).total_seconds()
    logger.info(f"[{filename}] Pipeline complete. Total server turnaround: {total_dur:.3f}s")

    return PredictResponse(
        prediction=prediction,
        raw_score=round(raw_score, 4),
        model_score=round(model_score, 4),
        model_score_percentage=round(model_score * 100.0, 1),
        threshold=settings.CLASSIFICATION_THRESHOLD,
        model_name="MobileNetV2",
        timestamp=timestamp_str,
        original_image=orig_base64,
        gradcam_image=gradcam_base64,
        heatmap_dimensions=f"{heatmap.shape[0]}x{heatmap.shape[1]}",
        disclaimer=(
            "Educational / Research Use Only. This system does not provide a medical diagnosis or clinical decision. "
            "Model outputs and Grad-CAM visualizations must not be used as a substitute for professional medical evaluation."
        ),
        case_metadata=case_meta_obj
    )
