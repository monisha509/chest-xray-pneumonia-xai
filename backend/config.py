"""
ChestXR Backend Configuration and Environment Settings
"""
import os
from pathlib import Path
from typing import List
from pydantic import BaseModel

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    PROJECT_NAME: str = "ChestXR API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Model configuration
    MODEL_PATH: Path = ROOT_DIR / "models" / "mobilenetv2.keras"
    CLASSIFICATION_THRESHOLD: float = 0.50
    GRAD_CAM_LAYER: str = "out_relu"
    
    # File upload validation
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    ALLOWED_CONTENT_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]
    MIN_IMAGE_DIMENSION: int = 32
    MAX_IMAGE_DIMENSION: int = 6000
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

settings = Settings()

