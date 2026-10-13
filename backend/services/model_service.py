"""
MobileNetV2 Inference and Model Management Service
Loads the frozen MobileNetV2 model once during application startup.
Reuses ml.preprocessing pipeline exactly without modifications.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
import keras

from backend.config import settings
from ml.preprocessing import preprocess_image_bytes

logger = logging.getLogger("chestxr.model")

class ModelService:
    def __init__(self):
        self._model: Optional[keras.Model] = None
        self._grad_model: Optional[keras.Model] = None
        self._is_ready: bool = False

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load the frozen MobileNetV2 model once.
        Also constructs the Grad-CAM multi-output model graph using the existing layers.
        """
        path = model_path or settings.MODEL_PATH
        if not Path(path).is_file():
            raise FileNotFoundError(f"Selected model weights file not found at: {path}")

        logger.info(f"Loading frozen MobileNetV2 model from {path}...")
        self._model = keras.models.load_model(str(path))
        
        # Verify layer existence for Grad-CAM
        layer_names = [l.name for l in self._model.layers]
        if settings.GRAD_CAM_LAYER not in layer_names:
            raise ValueError(f"Grad-CAM target layer '{settings.GRAD_CAM_LAYER}' not found in loaded model.")
        
        target_layer = self._model.get_layer(settings.GRAD_CAM_LAYER)
        
        # Construct Grad-CAM graph: inputs -> [conv_layer.output, model.output]
        self._grad_model = keras.Model(
            inputs=self._model.inputs,
            outputs=[target_layer.output, self._model.output]
        )
        
        # Warmup model graph to avoid JIT/tracing latency on first user request
        try:
            logger.info("Executing initial model warmup pass...")
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            self._model.predict(dummy_input, verbose=0)
            with tf.GradientTape() as tape:
                conv_out, preds = self._grad_model(tf.convert_to_tensor(dummy_input))
                loss = preds[:, 0]
            tape.gradient(loss, conv_out)
            logger.info("Model and Grad-CAM warmup complete.")
        except Exception as warmup_err:
            logger.warning(f"Model warmup encountered non-critical notice: {warmup_err}")

        self._is_ready = True
        logger.info("MobileNetV2 model and Grad-CAM graph initialized successfully.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._model is not None and self._grad_model is not None

    @property
    def model(self) -> keras.Model:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded or ready.")
        return self._model

    @property
    def grad_model(self) -> keras.Model:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded or ready.")
        return self._grad_model

    def predict_tensor(self, img_tensor: np.ndarray) -> Tuple[float, str, float]:
        """
        Given a preprocessed image tensor (224, 224, 3) in [0, 255]:
        Returns (raw_score, prediction_label, display_model_score).
        
        Operating threshold:
          score >= 0.50 -> Pneumonia
          score < 0.50  -> Normal
          
        Model score:
          Pneumonia: raw_score
          Normal: 1.0 - raw_score
        """
        if not self.is_ready:
            raise RuntimeError("Model is not loaded.")
        
        batch_input = np.expand_dims(img_tensor, axis=0)  # (1, 224, 224, 3)
        preds = self._model.predict(batch_input, verbose=0)
        raw_score = float(preds[0][0])
        
        if raw_score >= settings.CLASSIFICATION_THRESHOLD:
            prediction = "Pneumonia"
            model_score = raw_score
        else:
            prediction = "Normal"
            model_score = 1.0 - raw_score
            
        return raw_score, prediction, model_score


# Singleton instance
model_service = ModelService()

