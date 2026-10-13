"""
Explainable AI (XAI) Service: Grad-CAM Visualization
Computes gradient-weighted class activation mapping for the frozen MobileNetV2 architecture
targeting the 'out_relu' layer for the Pneumonia output.
"""
import io
import base64
import logging
from typing import Tuple
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

from backend.services.model_service import model_service
from backend.config import settings

logger = logging.getLogger("chestxr.xai")


def generate_gradcam_heatmap(
    img_tensor: np.ndarray,
    pred_index: int = 0,
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap array normalized in [0.0, 1.0].
    
    Args:
        img_tensor: Preprocessed numpy array of shape (224, 224, 3) with values in [0, 255].
        pred_index: Output index (0 for single sigmoid output representing Pneumonia).
        
    Returns:
        heatmap: 2D numpy array of shape (7, 7) or target spatial dims, values in [0.0, 1.0].
    """
    grad_model = model_service.grad_model
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_tensor, axis=0))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        # Class 1 (Pneumonia) activation
        loss = predictions[:, pred_index]

    # Compute gradient of the score with respect to feature map activations
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Failed to compute gradients for Grad-CAM. Gradient tape returned None.")

    # Global average pooling of gradients: mean across spatial dimensions (0, 1, 2)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map channel by its corresponding pooled gradient
    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to focus strictly on features having a positive influence on the target class
    heatmap = tf.maximum(heatmap, 0.0)
    
    # Normalize between 0.0 and 1.0 safely
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0.0:
        heatmap = heatmap / (max_val + 1e-10)
    else:
        # If all gradients or activations are zero, keep a zero heatmap
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy()


def overlay_gradcam_on_image(
    original_img_arr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Tuple[str, str]:
    """
    Combines the original preprocessed image and Grad-CAM heatmap into a colored overlay.
    
    Args:
        original_img_arr: Preprocessed image array (224, 224, 3) in [0, 255].
        heatmap: 2D heatmap array normalized in [0, 1].
        alpha: Overlay blend factor (0.0 to 1.0).
        colormap_name: Matplotlib colormap (default: 'jet').
        
    Returns:
        (original_base64_png, overlay_base64_png): Both encoded as base64 data URIs.
    """
    # 1. Prepare clean uint8 version of original image
    uint8_orig = np.clip(original_img_arr, 0, 255).astype(np.uint8)
    pil_orig = Image.fromarray(uint8_orig)

    # 2. Resize heatmap to match image dimensions (224, 224) using bicubic/bilinear interpolation
    pil_heatmap = Image.fromarray(np.uint8(255 * heatmap))
    pil_heatmap_resized = pil_heatmap.resize(
        (original_img_arr.shape[1], original_img_arr.shape[0]),
        resample=Image.Resampling.BILINEAR
    )
    resized_heatmap_norm = np.array(pil_heatmap_resized, dtype=np.float32) / 255.0

    # 3. Apply color map
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    colored_heatmap = cmap(resized_heatmap_norm)  # RGBA in [0, 1]
    colored_heatmap_rgb = np.uint8(255 * colored_heatmap[..., :3])  # RGB in uint8

    # 4. Superimpose heatmap on the original grayscale/stacked image
    overlay_arr = (1.0 - alpha) * uint8_orig.astype(np.float32) + alpha * colored_heatmap_rgb.astype(np.float32)
    overlay_arr = np.clip(overlay_arr, 0, 255).astype(np.uint8)
    pil_overlay = Image.fromarray(overlay_arr)

    # 5. Convert both to base64 PNG data strings
    def _to_base64_png(pil_im: Image.Image) -> str:
        buf = io.BytesIO()
        pil_im.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    original_b64 = _to_base64_png(pil_orig)
    overlay_b64 = _to_base64_png(pil_overlay)

    return original_b64, overlay_b64
