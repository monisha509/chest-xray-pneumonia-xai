"""
ChestXR — Preprocessing Pipeline (Member 1)
===========================================
Shared image preprocessing, memory-optimized tf.data input pipelines,
training-only augmentation, and class weight calculation.

Dataset split is frozen as splits-v1.
No raw images or split manifests are modified.
"""

import csv
import io
import os
from pathlib import Path
from typing import Dict, Tuple, Union

# Suppress verbose TensorFlow C++ informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image

# ── 1. Label Mapping ──────────────────────────────────────────────────────────
# NORMAL = 0, PNEUMONIA = 1
# Single sigmoid output represents model probability for PNEUMONIA.
LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 2. Shared Image Preprocessing ─────────────────────────────────────────────
def preprocess_image(
    image_input: Union[str, Path, bytes, io.BytesIO],
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Shared preprocessing function for both training and FastAPI inference.

    Steps:
      1. Open image with Pillow.
      2. Fully load pixels into memory inside try/finally block before closing.
      3. Convert to single-channel grayscale ('L') to eliminate color shortcuts uniformly.
      4. Resize to target dimensions (224, 224) using bilinear interpolation.
      5. Convert to float32 numpy array.
      6. Stack into 3 identical channels (224, 224, 3).
      7. Keep pixel values in raw 0–255 range.

    Model-specific scaling is applied inside model architectures:
      - Baseline CNN : Rescaling(1./255)
      - MobileNetV2  : mobilenet_v2.preprocess_input
    """
    # Open from file path or in-memory byte buffer
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        pil_img = Image.open(img_path)
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, io.BytesIO):
        pil_img = Image.open(image_input)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    try:
        # Fully load image data before any closing
        pil_img.load()

        # Convert to grayscale ('L')
        gray_img = pil_img.convert("L")

        # Resize to fixed input dimensions
        resized_img = gray_img.resize(target_size, Image.Resampling.BILINEAR)

        # Convert to float32 array in [0, 255]
        arr_2d = np.asarray(resized_img, dtype=np.float32)

        # Stack into 3 identical channels: (224, 224, 3)
        arr_3d = np.stack([arr_2d, arr_2d, arr_2d], axis=-1)

        # Explicit safety validation checks (replacing asserts with ValueError)
        if arr_3d.shape != (target_size[0], target_size[1], 3):
            raise ValueError(
                f"Invalid preprocessed image shape: {arr_3d.shape}, expected {(target_size[0], target_size[1], 3)}"
            )
        if arr_3d.dtype != np.float32:
            raise ValueError(f"Invalid preprocessed image dtype: {arr_3d.dtype}, expected float32")
        if not np.all(np.isfinite(arr_3d)):
            raise ValueError("Preprocessed image contains non-finite values (NaN or Inf)")
        if arr_3d.min() < 0.0 or arr_3d.max() > 255.0:
            raise ValueError(
                f"Preprocessed image values out of bounds [0, 255]: min={arr_3d.min()}, max={arr_3d.max()}"
            )

        return arr_3d
    finally:
        pil_img.close()


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convenience wrapper for API inference taking raw uploaded bytes."""
    return preprocess_image(image_bytes)


# ── 3. Class Weights Calculation ──────────────────────────────────────────────
def compute_class_weights(csv_path: Union[str, Path] = "data/splits/train.csv") -> Dict[int, float]:
    """
    Compute balanced class weights from training manifest only:
      weight = total_samples / (num_classes * class_count)
    """
    resolved_path = Path(csv_path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    counts = {0: 0, 1: 0}
    with open(resolved_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            cls_name = row["class_name"].strip().upper()
            filepath = row["filepath"]

            # Validate that CSV label matches folder name / class_name
            if cls_name == "NORMAL" and label != 0:
                raise ValueError(f"Label mismatch in {csv_path} for {filepath}: class is NORMAL but label is {label}")
            if cls_name == "PNEUMONIA" and label != 1:
                raise ValueError(f"Label mismatch in {csv_path} for {filepath}: class is PNEUMONIA but label is {label}")
            if label not in (0, 1):
                raise ValueError(f"Invalid label {label} in {csv_path}. Must be 0 or 1.")

            counts[label] += 1

    total_samples = sum(counts.values())
    num_classes = len(counts)

    weights = {}
    for label, count in counts.items():
        weights[label] = round(total_samples / (num_classes * count), 4)

    return weights


# ── 4. Training-Only Augmentation ─────────────────────────────────────────────
def get_training_augmentation():
    """
    Mild, anatomically safe data augmentation for training only:
      - Mild rotation (+/- 2% = +/- 7.2 degrees) with fill_mode='constant'
      - Mild zoom (+/- 5%) with fill_mode='constant'
      - Mild brightness change (+/- 10%)
    Note: Horizontal flip is strictly EXCLUDED to protect anatomical lateral orientation.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=0.02, fill_mode="constant"),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode="constant"
            ),
            tf.keras.layers.RandomBrightness(factor=0.1, value_range=(0.0, 255.0)),
        ],
        name="training_augmentation",
    )


# ── 5. Memory-Optimized TF.Data Pipeline ──────────────────────────────────────
def _decode_grayscale_uint8(filepath_bytes: bytes) -> np.ndarray:
    """
    Load image from path, fully decode, convert to grayscale, resize to (224, 224),
    and return as uint8 array with shape (224, 224, 1).
    This keeps the cached representation at only 1 byte per pixel.
    """
    path_str = filepath_bytes.decode("utf-8")
    img_path = Path(path_str)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / img_path
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found at path: {img_path}")

    with Image.open(img_path) as pil_img:
        pil_img.load()
        gray_img = pil_img.convert("L")
        resized_img = gray_img.resize((224, 224), Image.Resampling.BILINEAR)
        arr_2d = np.asarray(resized_img, dtype=np.uint8)
        return arr_2d[:, :, np.newaxis]


def load_dataset(
    csv_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
    augment: bool = False,
):
    """
    Build a memory-optimized, high-performance tf.data.Dataset from a split CSV manifest.

    Pipeline execution order:
      1. Decode & resize image to uint8 grayscale (224, 224, 1)
      2. Cache ONLY these compact uint8 grayscale images (1 byte/px vs 12 bytes/px)
      3. If shuffle: Shuffle AFTER cache with reshuffle_each_iteration=True
      4. Expand grayscale uint8 to 3 identical float32 channels (224, 224, 3)
      5. Batch
      6. If augment: Apply training augmentation AFTER cache/batch so transforms are fresh each epoch
      7. Prefetch at the end
    """
    import tensorflow as tf

    resolved_path = Path(csv_path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    filepaths = []
    labels = []

    with open(resolved_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = int(row["label"])
            cls_name = row["class_name"].strip().upper()
            filepath = row["filepath"]

            # Validate that CSV label matches folder name / class_name
            if cls_name == "NORMAL" and lbl != 0:
                raise ValueError(f"Label mismatch in {csv_path} for {filepath}: class is NORMAL but label is {lbl}")
            if cls_name == "PNEUMONIA" and lbl != 1:
                raise ValueError(f"Label mismatch in {csv_path} for {filepath}: class is PNEUMONIA but label is {lbl}")
            if lbl not in (0, 1):
                raise ValueError(f"Invalid label {lbl} in {csv_path}. Must be 0 or 1.")

            filepaths.append(filepath)
            labels.append(lbl)

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # Step 1: Decode & resize image to compact uint8 grayscale (224, 224, 1)
    def _map_decode_uint8(path_tensor, label_tensor):
        gray_u8 = tf.numpy_function(_decode_grayscale_uint8, [path_tensor], tf.uint8)
        gray_u8.set_shape((224, 224, 1))
        label = tf.cast(label_tensor, tf.float32)
        label.set_shape(())
        return gray_u8, label

    dataset = dataset.map(_map_decode_uint8, num_parallel_calls=tf.data.AUTOTUNE)

    # Step 2: Cache ONLY the compact uint8 grayscale images
    dataset = dataset.cache()

    # Step 3: Shuffle AFTER cache with reshuffle_each_iteration=True (for training only)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), seed=seed, reshuffle_each_iteration=True)

    # Step 4: Expand grayscale uint8 to 3 identical float32 channels
    def _expand_channels(gray_u8, label):
        gray_f32 = tf.cast(gray_u8, tf.float32)
        rgb_f32 = tf.repeat(gray_f32, 3, axis=-1)
        rgb_f32.set_shape((224, 224, 3))
        return rgb_f32, label

    dataset = dataset.map(_expand_channels, num_parallel_calls=tf.data.AUTOTUNE)

    # Step 5: Batch
    dataset = dataset.batch(batch_size)

    # Step 6: Apply training augmentation AFTER cache/batch so transforms are fresh each epoch
    if augment:
        aug_layer = get_training_augmentation()
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Step 7: Prefetch at the end
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# ── 6. Verification Tests ─────────────────────────────────────────────────────
def run_tests():
    """Run all preprocessing unit tests including the two newly requested verification tests."""
    print("=" * 70)
    print("Running Preprocessing Tests...")
    print("=" * 70)

    # 1. Class weights
    weights = compute_class_weights("data/splits/train.csv")
    print(f"Class weights (train.csv): NORMAL (0) = {weights[0]:.4f}, PNEUMONIA (1) = {weights[1]:.4f}")
    if abs(weights[0] - 1.8260) > 0.01:
        raise ValueError(f"Unexpected NORMAL weight: {weights[0]}")
    if abs(weights[1] - 0.6885) > 0.01:
        raise ValueError(f"Unexpected PNEUMONIA weight: {weights[1]}")

    # 2. Test shared preprocessing on real image
    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f))
    sample_path = first_row["filepath"]
    print(f"\nTesting sample image: {sample_path}")

    img_arr = preprocess_image(sample_path)
    print(f"  Output shape : {img_arr.shape}")
    print(f"  Output dtype : {img_arr.dtype}")
    print(f"  Min value    : {img_arr.min():.2f}")
    print(f"  Max value    : {img_arr.max():.2f}")
    print(f"  All finite   : {np.all(np.isfinite(img_arr))}")

    if img_arr.shape != (224, 224, 3):
        raise ValueError(f"Shape test failed: {img_arr.shape}")
    if img_arr.dtype != np.float32:
        raise ValueError(f"Dtype test failed: {img_arr.dtype}")
    if img_arr.min() < 0.0:
        raise ValueError(f"Min test failed: {img_arr.min()}")
    if img_arr.max() > 255.0:
        raise ValueError(f"Max test failed: {img_arr.max()}")
    if not np.all(np.isfinite(img_arr)):
        raise ValueError("Finite test failed")

    # Bytes test (API compatibility)
    with open(PROJECT_ROOT / sample_path, "rb") as f:
        raw_bytes = f.read()
    bytes_arr = preprocess_image_bytes(raw_bytes)
    if not np.array_equal(img_arr, bytes_arr):
        raise ValueError("Bytes preprocessing does not match path preprocessing")
    print("  Bytes preprocessing (API compatibility): PASS")

    # Channel uniformity check (all 3 channels must be identical copies of grayscale)
    if not np.array_equal(img_arr[:, :, 0], img_arr[:, :, 1]):
        raise ValueError("Channel 0 and 1 mismatch")
    if not np.array_equal(img_arr[:, :, 1], img_arr[:, :, 2]):
        raise ValueError("Channel 1 and 2 mismatch")
    print("  3-channel grayscale replication: PASS")

    # 3. Newly Requested Test 1:
    # load_dataset(val.csv, batch_size=4, shuffle=False, augment=False)
    # first tensor must exactly equal preprocess_image(first validation filepath)
    print("\n[New Test 1] Testing exact tensor equality against first validation image...")
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"
    with open(val_csv, "r", encoding="utf-8") as f:
        first_val_row = next(csv.DictReader(f))
    first_val_path = first_val_row["filepath"]
    expected_tensor = preprocess_image(first_val_path)

    val_ds = load_dataset(val_csv, batch_size=4, shuffle=False, augment=False)
    for batch_x, batch_y in val_ds.take(1):
        actual_first_tensor = batch_x[0].numpy()
        break

    if not np.array_equal(actual_first_tensor, expected_tensor):
        diff = np.max(np.abs(actual_first_tensor - expected_tensor))
        raise ValueError(f"Exact tensor equality failed! Max difference: {diff}")
    print(f"  Exact tensor match with {first_val_path}: PASS")

    # 4. Newly Requested Test 2:
    # Verify shuffle and augmentation are not frozen across epochs.
    print("\n[New Test 2] Verifying shuffle and augmentation are not frozen across epochs...")
    train_ds = load_dataset(train_csv, batch_size=4, shuffle=True, seed=42, augment=True)
    
    # Iterate across 2 epochs
    epoch1_batch0 = None
    for batch_x, _ in train_ds.take(1):
        epoch1_batch0 = batch_x.numpy()

    epoch2_batch0 = None
    for batch_x, _ in train_ds.take(1):
        epoch2_batch0 = batch_x.numpy()

    if np.array_equal(epoch1_batch0, epoch2_batch0):
        raise ValueError("Epoch 1 batch 0 and Epoch 2 batch 0 are identical! Augmentation/shuffle is frozen.")
    print("  Fresh shuffle and augmentation across epochs: PASS")

    print("\nAll preprocessing tests PASSED successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
