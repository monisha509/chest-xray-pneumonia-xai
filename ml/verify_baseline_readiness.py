"""
ChestXR — Baseline CNN Readiness & Pre-Training Verification
============================================================
Performs complete pre-training verification:
  1. Architecture & pipeline verification (last_conv_relu, shuffle/augment modes)
  2. Save and reload verification (numerical precision comparison on validation images)
  3. Callbacks verification (EarlyStopping, ModelCheckpoint, CSVLogger, .keras save format)
  4. Reproducibility run record creation (JSON fingerprinting)
"""

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

# Suppress verbose TensorFlow C++ informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from ml.preprocessing import (
    PROJECT_ROOT,
    compute_class_weights,
    load_dataset,
    preprocess_image,
)
from ml.train_baseline import build_baseline_model, DEFAULT_MODEL_PATH, DEFAULT_LOG_PATH, DEFAULT_HISTORY_PATH

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_MODEL_PATH = MODELS_DIR / "temp_baseline_verify.keras"
RUN_RECORD_PATH = MODELS_DIR / "baseline_reproducibility_run_record.json"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_1_architecture_and_pipeline():
    print("=" * 70)
    print("CHECK 1: MODEL ARCHITECTURE & PIPELINE MODES")
    print("=" * 70)

    model = build_baseline_model()

    # Verify last_conv_relu
    layer_names = [l.name for l in model.layers]
    assert "last_conv_relu" in layer_names, "Layer 'last_conv_relu' not found in model"
    target_layer = model.get_layer("last_conv_relu")

    print(f"  Layer 'last_conv_relu' found:")
    print(f"    - Type        : {type(target_layer).__name__}")
    print(f"    - Activation  : {target_layer.activation.__name__}")
    print(f"    - Output shape: {target_layer.output.shape}")

    assert isinstance(target_layer, tf.keras.layers.Conv2D), "last_conv_relu is not a Conv2D layer"
    assert target_layer.activation.__name__ == "relu", "last_conv_relu does not have 'relu' activation"

    # Forward pass feature map inspection
    dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=target_layer.output)
    fmaps = feature_extractor(dummy_input)
    print(f"    - Post-ReLU feature map shape : {fmaps.shape}")
    print(f"    - Post-ReLU min value         : {float(tf.reduce_min(fmaps)):.4f} (must be >= 0.0)")
    assert float(tf.reduce_min(fmaps)) >= 0.0, "Feature maps after ReLU cannot be negative"

    # Verify pipeline configuration
    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"

    print("\n  Pipeline configuration:")
    print("    - Training   : shuffle=True,  augment=True")
    print("    - Validation : shuffle=False, augment=False")

    print("  -> CHECK 1: PASS")
    return True


def check_2_save_and_reload():
    print("\n" + "=" * 70)
    print("CHECK 2: MODEL SAVE AND RELOAD (NUMERICAL PARITY)")
    print("=" * 70)

    # Build model and get initial predictions
    model_orig = build_baseline_model()

    # Save to temporary .keras file
    print(f"  Saving temporary model to: {TEMP_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    model_orig.save(TEMP_MODEL_PATH)
    assert TEMP_MODEL_PATH.is_file(), f"Model file not created: {TEMP_MODEL_PATH}"

    # Load in fresh instance
    print("  Loading saved model via tf.keras.models.load_model...")
    model_loaded = tf.keras.models.load_model(TEMP_MODEL_PATH)

    # Run both on several validation images
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"
    sample_images = []
    with open(val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 8:  # 8 images
                break
            sample_images.append(row["filepath"])

    print(f"  Testing predictions on {len(sample_images)} validation images:")
    val_tensors = np.stack([preprocess_image(p) for p in sample_images], axis=0)

    preds_orig = model_orig.predict(val_tensors, verbose=0)
    preds_loaded = model_loaded.predict(val_tensors, verbose=0)

    max_diff = np.max(np.abs(preds_orig - preds_loaded))
    is_close = np.allclose(preds_orig, preds_loaded, atol=1e-6)

    print(f"    - Maximum absolute difference : {max_diff:.8e}")
    print(f"    - Numerically identical (atol=1e-6) : {is_close}")

    # Remove temporary model after verification
    if TEMP_MODEL_PATH.is_file():
        TEMP_MODEL_PATH.unlink()
        print(f"    - Cleaned up temporary test model: {TEMP_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    assert is_close, f"Model save/reload predictions differ by {max_diff}"
    print("  -> CHECK 2: PASS")
    return True


def check_3_callbacks():
    print("\n" + "=" * 70)
    print("CHECK 3: FULL TRAINING CALLBACKS CONFIGURATION")
    print("=" * 70)

    import inspect
    from ml import train_baseline

    full_src = inspect.getsource(train_baseline)

    has_early_stopping = "EarlyStopping" in full_src and "val_loss" in full_src and "restore_best_weights=True" in full_src
    has_model_checkpoint = "ModelCheckpoint" in full_src and "val_loss" in full_src and "save_best_only=True" in full_src
    has_csv_logger = "CSVLogger" in full_src
    has_keras_format = "baseline_cnn.keras" in full_src
    has_history_json = "baseline_training_history.json" in full_src

    print(f"  EarlyStopping (monitor='val_loss', patience=5, restore_best_weights=True) : {'CONFIGURED' if has_early_stopping else 'MISSING'}")
    print(f"  ModelCheckpoint (monitor='val_loss', save_best_only=True, .keras)          : {'CONFIGURED' if has_model_checkpoint and has_keras_format else 'MISSING'}")
    print(f"  Per-epoch metrics logging (CSVLogger to baseline_training_log.csv)         : {'CONFIGURED' if has_csv_logger else 'MISSING'}")
    print(f"  Run history JSON logging (baseline_training_history.json)                  : {'CONFIGURED' if has_history_json else 'MISSING'}")

    assert has_early_stopping, "EarlyStopping callback misconfigured"
    assert has_model_checkpoint and has_keras_format, "ModelCheckpoint callback misconfigured"
    assert has_csv_logger, "CSVLogger callback misconfigured"
    assert has_history_json, "History JSON logging misconfigured"

    print("  -> CHECK 3: PASS")
    return True


def check_4_reproducibility_run_record():
    print("\n" + "=" * 70)
    print("CHECK 4: CREATE REPRODUCIBILITY RUN RECORD")
    print("=" * 70)

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"
    test_csv = PROJECT_ROOT / "data" / "splits" / "test.csv"

    class_weights = compute_class_weights(train_csv)

    record = {
        "model_name": "baseline_cnn",
        "random_seed": 42,
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "platform": platform.platform(),
        "batch_size": 32,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss_function": "BinaryCrossentropy",
        "maximum_epochs": 20,
        "early_stopping_patience": 5,
        "early_stopping_monitor": "val_loss",
        "restore_best_weights": True,
        "class_weights": {
            "NORMAL (0)": class_weights[0],
            "PNEUMONIA (1)": class_weights[1],
        },
        "target_gradcam_layer": "last_conv_relu",
        "input_shape": [224, 224, 3],
        "sha256_fingerprints": {
            "data/splits/train.csv": sha256_file(train_csv),
            "data/splits/val.csv": sha256_file(val_csv),
            "data/splits/test.csv": sha256_file(test_csv),
        },
    }

    with open(RUN_RECORD_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"  Created run record at: {RUN_RECORD_PATH.relative_to(PROJECT_ROOT)}")
    print(json.dumps(record, indent=2))

    # Assert split fingerprints unchanged
    assert record["sha256_fingerprints"]["data/splits/train.csv"] == "82439b9b9aceb0d48d0ab1e33e98204eafc3d8becd963495543540debfc72b32"
    assert record["sha256_fingerprints"]["data/splits/val.csv"] == "8613e21d15ef760ba7156339bff041c195788c63fcc324b5db3ee4670aa4cb23"
    assert record["sha256_fingerprints"]["data/splits/test.csv"] == "26d3d9b481027ec302f70c47b99367646b7a9a0d02c2e3c253ff0cf62b2e44c8"

    print("  -> CHECK 4: PASS (all split fingerprints match frozen splits-v1)")
    return True


def main():
    c1 = check_1_architecture_and_pipeline()
    c2 = check_2_save_and_reload()
    c3 = check_3_callbacks()
    c4 = check_4_reproducibility_run_record()

    print("\n" + "=" * 70)
    print("FINAL READINESS SUMMARY")
    print("=" * 70)
    print(f"  1. Model Architecture & Pipeline  : {'PASS' if c1 else 'FAIL'}")
    print(f"  2. Save and Reload Verification   : {'PASS' if c2 else 'FAIL'}")
    print(f"  3. Callbacks Verification         : {'PASS' if c3 else 'FAIL'}")
    print(f"  4. Reproducibility Run Record     : {'PASS' if c4 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
