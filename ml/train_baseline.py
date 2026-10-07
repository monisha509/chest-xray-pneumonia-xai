"""
ChestXR — Baseline CNN Training Script (Member 1)
=================================================
Simple Convolutional Neural Network baseline for chest X-ray binary classification
(NORMAL = 0, PNEUMONIA = 1).

Key Architecture Details:
  - Input: (224, 224, 3)
  - Rescaling(1./255) as first layer inside model graph
  - 4 convolutional blocks
  - Final conv activation layer explicitly named 'last_conv_relu' for Grad-CAM
  - GlobalAveragePooling2D + Dense(128) + Dropout(0.5)
  - Final Dense(1, activation='sigmoid') for binary prediction
  - Optimizer: Adam
  - Loss: BinaryCrossentropy
  - Class weights computed from train.csv (NORMAL ~1.83, PNEUMONIA ~0.69)
  - EarlyStopping + ModelCheckpoint

No test.csv evaluation is performed here.
Dataset splits remain frozen as splits-v1.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Suppress verbose TensorFlow C++ informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from ml.preprocessing import (
    PROJECT_ROOT,
    compute_class_weights,
    load_dataset,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = MODELS_DIR / "baseline_cnn.keras"
DEFAULT_HISTORY_PATH = MODELS_DIR / "baseline_training_history.json"
DEFAULT_LOG_PATH = MODELS_DIR / "baseline_training_log.csv"


def build_baseline_model(input_shape=(224, 224, 3)) -> tf.keras.Model:
    """
    Build a simple baseline CNN with Grad-CAM compatible naming.

    The final convolution layer before pooling is named 'last_conv_relu'
    so that Explainable AI (Grad-CAM) can hook into feature maps cleanly.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Model-level rescaling: [0, 255] -> [0.0, 1.0]
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4 — Grad-CAM target layer
    x = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", activation="relu", name="last_conv_relu"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)

    # Single sigmoid unit: probability of PNEUMONIA (Class 1)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def run_smoke_test():
    """
    Run a short smoke test to verify model build, forward pass, finite loss,
    label correctness, class weights integration, and validation execution.
    """
    print("=" * 70)
    print("CHESTXR BASELINE CNN — SMOKE TEST")
    print("=" * 70)

    # 1. Build model
    print("\n[Smoke Test Step 1] Building baseline CNN model...")
    model = build_baseline_model()
    model.summary()

    # Verify key architectural requirements
    assert "last_conv_relu" in [layer.name for layer in model.layers], "Missing 'last_conv_relu' layer"
    assert model.input_shape == (None, 224, 224, 3), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, 1), f"Unexpected output shape: {model.output_shape}"
    print("  -> Architectural assertions PASSED.")

    # 2. Compute class weights
    print("\n[Smoke Test Step 2] Loading class weights from train.csv...")
    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"
    class_weights = compute_class_weights(train_csv)
    print(f"  Class weights: NORMAL (0) = {class_weights[0]:.4f}, PNEUMONIA (1) = {class_weights[1]:.4f}")

    # 3. Load datasets
    print("\n[Smoke Test Step 3] Initializing training and validation datasets...")
    train_ds = load_dataset(train_csv, batch_size=32, shuffle=True, augment=True)
    val_ds = load_dataset(val_csv, batch_size=32, shuffle=False, augment=False)

    # 4. Inspect single batch & forward pass
    print("\n[Smoke Test Step 4] Testing forward pass on one training batch...")
    for batch_x, batch_y in train_ds.take(1):
        x_sample = batch_x
        y_sample = batch_y
        break

    print(f"  Batch X shape : {x_sample.shape}")
    print(f"  Batch Y shape : {y_sample.shape}")
    print(f"  Batch Y unique labels : {np.unique(y_sample.numpy())}")
    assert set(np.unique(y_sample.numpy())).issubset({0.0, 1.0}), "Labels must be strictly 0 or 1"

    preds = model(x_sample, training=False).numpy()
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Predictions min: {preds.min():.4f}, max: {preds.max():.4f}")
    assert not np.any(np.isnan(preds)), "Predictions contain NaN values"
    assert np.all(np.isfinite(preds)), "Predictions contain Inf values"
    assert 0.0 <= preds.min() and preds.max() <= 1.0, "Predictions out of [0, 1] range"
    print("  -> Forward pass PASSED with finite values.")

    # 5. Short fit execution (2 training steps, 2 validation steps, 1 epoch)
    print("\n[Smoke Test Step 5] Executing short smoke-test fit (2 train steps, 2 val steps)...")
    smoke_model_path = MODELS_DIR / "baseline_smoke_test.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(smoke_model_path), monitor="val_loss", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        train_ds.take(2),
        validation_data=val_ds.take(2),
        epochs=1,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    train_loss = history.history["loss"][0]
    val_loss = history.history["val_loss"][0]
    train_acc = history.history["accuracy"][0]
    val_acc = history.history["val_accuracy"][0]

    print(f"\n  Smoke Test Training Loss   : {train_loss:.4f}")
    print(f"  Smoke Test Validation Loss : {val_loss:.4f}")
    print(f"  Smoke Test Training Acc    : {train_acc:.4f}")
    print(f"  Smoke Test Validation Acc  : {val_acc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert np.isfinite(train_loss), "Training loss is not finite"
    assert np.isfinite(val_loss), "Validation loss is not finite"

    # Clean up temporary smoke test checkpoint
    if smoke_model_path.exists():
        smoke_model_path.unlink()

    print("\n" + "=" * 70)
    print("BASELINE CNN SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "status": "PASSED",
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "class_weights": class_weights,
    }


def train_full(epochs=20, batch_size=32, patience=5):
    """
    Run full baseline training. To be executed only after smoke test confirmation.
    """
    print("=" * 70)
    print("CHESTXR — FULL BASELINE CNN TRAINING")
    print(f"Target epochs: {epochs} | Batch size: {batch_size} | EarlyStopping patience: {patience}")
    print("=" * 70)

    # Set random seed = 42 for reproducibility across Python, NumPy, and TensorFlow
    tf.keras.utils.set_random_seed(42)

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"

    class_weights = compute_class_weights(train_csv)
    print(f"Class weights: NORMAL={class_weights[0]:.4f}, PNEUMONIA={class_weights[1]:.4f}")

    train_ds = load_dataset(train_csv, batch_size=batch_size, shuffle=True, augment=True)
    val_ds = load_dataset(val_csv, batch_size=batch_size, shuffle=False, augment=False)

    model = build_baseline_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(DEFAULT_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(DEFAULT_LOG_PATH),
            separator=",",
            append=False,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(DEFAULT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"\nSaved training history to: {DEFAULT_HISTORY_PATH}")
    print(f"Saved best baseline model to: {DEFAULT_MODEL_PATH}")

    return history_dict


def main():
    parser = argparse.ArgumentParser(description="ChestXR Baseline CNN Training Script")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run short smoke test (2 train steps, 2 val steps) to verify pipeline without full training",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    else:
        train_full(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)


if __name__ == "__main__":
    main()
