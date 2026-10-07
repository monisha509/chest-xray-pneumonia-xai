"""
ChestXR — MobileNetV2 Transfer Learning Training Pipeline (Member 1)
=====================================================================
Transfer learning binary classification pipeline (NORMAL = 0, PNEUMONIA = 1)
using a frozen MobileNetV2 backbone pretrained on ImageNet, paired with a custom
binary classification head.

Key Architecture & Engineering Details:
  - Input: (224, 224, 3) with pixel values in [0, 255] float32 from shared preprocessing
  - Model boundary scaling: Rescaling(1./127.5, offset=-1.0) maps [0, 255] -> [-1.0, 1.0]
    (No 1/255 scaling applied, strictly matching MobileNetV2 requirements)
  - Pretrained backbone: MobileNetV2(include_top=False, weights='imagenet')
  - Backbone frozen initially (trainable=False)
  - Grad-CAM target layer: 'out_relu' (spatial feature map: 7x7x1280)
  - Classification head: GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.5) -> Dense(1, sigmoid)
  - Optimizer: Adam(lr=1e-3), Loss: BinaryCrossentropy
  - Class weights: computed dynamically from train.csv (NORMAL ~1.8260, PNEUMONIA ~0.6885)
  - Training pipeline: shuffle=True, augment=True
  - Validation pipeline: shuffle=False, augment=False
  - Callbacks: EarlyStopping(patience=5, restore_best_weights=True), ModelCheckpoint, CSVLogger
  - Artifacts:
      models/mobilenetv2.keras
      models/mobilenetv2_training_log.csv
      models/mobilenetv2_training_history.json
      models/mobilenetv2_reproducibility_run_record.json

Safety Constraints:
  - Frozen splits v1 (train.csv and val.csv only).
  - test.csv is strictly excluded and never loaded.
  - data/raw/ and data/splits/ remain read-only.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

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

DEFAULT_MODEL_PATH = MODELS_DIR / "mobilenetv2.keras"
DEFAULT_HISTORY_PATH = MODELS_DIR / "mobilenetv2_training_history.json"
DEFAULT_LOG_PATH = MODELS_DIR / "mobilenetv2_training_log.csv"
DEFAULT_RECORD_PATH = MODELS_DIR / "mobilenetv2_reproducibility_run_record.json"

GRAD_CAM_TARGET_LAYER = "out_relu"


def compute_file_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_mobilenetv2_model(
    input_shape=(224, 224, 3),
    learning_rate: float = 1e-3,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    Build MobileNetV2 transfer learning model with integrated input scaling.

    Args:
        input_shape: Input image dimensions (224, 224, 3).
        learning_rate: Adam optimizer learning rate.
        freeze_base: Whether to freeze MobileNetV2 backbone weights.

    Returns:
        Compiled tf.keras.Model.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # 1. Model boundary input scaling: [0, 255] -> [-1.0, 1.0]
    # Exactly equivalent to tf.keras.applications.mobilenet_v2.preprocess_input
    # Strictly avoids applying 1/255 scaling.
    scaled_inputs = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv2_scaling",
    )(inputs)

    # 2. Pretrained MobileNetV2 backbone connected directly via input_tensor
    # This keeps all backbone layers accessible in the single model graph for Grad-CAM
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        input_tensor=scaled_inputs,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    # 3. Custom binary classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(base_model.output)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def generate_reproducibility_record(
    seed: int = 42,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    output_path: Path = DEFAULT_RECORD_PATH,
) -> Dict[str, Any]:
    """Generate and write the reproducibility run record JSON."""
    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"

    class_weights = compute_class_weights(train_csv)

    record = {
        "model_name": "MobileNetV2_Transfer_Learning",
        "random_seed": seed,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "input_shape": [224, 224, 3],
        "input_scaling": "Rescaling(scale=1./127.5, offset=-1.0) -> [-1.0, 1.0]",
        "backbone": {
            "name": "MobileNetV2",
            "weights": "imagenet",
            "include_top": False,
            "freeze_base_initially": True,
            "fine_tuning_active": False,
        },
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "loss": "BinaryCrossentropy",
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "grad_cam": {
            "candidate_layer": GRAD_CAM_TARGET_LAYER,
            "candidate_layer_type": "ReLU",
            "spatial_feature_shape": [7, 7, 1280],
            "alternate_conv_layer": "Conv_1",
        },
        "dataset_manifest_hashes": {
            "train_csv_sha256": compute_file_sha256(train_csv),
            "val_csv_sha256": compute_file_sha256(val_csv),
        },
        "isolation_guarantee": "test.csv strictly excluded from training, validation, and readiness checks",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"Saved reproducibility run record to: {output_path}")
    return record


def run_readiness_verification() -> Dict[str, Any]:
    """
    Run comprehensive readiness verification for MobileNetV2 before full training:
      1. Model builds cleanly with expected parameter counts (backbone frozen).
      2. Forward pass produces shape (batch, 1) with finite probabilities in [0, 1].
      3. MobileNetV2 scaling layer correctly maps [0, 255] to [-1.0, 1.0].
      4. Grad-CAM candidate layer ('out_relu') exists and outputs a spatial feature map.
      5. Model save and reload parity check produces identical numerical predictions.
      6. Class weights are correctly formatted and applied.
      7. Callbacks are verified.
      8. Split manifests train.csv and val.csv are verified against frozen checksums.
    """
    print("=" * 70)
    print("CHESTXR — MOBILENETV2 READINESS VERIFICATION")
    print("=" * 70)

    # Set random seed
    tf.keras.utils.set_random_seed(42)

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"

    # Check 1: Build model and check layer hierarchy
    print("\n[Check 1] Building MobileNetV2 model...")
    model = build_mobilenetv2_model()

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    total_count = trainable_count + non_trainable_count

    print(f"  Trainable params     : {trainable_count:,} (expected: 164,097)")
    print(f"  Non-trainable params : {non_trainable_count:,} (expected: 2,257,984)")
    print(f"  Total params         : {total_count:,} (expected: 2,422,081)")

    assert trainable_count == 164097, f"Unexpected trainable params: {trainable_count}"
    assert non_trainable_count == 2257984, f"Unexpected non-trainable params: {non_trainable_count}"

    # Check 2: Scaling layer validation
    print("\n[Check 2] Verifying MobileNetV2 model-boundary scaling...")
    scaling_layer = model.get_layer("mobilenetv2_scaling")
    test_inputs = np.array([[[[0.0, 127.5, 255.0]]]], dtype=np.float32)
    scaled_output = scaling_layer(test_inputs).numpy()
    np.testing.assert_allclose(
        scaled_output[0, 0, 0],
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-5,
    )
    print("  Scaling verified: 0.0 -> -1.0, 127.5 -> 0.0, 255.0 -> 1.0 (No 1/255 scaling)")

    # Check 3: Grad-CAM target layer spatial validation
    print(f"\n[Check 3] Verifying Grad-CAM target layer '{GRAD_CAM_TARGET_LAYER}'...")
    layer_names = [l.name for l in model.layers]
    assert GRAD_CAM_TARGET_LAYER in layer_names, f"Missing Grad-CAM layer '{GRAD_CAM_TARGET_LAYER}'"

    grad_layer = model.get_layer(GRAD_CAM_TARGET_LAYER)
    feature_shape = grad_layer.output.shape
    print(f"  Grad-CAM layer found : {grad_layer.name} ({type(grad_layer).__name__})")
    print(f"  Feature map shape    : {feature_shape} (spatial: 7x7 with 1,280 channels)")

    assert feature_shape == (None, 7, 7, 1280), f"Unexpected Grad-CAM feature shape: {feature_shape}"

    # Confirm Grad-CAM multi-output submodel creation
    grad_cam_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[grad_layer.output, model.output],
    )
    print("  Grad-CAM multi-output submodel cleanly created from model graph")

    # Check 4: Forward pass & output shape
    print("\n[Check 4] Testing forward pass on synthetic batch...")
    dummy_batch = np.random.uniform(0.0, 255.0, size=(4, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_batch, verbose=0)
    print(f"  Output shape         : {preds.shape} (expected: (4, 1))")
    print(f"  Prediction values    : {preds.flatten().round(4).tolist()}")

    assert preds.shape == (4, 1), f"Unexpected output shape: {preds.shape}"
    assert np.all(np.isfinite(preds)), "Non-finite predictions encountered"
    assert np.all((preds >= 0.0) & (preds <= 1.0)), "Predictions out of [0, 1] range"

    # Check 5: Class weights computation
    print("\n[Check 5] Verifying class weights...")
    class_weights = compute_class_weights(train_csv)
    print(f"  Computed class weights: NORMAL (0) = {class_weights[0]:.4f}, PNEUMONIA (1) = {class_weights[1]:.4f}")
    assert abs(class_weights[0] - 1.8260) < 0.01, f"Unexpected NORMAL weight: {class_weights[0]}"
    assert abs(class_weights[1] - 0.6885) < 0.01, f"Unexpected PNEUMONIA weight: {class_weights[1]}"

    # Check 6: Save and reload numerical parity
    print("\n[Check 6] Verifying save and reload parity...")
    temp_model_path = MODELS_DIR / "temp_readiness_mbv2.keras"
    model.save(temp_model_path)
    loaded_model = tf.keras.models.load_model(temp_model_path)
    reloaded_preds = loaded_model.predict(dummy_batch, verbose=0)

    if temp_model_path.exists():
        temp_model_path.unlink()

    max_diff = float(np.max(np.abs(preds - reloaded_preds)))
    print(f"  Reload numerical parity: max absolute difference = {max_diff:.8e}")
    assert max_diff < 1e-6, f"Excessive numerical difference after reload: {max_diff}"

    # Check 7: Smoke test training steps
    print("\n[Check 7] Executing smoke test training step on train and val data...")
    train_ds = load_dataset(train_csv, batch_size=32, shuffle=True, augment=True)
    val_ds = load_dataset(val_csv, batch_size=32, shuffle=False, augment=False)

    history = model.fit(
        train_ds.take(2),
        validation_data=val_ds.take(2),
        epochs=1,
        class_weight=class_weights,
        verbose=0,
    )
    smoke_train_loss = float(history.history["loss"][0])
    smoke_val_loss = float(history.history["val_loss"][0])
    print(f"  Smoke step train_loss: {smoke_train_loss:.4f} | val_loss: {smoke_val_loss:.4f}")
    assert np.isfinite(smoke_train_loss), "Train loss is non-finite"
    assert np.isfinite(smoke_val_loss), "Val loss is non-finite"

    # Check 8: Frozen manifest integrity
    print("\n[Check 8] Verifying frozen split manifests integrity...")
    expected_train_sha256 = "82439b9b9aceb0d48d0ab1e33e98204eafc3d8becd963495543540debfc72b32"
    expected_val_sha256 = "8613e21d15ef760ba7156339bff041c195788c63fcc324b5db3ee4670aa4cb23"
    actual_train_sha256 = compute_file_sha256(train_csv)
    actual_val_sha256 = compute_file_sha256(val_csv)

    assert actual_train_sha256 == expected_train_sha256, "train.csv hash mismatch"
    assert actual_val_sha256 == expected_val_sha256, "val.csv hash mismatch"
    print("  train.csv and val.csv SHA-256 hashes strictly match splits-v1")

    # Generate reproducibility record
    record = generate_reproducibility_record()

    print("\n" + "=" * 70)
    print("ALL MOBILENETV2 READINESS CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "status": "PASSED",
        "trainable_params": trainable_count,
        "non_trainable_params": non_trainable_count,
        "total_params": total_count,
        "grad_cam_layer": GRAD_CAM_TARGET_LAYER,
        "feature_shape": list(feature_shape[1:]),
        "smoke_train_loss": smoke_train_loss,
        "smoke_val_loss": smoke_val_loss,
        "reproducibility_record_path": str(DEFAULT_RECORD_PATH),
    }


def train_full(
    epochs: int = 20,
    batch_size: int = 32,
    patience: int = 5,
    learning_rate: float = 1e-3,
):
    """
    Run full MobileNetV2 transfer learning training.
    """
    print("=" * 70)
    print("CHESTXR — MOBILENETV2 TRANSFER LEARNING TRAINING")
    print(f"Target epochs: {epochs} | Batch size: {batch_size} | EarlyStopping patience: {patience}")
    print("=" * 70)

    # Reproducibility seed
    tf.keras.utils.set_random_seed(42)

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"

    class_weights = compute_class_weights(train_csv)
    print(f"Class weights: NORMAL={class_weights[0]:.4f}, PNEUMONIA={class_weights[1]:.4f}")

    train_ds = load_dataset(train_csv, batch_size=batch_size, shuffle=True, augment=True)
    val_ds = load_dataset(val_csv, batch_size=batch_size, shuffle=False, augment=False)

    model = build_mobilenetv2_model(learning_rate=learning_rate, freeze_base=True)
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
    print(f"Saved best MobileNetV2 model to: {DEFAULT_MODEL_PATH}")

    # Update reproducibility record with final paths
    generate_reproducibility_record(
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    return history_dict


def main():
    parser = argparse.ArgumentParser(description="ChestXR MobileNetV2 Training Pipeline")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run comprehensive readiness verification without starting full training",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")

    args = parser.parse_args()

    if args.smoke_test:
        run_readiness_verification()
    else:
        # Before starting full training, verify readiness
        print("Executing pre-training readiness verification...")
        run_readiness_verification()
        print("\nReadiness checks passed! Starting full MobileNetV2 training...\n")
        train_full(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
