"""
ChestXR — Evaluation & Model Diagnostics System (Member 1)
============================================================
Validation-only evaluation system for ChestXR binary classification:
  NORMAL = 0 (Model Prediction: Normal)
  PNEUMONIA = 1 (Model Prediction: Pneumonia)

Project Wording Standards:
  - "Model Prediction: Pneumonia" / "Model Prediction: Normal"
  - Disallowed: "diagnosis", "patient has pneumonia", "pneumonia detected"
  - Research / educational prototype only; not a clinical diagnostic system.
  - Group IDs are filename-derived group ID numbers, NOT confirmed patient identities.

Core Engineering Principles:
  1. Primary Metric Engine: scikit-learn (cross-checked against verified reference implementation).
  2. Non-interactive Plotting: matplotlib with 'Agg' backend.
  3. Validation-Only Isolation: strictly uses data/splits/val.csv.
  4. Provenance & Isolation: enforces splits-v1 hash integrity.
  5. Exact Threshold Analysis: based on actual sorted predicted probabilities.
  6. Group-ID Bootstrap CIs: 2,000 resamples stratified by class.
  7. Locked Final Test Gate: structurally locked behind a frozen selection file,
     model SHA-256 verification, single-execution enforcement, and typed confirmation.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress verbose TensorFlow C++ informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Non-interactive matplotlib backend to prevent any GUI display hangs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from PIL import Image
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sk_confusion_matrix,
    roc_curve as sk_roc_curve,
)

from ml.preprocessing import (
    PROJECT_ROOT,
    load_dataset,
    preprocess_image,
)

# ── Directories & File Paths ──────────────────────────────────────────────────
EVALUATION_DIR = PROJECT_ROOT / "models" / "evaluation"
PLOTS_DIR = EVALUATION_DIR / "plots"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_CSV_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"
TEST_CSV_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"

BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_cnn.keras"
MOBILENETV2_MODEL_PATH = PROJECT_ROOT / "models" / "mobilenetv2.keras"

BASELINE_HISTORY_PATH = PROJECT_ROOT / "models" / "baseline_training_history.json"
MOBILENETV2_HISTORY_PATH = PROJECT_ROOT / "models" / "mobilenetv2_training_history.json"

FINAL_SELECTION_PATH = PROJECT_ROOT / "models" / "final_selection.json"
FINAL_TEST_RECORD_PATH = PROJECT_ROOT / "models" / "final_test_run_record.json"

# Frozen splits-v1 SHA-256 checksums
EXPECTED_VAL_SHA256 = "8613e21d15ef760ba7156339bff041c195788c63fcc324b5db3ee4670aa4cb23"
EXPECTED_TEST_SHA256 = "26d3d9b481027ec302f70c47b99367646b7a9a0d02c2e3c253ff0cf62b2e44c8"

CONFIRMATION_STRING = "RUN FINAL TEST EVALUATION"


# ── Utility: Hashing ──────────────────────────────────────────────────────────
def compute_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_manifest_integrity(csv_path: Path, expected_hash: str, manifest_name: str) -> None:
    """Verify that a split manifest matches its frozen SHA-256 checksum."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    actual_hash = compute_sha256(csv_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Integrity check FAILED for {manifest_name} ({csv_path.name})!\n"
            f"  Expected SHA-256: {expected_hash}\n"
            f"  Actual SHA-256  : {actual_hash}\n"
            f"Evaluation halted. Split manifests must remain strictly frozen."
        )


# ── Manifest Parsing & Label Alignment ────────────────────────────────────────
def read_validation_manifest(csv_path: Path = VAL_CSV_PATH) -> List[Dict[str, Any]]:
    """
    Read validation manifest and strictly validate row-by-row structure.
    Removes redundant patient_id, keeping solely the filename-derived group_id.
    """
    verify_manifest_integrity(csv_path, EXPECTED_VAL_SHA256, "Validation Split (val.csv)")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"filepath", "filename", "group_id", "class_name", "label"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest {csv_path} missing required columns: {required_cols - set(reader.fieldnames or [])}")

        for idx, row in enumerate(reader):
            label = int(row["label"])
            cls_name = row["class_name"].strip().upper()
            if cls_name == "NORMAL" and label != 0:
                raise ValueError(f"Label alignment mismatch at row {idx}: NORMAL has label {label}")
            if cls_name == "PNEUMONIA" and label != 1:
                raise ValueError(f"Label alignment mismatch at row {idx}: PNEUMONIA has label {label}")

            # Keep group_id; omit redundant patient_id
            records.append({
                "filepath": row["filepath"],
                "filename": row["filename"],
                "group_id": row["group_id"],
                "class_name": cls_name,
                "label": label,
                "sub_type": row.get("sub_type", ""),
            })

    if not records:
        raise ValueError(f"Validation manifest {csv_path} is empty.")
    return records


# ── Metric Engines: Scikit-Learn (Primary) & Handwritten (Reference) ──────────
def compute_metrics_sklearn(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Primary metric calculation engine using scikit-learn.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float32)
    y_pred = (y_probs >= threshold).astype(np.int32)

    cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0.0))
    sens = float(recall_score(y_true, y_pred, zero_division=0.0))  # Pneumonia recall (Sensitivity)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0          # Normal recall (Specificity)
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

    try:
        auc_score = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc_score = 0.5

    fpr, tpr, roc_thresh = sk_roc_curve(y_true, y_probs)

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "pneumonia_recall_sensitivity": round(sens, 4),
        "specificity_normal_recall": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc_score, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "roc_curve": {
            "fpr": [round(float(v), 5) for v in fpr],
            "tpr": [round(float(v), 5) for v in tpr],
            "thresholds": [round(float(v), 5) for v in roc_thresh],
        },
        "support": {
            "total_samples": int(len(y_true)),
            "normal_count": int(tn + fp),
            "pneumonia_count": int(tp + fn),
        },
    }


def compute_metrics_handwritten(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Reference/test handwritten NumPy implementation for cross-checking.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float32)
    y_pred = (y_probs >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

    # Trapezoidal ROC-AUC
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)
    if pos_count == 0 or neg_count == 0:
        auc_val = 0.5
    else:
        desc_indices = np.argsort(y_probs, kind="mergesort")[::-1]
        y_true_sorted = y_true[desc_indices]
        y_probs_sorted = y_probs[desc_indices]
        distinct_indices = np.where(np.diff(y_probs_sorted))[0]
        threshold_idxs = np.r_[distinct_indices, y_true.size - 1]
        tps = np.r_[0, np.cumsum(y_true_sorted)[threshold_idxs]]
        fps = np.r_[0, 1 + threshold_idxs - tps[1:]]
        fpr = fps / fps[-1]
        tpr = tps / tps[-1]
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc_val = float(trapz_fn(tpr, fpr)) if trapz_fn else 0.5

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "pneumonia_recall_sensitivity": round(float(sens), 4),
        "specificity_normal_recall": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc_val), 4),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def run_cross_check(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Perform cross-check between handwritten NumPy and scikit-learn metrics.
    """
    sk_m = compute_metrics_sklearn(y_true, y_probs, threshold=threshold)
    hw_m = compute_metrics_handwritten(y_true, y_probs, threshold=threshold)

    checks = [
        ("accuracy", hw_m["accuracy"], sk_m["accuracy"], 1e-4),
        ("precision", hw_m["precision"], sk_m["precision"], 1e-4),
        ("pneumonia_recall_sensitivity", hw_m["pneumonia_recall_sensitivity"], sk_m["pneumonia_recall_sensitivity"], 1e-4),
        ("specificity_normal_recall", hw_m["specificity_normal_recall"], sk_m["specificity_normal_recall"], 1e-4),
        ("f1_score", hw_m["f1_score"], sk_m["f1_score"], 1e-4),
        ("roc_auc", hw_m["roc_auc"], sk_m["roc_auc"], 1e-3),
        ("true_negatives_tn", hw_m["tn"], sk_m["confusion_matrix"]["tn"], 0),
        ("false_positives_fp", hw_m["fp"], sk_m["confusion_matrix"]["fp"], 0),
        ("false_negatives_fn", hw_m["fn"], sk_m["confusion_matrix"]["fn"], 0),
        ("true_positives_tp", hw_m["tp"], sk_m["confusion_matrix"]["tp"], 0),
    ]

    results = []
    for name, hw_val, sk_val, tol in checks:
        diff = abs(hw_val - sk_val)
        passed = diff <= tol
        results.append({
            "metric": name,
            "handwritten_value": hw_val,
            "scikit_learn_value": sk_val,
            "absolute_difference": round(float(diff), 6),
            "tolerance": tol,
            "status": "PASS" if passed else "FAIL",
        })
        if not passed:
            raise AssertionError(f"Metric cross-check FAILED for {name}: HW={hw_val}, SK={sk_val}, diff={diff} > {tol}")

    return results


# ── Group-ID Bootstrap Confidence Intervals ───────────────────────────────────
def compute_group_id_bootstrap_ci(
    records: List[Dict[str, Any]],
    y_probs: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    2,000 Group-ID-level bootstrap resamples stratified by class.
    Resampling unit: filename-derived group ID (not confirmed patient identities).
    """
    rng = np.random.RandomState(seed)

    # Organize record indices by group_id and class
    normal_groups: Dict[str, List[int]] = {}
    pneumonia_groups: Dict[str, List[int]] = {}

    for idx, rec in enumerate(records):
        gid = rec["group_id"]
        if rec["label"] == 0:
            normal_groups.setdefault(gid, []).append(idx)
        else:
            pneumonia_groups.setdefault(gid, []).append(idx)

    normal_gids = list(normal_groups.keys())
    pneumonia_gids = list(pneumonia_groups.keys())

    sensitivities = []
    specificities = []
    aucs = []

    y_true_all = np.array([r["label"] for r in records], dtype=np.int32)

    for _ in range(n_bootstraps):
        sampled_norm_gids = rng.choice(normal_gids, size=len(normal_gids), replace=True)
        sampled_pneu_gids = rng.choice(pneumonia_gids, size=len(pneumonia_gids), replace=True)

        sample_indices = []
        for gid in sampled_norm_gids:
            sample_indices.extend(normal_groups[gid])
        for gid in sampled_pneu_gids:
            sample_indices.extend(pneumonia_groups[gid])

        s_true = y_true_all[sample_indices]
        s_probs = y_probs[sample_indices]
        s_pred = (s_probs >= threshold).astype(np.int32)

        tp = np.sum((s_true == 1) & (s_pred == 1))
        fn = np.sum((s_true == 1) & (s_pred == 0))
        tn = np.sum((s_true == 0) & (s_pred == 0))
        fp = np.sum((s_true == 0) & (s_pred == 1))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            auc_val = float(roc_auc_score(s_true, s_probs))
        except ValueError:
            auc_val = 0.5

        sensitivities.append(sens)
        specificities.append(spec)
        aucs.append(auc_val)

    def _ci(vals: List[float]) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(vals)), 4),
            "ci_lower_95": round(float(np.percentile(vals, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(vals, 97.5)), 4),
        }

    return {
        "n_bootstraps": n_bootstraps,
        "seed": seed,
        "method": "Class-stratified Group-ID cluster bootstrap",
        "annotation": "Bootstrap resampling unit: filename-derived group ID; IDs are not confirmed patient identities.",
        "pneumonia_recall_sensitivity": _ci(sensitivities),
        "specificity_normal_recall": _ci(specificities),
        "roc_auc": _ci(aucs),
    }


# ── Exact Threshold Analysis ──────────────────────────────────────────────────
def compute_exact_threshold_targets(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, Any]:
    """
    Derived from actual sorted predicted probabilities.
    Finds highest threshold that satisfies at least 90%, 95%, 97% Pneumonia Sensitivity.
    Rule: predicted_probability >= threshold -> Pneumonia (1), < threshold -> Normal (0).
    """
    total_pneumonia = int(np.sum(y_true == 1))  # 701
    total_normal = int(np.sum(y_true == 0))     # 237

    # Sorted candidate thresholds from unique probabilities
    unique_scores = np.sort(np.unique(y_probs))
    # Add boundary
    candidate_thresholds = np.concatenate(([0.0], unique_scores, [1.0]))

    targets = [0.90, 0.95, 0.97]
    target_results = {}

    for target in targets:
        target_name = f"target_{int(target * 100)}pct"
        best_match = None

        for t in reversed(candidate_thresholds):
            y_pred = (y_probs >= t).astype(np.int32)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if sens >= target:
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = (2.0 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
                best_match = {
                    "target_recall": target,
                    "selected_threshold": round(float(t), 4),
                    "pneumonia_recall_sensitivity": round(float(sens), 4),
                    "specificity_normal_recall": round(float(spec), 4),
                    "precision": round(float(prec), 4),
                    "f1_score": round(float(f1), 4),
                    "false_negatives": fn,
                    "false_positives": fp,
                    "true_positives": tp,
                    "true_negatives": tn,
                    "total_pneumonia": total_pneumonia,
                    "total_normal": total_normal,
                    "status": "Target reached",
                }
                break

        if best_match:
            target_results[target_name] = best_match
        else:
            target_results[target_name] = {
                "target_recall": target,
                "status": "Target cannot be reached on this validation distribution.",
            }

    return target_results


def build_threshold_table(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Standard threshold sweep table (0.05 to 0.95 in 0.05 steps + 0.50).
    """
    thresholds = sorted(list(set([round(t, 2) for t in np.linspace(0.05, 0.95, 19)] + [0.50])))
    table = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(np.int32)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2.0 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        table.append({
            "threshold": t,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "pneumonia_recall_sensitivity": round(float(sens), 4),
            "specificity_normal_recall": round(float(spec), 4),
            "f1_score": round(float(f1), 4),
            "false_negatives": fn,
            "false_positives": fp,
        })
    return table


# ── Per-Image Predictions & Error Diagnostics ─────────────────────────────────
def extract_error_diagnostics(
    records: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Extract most confident false positives and false negatives.
    Omit redundant patient_id, keeping solely group_id.
    """
    y_pred = (y_probs >= threshold).astype(np.int32)
    fp_list = []
    fn_list = []

    for rec, yt, yp, ypr in zip(records, y_true, y_pred, y_probs):
        entry = {
            "image_path": rec["filepath"],
            "filename": rec["filename"],
            "group_id": rec["group_id"],
            "true_label": int(yt),
            "class_name": rec["class_name"],
            "predicted_probability": round(float(ypr), 4),
            "model_prediction": "Model Prediction: Pneumonia" if yp == 1 else "Model Prediction: Normal",
        }
        if yt == 0 and yp == 1:
            fp_list.append(entry)
        elif yt == 1 and yp == 0:
            fn_list.append(entry)

    fp_list.sort(key=lambda x: x["predicted_probability"], reverse=True)
    fn_list.sort(key=lambda x: x["predicted_probability"])

    return {
        "analysis_notes": (
            "Model predictions represent prototype classification scores. "
            "False Positives: NORMAL cases predicted as Pneumonia. "
            "False Negatives: PNEUMONIA cases predicted as Normal. "
            "Group IDs are filename-derived group ID numbers, not confirmed patient identities."
        ),
        "total_false_positives": len(fp_list),
        "total_false_negatives": len(fn_list),
        "most_confident_false_positives": fp_list[:top_n],
        "most_confident_false_negatives": fn_list[:top_n],
    }


# ── Matplotlib Non-Interactive Plots ──────────────────────────────────────────
def plot_confusion_matrix_matplotlib(
    cm_dict: Dict[str, int],
    model_name: str,
    output_path: Path,
) -> None:
    """Render 2x2 confusion matrix using matplotlib Agg backend."""
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    total = np.sum(matrix)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.7)
    fig.colorbar(cax)

    labels = [
        [f"TN\n{tn}\n({tn/total:.1%})", f"FP\n{fp}\n({fp/total:.1%})"],
        [f"FN\n{fn}\n({fn/total:.1%})", f"TP\n{tp}\n({tp/total:.1%})"],
    ]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center", color="black", fontsize=11, weight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Model Prediction:\nNormal (0)", "Model Prediction:\nPneumonia (1)"])
    ax.set_yticklabels(["Actual: Normal (0)", "Actual: Pneumonia (1)"])
    ax.set_title(f"Validation Confusion Matrix — {model_name}\n(N={total})", pad=20, fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve_matplotlib(
    fpr: List[float],
    tpr: List[float],
    auc_score: float,
    model_name: str,
    output_path: Path,
) -> None:
    """Render ROC curve using matplotlib Agg backend."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"{model_name} (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title(f"Validation ROC Curve — {model_name}", fontsize=12, weight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Full Evaluation Execution Function ────────────────────────────────────────
def evaluate_model(
    model_path: Path,
    model_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run complete, hardened validation evaluation for a given model.
    """
    print("=" * 75)
    print(f"EVALUATING MODEL ON VALIDATION: {model_name}")
    print(f"Model Path: {model_path}")
    print("=" * 75)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Step 1: Read and verify validation manifest
    records = read_validation_manifest(VAL_CSV_PATH)
    val_sha256 = compute_sha256(VAL_CSV_PATH)
    model_sha256 = compute_sha256(model_path)
    print(f"  Validation manifest verified (N={len(records)}, SHA-256: {val_sha256[:12]}...)")
    print(f"  Model verified (SHA-256: {model_sha256[:12]}...)")

    # Step 2: Load dataset sequentially (unshuffled, unaugmented)
    val_ds = load_dataset(VAL_CSV_PATH, batch_size=32, shuffle=False, augment=False)

    # Step 3: Load model and predict
    print(f"  Loading model from {model_path.name}...")
    model = tf.keras.models.load_model(model_path)
    print("  Running inference on validation set...")
    raw_predictions = model.predict(val_ds, verbose=0)
    y_probs = raw_predictions.flatten()

    # Step 4: Strict Label Alignment Verification
    y_true = np.array([r["label"] for r in records], dtype=np.int32)
    if len(y_probs) != len(y_true):
        raise ValueError(f"Prediction count mismatch! Manifest: {len(y_true)}, Preds: {len(y_probs)}")
    if not np.all(np.isfinite(y_probs)):
        raise ValueError("Model predictions contain NaN or Inf values!")

    print(f"  Label alignment confirmed: {len(y_true)} rows matched sequentially.")

    # Step 5: Primary scikit-learn metrics computation
    metrics = compute_metrics_sklearn(y_true, y_probs, threshold=threshold)

    # Step 6: Cross-check against handwritten reference
    cross_check_results = run_cross_check(y_true, y_probs, threshold=threshold)
    print("  Scikit-learn vs handwritten metric cross-check: 10/10 checks PASSED.")

    # Step 7: Group-ID Bootstrap Confidence Intervals (2,000 resamples)
    print("  Calculating 2,000 Group-ID cluster bootstrap confidence intervals...")
    ci_dict = compute_group_id_bootstrap_ci(records, y_probs, threshold=threshold, n_bootstraps=2000, seed=42)

    # Step 8: Exact Threshold Targets & Sweep Table
    exact_targets = compute_exact_threshold_targets(y_true, y_probs)
    threshold_table = build_threshold_table(y_true, y_probs)

    prefix = model_name.lower().replace(" ", "_")
    thresh_csv = EVALUATION_DIR / f"{prefix}_threshold_table.csv"
    with open(thresh_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_table[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_table)

    # Step 9: Save Per-Image Predictions (with group_id, removing redundant patient_id)
    pred_csv = EVALUATION_DIR / f"{prefix}_validation_predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "filename",
                "group_id",
                "true_label",
                "class_name",
                "predicted_probability",
                "predicted_label",
                "threshold_used",
                "model_prediction",
                "model_name",
            ],
        )
        writer.writeheader()
        for rec, yt, yp in zip(records, y_true, y_probs):
            pred_lbl = int(yp >= threshold)
            writer.writerow({
                "image_path": rec["filepath"],
                "filename": rec["filename"],
                "group_id": rec["group_id"],
                "true_label": yt,
                "class_name": rec["class_name"],
                "predicted_probability": round(float(yp), 4),
                "predicted_label": pred_lbl,
                "threshold_used": threshold,
                "model_prediction": "Model Prediction: Pneumonia" if pred_lbl == 1 else "Model Prediction: Normal",
                "model_name": model_name,
            })

    # Step 10: Error Diagnostics
    error_summary = extract_error_diagnostics(records, y_true, y_probs, threshold=threshold)

    # Step 11: Plots via matplotlib
    cm_plot_path = PLOTS_DIR / f"{prefix}_confusion_matrix.png"
    roc_plot_path = PLOTS_DIR / f"{prefix}_roc_curve.png"
    plot_confusion_matrix_matplotlib(metrics["confusion_matrix"], model_name, cm_plot_path)
    plot_roc_curve_matplotlib(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], metrics["roc_auc"], model_name, roc_plot_path)

    # Step 12: Load Training History if available
    training_history_data = None
    if "baseline" in prefix and BASELINE_HISTORY_PATH.exists():
        with open(BASELINE_HISTORY_PATH, "r", encoding="utf-8") as f:
            training_history_data = json.load(f)
    elif "mobilenetv2" in prefix and MOBILENETV2_HISTORY_PATH.exists():
        with open(MOBILENETV2_HISTORY_PATH, "r", encoding="utf-8") as f:
            training_history_data = json.load(f)

    # Format structured training history list
    formatted_history = []
    if training_history_data:
        epochs_count = len(training_history_data.get("loss", []))
        for ep in range(epochs_count):
            formatted_history.append({
                "epoch": ep + 1,
                "loss": round(float(training_history_data["loss"][ep]), 4),
                "accuracy": round(float(training_history_data["accuracy"][ep]), 4),
                "auc": round(float(training_history_data["auc"][ep]), 4),
                "precision": round(float(training_history_data["precision"][ep]), 4),
                "recall": round(float(training_history_data["recall"][ep]), 4),
                "val_loss": round(float(training_history_data["val_loss"][ep]), 4),
                "val_accuracy": round(float(training_history_data["val_accuracy"][ep]), 4),
                "val_auc": round(float(training_history_data["val_auc"][ep]), 4),
                "val_precision": round(float(training_history_data["val_precision"][ep]), 4),
                "val_recall": round(float(training_history_data["val_recall"][ep]), 4),
            })

    # Step 13: Assemble Hard-to-Misread Structured JSON
    result_doc = {
        "schema_version": "1.0",
        "model_name": model_name,
        "data_split": "validation",
        "class_names": ["NORMAL", "PNEUMONIA"],
        "label_convention": {"NORMAL": 0, "PNEUMONIA": 1},
        "positive_class": "PNEUMONIA",
        "threshold_rule": "PNEUMONIA when predicted_probability >= threshold",
        "threshold_used": threshold,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "pneumonia_recall_sensitivity": metrics["pneumonia_recall_sensitivity"],
            "specificity_normal_recall": metrics["specificity_normal_recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "roc_curve": metrics["roc_curve"],
        "threshold_table": threshold_table,
        "exact_threshold_targets": exact_targets,
        "training_history": formatted_history,
        "error_diagnostics": error_summary,
        "bootstrap_confidence_intervals": ci_dict,
        "cross_check": cross_check_results,
        "provenance": {
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "manifest_path": str(VAL_CSV_PATH),
            "manifest_sha256": val_sha256,
            "random_seed": 42,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "scikit_learn_version": sklearn.__version__,
            "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    }

    metrics_json_path = EVALUATION_DIR / f"{prefix}_validation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2)

    print(f"  Accuracy                      : {metrics['accuracy']:.4f}")
    print(f"  Precision                     : {metrics['precision']:.4f}")
    print(f"  Sensitivity (Pneumonia Recall): {metrics['pneumonia_recall_sensitivity']:.4f}")
    print(f"  Specificity (NORMAL Recall)   : {metrics['specificity_normal_recall']:.4f}")
    print(f"  F1-Score                      : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC                       : {metrics['roc_auc']:.4f}")
    print(f"  Saved structured JSON to      : {metrics_json_path}")

    return result_doc


# ── Side-by-Side Model Comparison ─────────────────────────────────────────────
def generate_model_comparison_file() -> Optional[Path]:
    """
    Generate side-by-side comparison table for Baseline CNN and MobileNetV2.
    """
    baseline_json = EVALUATION_DIR / "baseline_cnn_validation_metrics.json"
    mbv2_json = EVALUATION_DIR / "mobilenetv2_validation_metrics.json"

    models_data = []
    if baseline_json.exists():
        with open(baseline_json, "r", encoding="utf-8") as f:
            models_data.append(json.load(f))
    if mbv2_json.exists():
        with open(mbv2_json, "r", encoding="utf-8") as f:
            models_data.append(json.load(f))

    if not models_data:
        print("No evaluation metric files found yet. Run evaluation first.")
        return None

    comparison_rows = []
    for m in models_data:
        met = m["metrics"]
        ci = m.get("bootstrap_confidence_intervals", {})
        comparison_rows.append({
            "model_name": m["model_name"],
            "threshold": m["threshold_used"],
            "accuracy": met["accuracy"],
            "precision": met["precision"],
            "pneumonia_recall_sensitivity": met["pneumonia_recall_sensitivity"],
            "sensitivity_95ci": f"[{ci.get('pneumonia_recall_sensitivity', {}).get('ci_lower_95', 'N/A')}, {ci.get('pneumonia_recall_sensitivity', {}).get('ci_upper_95', 'N/A')}]",
            "specificity_normal_recall": met["specificity_normal_recall"],
            "specificity_95ci": f"[{ci.get('specificity_normal_recall', {}).get('ci_lower_95', 'N/A')}, {ci.get('specificity_normal_recall', {}).get('ci_upper_95', 'N/A')}]",
            "f1_score": met["f1_score"],
            "roc_auc": met["roc_auc"],
            "auc_95ci": f"[{ci.get('roc_auc', {}).get('ci_lower_95', 'N/A')}, {ci.get('roc_auc', {}).get('ci_upper_95', 'N/A')}]",
            "model_sha256": m["provenance"]["model_sha256"][:12] + "...",
        })

    out_csv = EVALUATION_DIR / "model_comparison_validation.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\nSaved side-by-side validation comparison to: {out_csv}")

    # Generate combined ROC comparison plot if both models are evaluated
    if len(models_data) >= 2:
        comp_roc_path = PLOTS_DIR / "model_comparison_roc_curve.png"
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        colors = ["#2563eb", "#16a34a"]
        for idx, m in enumerate(models_data):
            fpr = m["roc_curve"]["fpr"]
            tpr = m["roc_curve"]["tpr"]
            auc_val = m["metrics"]["roc_auc"]
            ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2, label=f"{m['model_name']} (AUC = {auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
        ax.set_title("Validation ROC Comparison — Baseline CNN vs MobileNetV2", fontsize=12, weight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(comp_roc_path, dpi=150)
        plt.close(fig)
        print(f"Saved side-by-side ROC comparison plot to: {comp_roc_path}")

    return out_csv


# ── Locked Final Test-Set Evaluation Gate ─────────────────────────────────────
def run_final_test_evaluation(
    confirmation: str,
    override_repeat: bool = False,
    override_reason: str = "",
) -> Dict[str, Any]:
    """
    Locked Final Test-Set Evaluation Routine.

    Gate Conditions (All must pass):
      1. models/final_selection.json MUST exist.
      2. Threshold is read ONLY from models/final_selection.json. Never accepted from CLI.
      3. Selected model SHA-256 and validation split SHA-256 must match current files.
      4. Frozen test manifest SHA-256 must match expected splits-v1 hash.
      5. Confirmation string must match exactly: 'RUN FINAL TEST EVALUATION'.
      6. Single-Execution Enforcement: if models/final_test_run_record.json exists,
         execution is blocked unless override_repeat is explicitly True.
         Any override is permanently recorded with repeat_override_used: True and reason.
      7. Final test report contains ONLY metrics at the frozen threshold, ROC-AUC,
         and confusion matrix (NO threshold sweep on test set!).
    """
    print("=" * 75)
    print("FINAL TEST EVALUATION GATE")
    print("=" * 75)

    # Gate Check 1: Selection File
    if not FINAL_SELECTION_PATH.exists():
        raise PermissionError(
            f"ACCESS DENIED: Final selection file not found at {FINAL_SELECTION_PATH}!\n"
            f"You must first select and freeze a model and decision threshold based on validation data."
        )

    with open(FINAL_SELECTION_PATH, "r", encoding="utf-8") as f:
        selection = json.load(f)

    # Gate Check 2: Typed Confirmation
    if confirmation.strip() != CONFIRMATION_STRING:
        raise ValueError(
            f"ACCESS DENIED: Confirmation string mismatch!\n"
            f"  Expected: '{CONFIRMATION_STRING}'\n"
            f"  Received: '{confirmation}'\n"
            f"Final test evaluation aborted to protect test set integrity."
        )

    # Gate Check 3: Single-Execution Enforcement
    repeat_override_used = False
    if FINAL_TEST_RECORD_PATH.exists():
        if not override_repeat:
            raise PermissionError(
                f"ACCESS DENIED: Final test evaluation has ALREADY been executed previously!\n"
                f"A permanent record exists at: {FINAL_TEST_RECORD_PATH}\n"
                f"Repeated test evaluation violates model validation integrity.\n"
                f"To deliberately override, pass --override-repeat-test and specify --override-reason."
            )
        repeat_override_used = True
        print(f"WARNING: Repeating final test evaluation under deliberate override! Reason: {override_reason}")

    # Gate Check 4: Model Verification
    model_path = Path(selection["selected_model_file_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Selected model not found at: {model_path}")

    actual_model_hash = compute_sha256(model_path)
    if actual_model_hash != selection["selected_model_sha256"]:
        raise ValueError(
            f"Model SHA-256 mismatch!\n"
            f"  Expected in selection: {selection['selected_model_sha256']}\n"
            f"  Actual on disk       : {actual_model_hash}"
        )

    # Gate Check 5: Validation & Test Manifest Integrity
    verify_manifest_integrity(VAL_CSV_PATH, EXPECTED_VAL_SHA256, "Validation Split (val.csv)")
    verify_manifest_integrity(TEST_CSV_PATH, EXPECTED_TEST_SHA256, "Test Split (test.csv)")

    # Gate Check 6: Read Threshold STRICTLY from selection file
    frozen_threshold = float(selection["selected_threshold"])
    print(f"All final test gate criteria satisfied. Frozen threshold: {frozen_threshold}")

    # NOTE: This routine is deliberately not invoked by default.
    records = []
    with open(TEST_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append({
                "filepath": r["filepath"],
                "filename": r["filename"],
                "group_id": r["group_id"],
                "class_name": r["class_name"].strip().upper(),
                "label": int(r["label"]),
            })

    test_ds = load_dataset(TEST_CSV_PATH, batch_size=32, shuffle=False, augment=False)
    y_true = np.array([r["label"] for r in records], dtype=np.int32)

    model = tf.keras.models.load_model(model_path)
    preds = model.predict(test_ds, verbose=0).flatten()

    # Calculate metrics at frozen threshold ONLY (NO threshold sweep on test set!)
    test_metrics = compute_metrics_sklearn(y_true, preds, threshold=frozen_threshold)

    run_record = {
        "record_type": "FINAL_FROZEN_TEST_EVALUATION",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selected_model": selection["selected_model_name"],
        "model_file": str(model_path),
        "model_sha256": actual_model_hash,
        "selected_threshold": frozen_threshold,
        "test_manifest_sha256": EXPECTED_TEST_SHA256,
        "sample_count": len(y_true),
        "final_test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "pneumonia_recall_sensitivity": test_metrics["pneumonia_recall_sensitivity"],
            "specificity_normal_recall": test_metrics["specificity_normal_recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
        },
        "confusion_matrix": test_metrics["confusion_matrix"],
        "repeat_override_used": repeat_override_used,
        "repeat_override_reason": override_reason if repeat_override_used else None,
        "seed": selection.get("seed", 42),
        "tensorflow_version": tf.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    with open(FINAL_TEST_RECORD_PATH, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    print(f"Final test evaluation completed and recorded to: {FINAL_TEST_RECORD_PATH}")
    return run_record


# ── Main CLI Dispatcher ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ChestXR Evaluation & Model Diagnostics System",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "mobilenetv2", "both"],
        default="baseline",
        help="Model to evaluate on validation set (default: baseline)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification (default: 0.5)",
    )
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="Execute final test-set evaluation (requires frozen selection & confirmation)",
    )
    parser.add_argument(
        "--confirmation",
        type=str,
        default="",
        help="Required confirmation string for final test evaluation: 'RUN FINAL TEST EVALUATION'",
    )
    parser.add_argument(
        "--override-repeat-test",
        action="store_true",
        help="Allow re-running final test evaluation after record exists",
    )
    parser.add_argument(
        "--override-reason",
        type=str,
        default="",
        help="Reason for repeating final test evaluation if overriding",
    )

    args = parser.parse_args()

    # Route 1: Locked Final Test Evaluation Gate
    if args.run_final_test:
        run_final_test_evaluation(
            confirmation=args.confirmation,
            override_repeat=args.override_repeat_test,
            override_reason=args.override_reason,
        )
        return

    # Route 2: Standard Validation Evaluation
    if args.model in ("baseline", "both"):
        evaluate_model(
            model_path=BASELINE_MODEL_PATH,
            model_name="Baseline CNN",
            threshold=args.threshold,
        )

    if args.model in ("mobilenetv2", "both"):
        if MOBILENETV2_MODEL_PATH.exists():
            evaluate_model(
                model_path=MOBILENETV2_MODEL_PATH,
                model_name="MobileNetV2",
                threshold=args.threshold,
            )
        else:
            print(f"\nNotice: MobileNetV2 model not found at {MOBILENETV2_MODEL_PATH}. Skipping MobileNetV2 for now.")

    if args.model in ("baseline", "mobilenetv2", "both"):
        generate_model_comparison_file()


if __name__ == "__main__":
    main()
