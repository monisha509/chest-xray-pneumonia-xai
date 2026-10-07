# ChestXR — Member 1 ML Handoff Specification

**Document Version:** 1.0  
**Milestone:** ML Model Selection & Evaluation Complete (`model-selected-v1`, `final-test-v1`)  
**Scope:** Dataset & Machine Learning Package (Member 1) $\rightarrow$ Application Engineering & Integration (Member 2)  
**Notice:** Educational/research prototype — not a medical diagnostic system.

---

## 1. Selected Model Specification

- **Model Identifier:** `MobileNetV2`
- **Model Checkpoint Path:** `models/mobilenetv2.keras`
- **Model Checkpoint SHA-256:** `261b40749198dff988d60c0cf545e32251edbac1cd44e7c154dd0644f180af61` *(source: `models/final_selection.json`)*
- **Selected Operating Threshold:** `0.50`
- **Input Tensor Dimensions:** `(224, 224, 3)`
- **Input Value Dynamic Range Expected by Model:** Float32 array with pixel values in `[0.0, 255.0]`
- **Architecture Summary:**
  - **Backbone:** MobileNetV2 pretrained on ImageNet (`weights='imagenet'`, `include_top=False`), frozen backbone (`trainable=False`).
  - **Integrated Model-Boundary Scaling:** Keras `Rescaling(scale=1.0/127.5, offset=-1.0, name="mobilenetv2_scaling")` directly connected to input tensor, mapping `[0.0, 255.0] -> [-1.0, 1.0]`. *(Strictly matches ImageNet MobileNetV2 expectations; avoids division by 255).*
  - **Classification Head:**
    - `GlobalAveragePooling2D(name="global_pool")`
    - `Dense(128, activation="relu", name="dense_features")`
    - `Dropout(0.5, name="dropout")`
    - `Dense(1, activation="sigmoid", name="predictions")`
  - **Target Layer for Grad-CAM:** `out_relu` (spatial feature map: $7 \times 7 \times 1280$).

---

## 2. Shared Preprocessing Pipeline

Member 2 must **not** rewrite, modify, or duplicate preprocessing in the backend. Instead, import and reuse the verified implementation directly from `ml.preprocessing`:

```python
from ml.preprocessing import preprocess_image, preprocess_image_bytes
```

### Exact Pipeline Execution Order
1. **Grayscale Conversion:** Load image via Pillow and convert to single-channel grayscale (`PIL.Image.convert("L")`) to eliminate spurious color shortcuts.
2. **Bilinear Resize:** Resize to `(224, 224)` via `PIL.Image.Resampling.BILINEAR`.
3. **Channel Expansion:** Convert to float32 2D array and stack across the channel axis to create 3 identical channels: shape `(224, 224, 3)`.
4. **Retain Raw `[0.0, 255.0]` Range:** Keep pixel values unnormalized in `[0.0, 255.0]`. Preprocessing strictly **does not divide by 255**.
5. **Model-Boundary Scaling:** Scaling to `[-1.0, 1.0]` is handled internally by the first layer of `models/mobilenetv2.keras`.

---

## 3. Prediction Semantics & Confidence Calculation

The model outputs a single scalar sigmoid score $\in [0.0, 1.0]$.

### Operating Threshold Rule
$$\text{pneumonia\_score} = \text{float}(\text{model.predict}(x)[0][0])$$

- If $\text{pneumonia\_score} \ge 0.50 \implies$ **`Model Prediction: Pneumonia`**
- If $\text{pneumonia\_score} < 0.50 \implies$ **`Model Prediction: Normal`**

### Confidence Calculation Rule
Confidence must be computed symmetrically relative to the selected class:
- For a **Pneumonia** prediction ($\text{score} \ge 0.50$):
  $$\text{confidence} = \text{pneumonia\_score}$$
- For a **Normal** prediction ($\text{score} < 0.50$):
  $$\text{confidence} = 1.0 - \text{pneumonia\_score}$$

> **Important Clinical & Engineering Constraint:**
> The `confidence` value is a model output-derived score used for displaying the selected class score. It is **NOT a calibrated medical probability, clinical confidence measure, or risk assessment**.

---

## 4. Frozen Test Set Performance

The official test evaluation was executed as a sealed, one-time gate operation recorded in `models/final_test_run_record.json`. It must **not** be rerun or used for model tuning.

- **Test Manifest:** `data/splits/test.csv` ($N = 624$ images: 234 Normal, 390 Pneumonia)
- **Frozen Operating Threshold:** `0.50`
- **Accuracy:** `0.8478` (84.78%, 529 / 624)
- **Precision:** `0.8092` (80.92%)
- **Pneumonia Recall / Sensitivity:** `0.9897` (98.97%, 386 / 390 pneumonia cases correctly identified)
- **Specificity / Normal Recall:** `0.6111` (61.11%, 143 / 234 normal cases correctly identified)
- **F1-Score:** `0.8904`
- **ROC-AUC:** `0.9658`
- **Confusion Matrix:**
  - True Negatives (TN): `143`
  - False Positives (FP): `91`
  - False Negatives (FN): `4`
  - True Positives (TP): `386`

---

## 5. Dataset Split Manifests & Provenance

All splits are frozen under tag `splits-v1` and verified against exact SHA-256 checksums:

| Manifest File | Images | Normal | Pneumonia | Frozen SHA-256 Checksum |
|---|---|---|---|---|
| `data/splits/train.csv` | 3,714 | 1,017 | 2,697 | `82439b9b9aceb0d48d0ab1e33e98204eafc3d8becd963495543540debfc72b32` |
| `data/splits/val.csv` | 938 | 237 | 701 | `8613e21d15ef760ba7156339bff041c195788c63fcc324b5db3ee4670aa4cb23` |
| `data/splits/test.csv` | 624 | 234 | 390 | `26d3d9b481027ec302f70c47b99367646b7a9a0d02c2e3c253ff0cf62b2e44c8` |
| `data/splits/excluded_images.csv` | 580 | — | — | `b5da75b7c5e8b31c3ab0e5a809b96b6a6bb2cc7a5f935470d45ff6e1e301b571` |

*Note on Identifiers:* Splitting was performed using filename-derived ID numbers/group IDs (`group_id`). These represent clustered file naming tokens and must **not** be presented or claimed as verified patient identities.

---

## 6. Available Model Files

- **`models/mobilenetv2.keras`** (Selected Model):
  - Size: 11.06 MB
  - SHA-256: `261b40749198dff988d60c0cf545e32251edbac1cd44e7c154dd0644f180af61`
  - Restored best weights from Epoch 17 (`val_loss = 0.091601`).
- **`models/baseline_cnn.keras`** (Benchmark Reference):
  - Size: 3.01 MB
  - SHA-256: `13af543c0cd8d7df0351e2f796a1860fc6276bd64b08ca7d51d364c13307d95d`
  - Restored best weights from Epoch 2 (`val_loss = 0.1704`).

---

## 7. Responsibilities & Handoff to Member 2

### Member 2 Ownership Scope
1. **Backend Service (`backend/`):** FastAPI application loading `models/mobilenetv2.keras`, exposing `/predict`, `/health`, and demo image endpoints.
2. **Explainable AI (`backend/services/xai.py`):** Grad-CAM heat-map generation and visual overlay synthesis.
3. **Frontend Application (`frontend/`):** User interface for image upload, prediction display, Grad-CAM visualization, and user controls.
4. **End-to-End Integration:** CORS configuration, API wiring, and client-server orchestration.

### Grad-CAM Implementation Guidance for Member 2
- Target layer: **`out_relu`** in `models/mobilenetv2.keras`.
- Feature map shape: $(7, 7, 1280)$.
- Generate Grad-CAM for the **Pneumonia / Class-1** model output.
- Do **not** present Grad-CAM as a medical explanation or proof of disease; it is an experimental visualization of internal model feature activations.
- Keep the implementation strictly consistent with the frozen MobileNetV2 model architecture.

---

## 8. Patient and Case Metadata Constraints

- Patient/case information is **optional application metadata only**.
- It is **not** an ML input.
- It should **not** be stored or persisted by this prototype.
- The UI must warn users not to enter real patient identifiers or sensitive clinical information.

---

## 9. Required Clinical & Interface Wording Standards

To ensure safety and prevent misinterpretation, Member 2 must adhere to the following terminology rules across all frontend and API outputs:

- **Approved Terms:**
  - `"Model Prediction: Pneumonia"`
  - `"Model Prediction: Normal"`
  - `"Confidence: XX.X%"`
  - `"Educational/research prototype — not a medical diagnostic system."`
- **Strictly Prohibited Terms:**
  - ❌ `"Diagnosis"` or `"Diagnosed"`
  - ❌ `"Patient has pneumonia"`
  - ❌ `"Pneumonia detected"`
  - ❌ `"Clinical recommendation"`

---

## 10. Reproducibility & Environment Provenance

- **Global Random Seed:** `42`
- **Python Version:** `3.13.15` (64-bit)
- **TensorFlow Version:** `2.21.0`
- **scikit-learn Version:** `1.9.1`
- **NumPy Version:** `2.5.3`
- **Pillow Version:** `12.3.0`
- **Git Checkpoints:**
  - `splits-v1`: Frozen dataset splits
  - `model-selected-v1` (`51b584b`): Frozen model selection
  - `final-test-v1` (`6a02ff5`): Sealed test run record
