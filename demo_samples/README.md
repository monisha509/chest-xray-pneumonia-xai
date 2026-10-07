# ChestXR — Demo Samples Specification

This directory contains 5 demonstration chest X-ray images curated exclusively from the frozen **validation split** (`data/splits/val.csv`) for Member 2 to test application features (image upload, preprocessing, model inference, and Grad-CAM visualization).

> **Important Notices:**
> - **Educational/Research Demonstration Only:** These images are provided solely as test samples for interface and pipeline demonstration; they are not medical cases, clinical records, or diagnostic benchmarks.
> - **Zero Test Set Contamination:** **No images from `data/splits/test.csv` are included.** All samples are derived strictly from `data/splits/val.csv`.
> - **No Personal Information:** Filenames and records are anonymized file tokens; no patient names, identifiers, or personal information are included or referenced.

---

## Sample Inventory & Provenance

| Filename | Source Validation Path | Actual Label | MobileNetV2 Output | Model Prediction | Purpose |
|---|---|---|---|---|---|
| `normal_01.jpg` | `data/raw/archive/chest_xray/train/NORMAL/IM-0125-0001.jpeg` | NORMAL (0) | `0.2661` | `Model Prediction: Normal` | Representative Normal sample |
| `normal_02.jpg` | `data/raw/archive/chest_xray/train/NORMAL/IM-0133-0001.jpeg` | NORMAL (0) | `0.2607` | `Model Prediction: Normal` | Representative Normal sample |
| `pneumonia_01.jpg` | `data/raw/archive/chest_xray/train/PNEUMONIA/person1002_bacteria_2933.jpeg` | PNEUMONIA (1) | `0.9809` | `Model Prediction: Pneumonia` | Representative Pneumonia sample |
| `pneumonia_02.jpg` | `data/raw/archive/chest_xray/train/PNEUMONIA/person1005_bacteria_2936.jpeg` | PNEUMONIA (1) | `1.0000` | `Model Prediction: Pneumonia` | Representative Pneumonia sample |
| `validation_failure_01.jpg` | `data/raw/archive/chest_xray/train/NORMAL/IM-0517-0001.jpeg` | NORMAL (0) | `0.5532` | `Model Prediction: Pneumonia` | Intentional failure case (False Positive) |

---

## Detailed Notes on `validation_failure_01.jpg`

- **Actual Label:** Normal (`label = 0`, `class_name = NORMAL`)
- **MobileNetV2 Sigmoid Output:** `0.5532`
- **Frozen Operating Threshold:** `0.50`
- **Model Classification:** Pneumonia (`predicted_label = 1`)
- **Failure Classification:** False Positive
- **Rationale for Inclusion:** This sample is deliberately provided so Member 2 can verify how the frontend and Grad-CAM behave when presenting a borderline classification near the decision boundary, ensuring the interface handles misclassifications responsibly without claiming definitive diagnosis.
