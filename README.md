# ChestXR — Chest X-Ray Pneumonia Detection Using CNN and Explainable AI

A 4-day college project that trains a Convolutional Neural Network (CNN) to classify
chest X-ray images as **Normal** or **Pneumonia**, and uses **Explainable AI (XAI)**
techniques (Grad-CAM) to highlight the image regions that influenced each prediction.
The solution is served through a **FastAPI** backend and a lightweight **frontend**
for real-time inference and visual explanation.

## Dataset Split Version — splits-v1

The dataset split is frozen and verified with zero ID-group overlap and zero cross-split MD5 duplicates.

| File | Rows | SHA-256 Checksum |
|---|---|---|
| `data/splits/train.csv` | 3 714 | `82439b9b9aceb0d48d0ab1e33e98204eafc3d8becd963495543540debfc72b32` |
| `data/splits/val.csv` | 938 | `8613e21d15ef760ba7156339bff041c195788c63fcc324b5db3ee4670aa4cb23` |
| `data/splits/test.csv` | 624 | `26d3d9b481027ec302f70c47b99367646b7a9a0d02c2e3c253ff0cf62b2e44c8` |
| `data/splits/excluded_images.csv` | 580 | `b5da75b7c5e8b31c3ab0e5a809b96b6a6bb2cc7a5f935470d45ff6e1e301b571` |

## Application Architecture

- **Machine Learning Core (Member 1 — Frozen):** Pretrained MobileNetV2 with custom binary head (`models/mobilenetv2.keras`), operating at fixed threshold `0.50`. Preprocessing pipeline strictly reused from `ml/preprocessing.py`.
- **Backend Service (Member 2):** FastAPI application (`backend/main.py`) serving inference and Grad-CAM explainability via `out_relu` activation layer.
- **Frontend Client (Member 2):** React + Vite application (`frontend/`) supporting six-stage analysis workflow: Login -> Dashboard -> Optional Case Info -> Upload -> Processing -> Results.

## API Documentation

### 1. Health & Readiness Check
- **Endpoint:** `GET /health`
- **Response:**
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "model_name": "MobileNetV2",
    "target_layer": "out_relu",
    "threshold": 0.50,
    "version": "1.0.0"
  }
  ```

### 2. Prediction & Explainability
- **Endpoint:** `POST /predict`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: Image file (`.jpg`, `.jpeg`, `.png`, max 15MB)
  - `metadata`: Optional JSON string containing `{ case_id, age, sex, examination_date, notes }`
- **Response:**
  ```json
  {
    "prediction": "Normal" | "Pneumonia",
    "raw_score": 0.2661,
    "model_score": 0.7339,
    "model_score_percentage": 73.4,
    "threshold": 0.50,
    "model_name": "MobileNetV2",
    "timestamp": "2026-10-09T17:32:00Z",
    "original_image": "data:image/png;base64,...",
    "gradcam_image": "data:image/png;base64,...",
    "heatmap_dimensions": "7x7",
    "disclaimer": "Educational / Research Use Only...",
    "case_metadata": { ... }
  }
  ```

## Running the Application

### 1. Start the FastAPI Backend
From repository root:
```bash
py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API runs on `http://localhost:8000` (docs available at `http://localhost:8000/docs`).

### 2. Start the React Frontend
In a separate terminal:
```bash
cd frontend
npm run dev
```
Open browser to `http://localhost:5173`.

### 3. Automated Backend Tests
Run the pytest suite:
```bash
py -m pytest backend/tests/test_backend.py -v
```

### 4. Frontend Build & Lint
```bash
cd frontend
npm run lint
npm run build
```

