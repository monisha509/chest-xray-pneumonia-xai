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

