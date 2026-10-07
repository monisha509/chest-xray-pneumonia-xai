# Dataset QA Report — ChestXR

**Date:** 2026-10-04  
**Analyst:** Member 1  
**Tool:** Pillow (PIL) + MD5 hashing, Python pathlib scan

---

## 1. Dataset Root

| Item | Path |
|---|---|
| Extracted archive root | `data/raw/archive/chest_xray/` |
| **Active dataset root (used for all splits)** | **`data/raw/archive/chest_xray/`** |

> The `.zip` extracted to `data/raw/archive/chest_xray/` which contains the three canonical splits (`train/`, `test/`, `val/`) plus two artefacts (see §4).

---

## 2. Split Folder Structure

Each split contains exactly two class sub-folders:

```
data/raw/archive/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

No unexpected files were found directly inside `train/`, `test/`, or `val/` (only sub-folders, no loose files at that level).

---

## 3. Image Counts per Split

| Split | NORMAL | PNEUMONIA | Split Total |
|---|---|---|---|
| **train** | 1,341 | 3,875 | **5,216** |
| **test** | 234 | 390 | **624** |
| **val** | 8 | 8 | **16** |
| **TOTAL** | **1,583** | **4,273** | **5,856** |

### Class Imbalance

| Metric | Value |
|---|---|
| Overall PNEUMONIA ratio | 73.0% |
| Overall NORMAL ratio | 27.0% |
| Train imbalance ratio (PNEUMONIA:NORMAL) | **2.89 : 1** |

**WARNING — Significant class imbalance:** PNEUMONIA images outnumber NORMAL by ~2.9× in training. This must be addressed during training (class weights or oversampling).

### Val Set Size

**WARNING — Val set is tiny:** The `val/` split contains only **16 images** (8 NORMAL, 8 PNEUMONIA). This is far too small for reliable validation. A custom validation split should be carved from `train/` during the data-splitting step.

---

## 4. Unexpected / Extra Folders

| Folder | Status | Action Required |
|---|---|---|
| `data/raw/archive/chest_xray/__MACOSX/` | macOS ZIP artefact | Ignore — do NOT delete; not part of dataset |
| `data/raw/archive/chest_xray/chest_xray/` | Nested duplicate of root | Ignore — contains identical `train/test/val` sub-tree; use only the top-level splits |
| `data/raw/archive/chest_xray/chest_xray/.DS_Store` | macOS metadata file | Ignore |

**IMPORTANT:** The **actual dataset to use** is the top-level `data/raw/archive/chest_xray/{train,test,val}/`. The nested `chest_xray/chest_xray/` folder is a ZIP extraction artefact and must be excluded from all paths in code.

---

## 5. Corrupt / Unreadable Files

| Result | Count |
|---|---|
| Files scanned | 5,856 |
| Corrupt / unreadable files | **0** |

All 5,856 images passed Pillow `verify()` + `load()` (header check + full pixel decode). No corrupt files found.

---

## 6. Image File Extensions

| Extension | Count |
|---|---|
| `.jpeg` | 5,856 |

All images are `.jpeg`. No mixed formats to handle.

---

## 7. Total Image Count

| | Count |
|---|---|
| **Total images** | **5,856** |

---

## 8. Filename Patterns & Patient-Level Grouping

### NORMAL filenames

| Pattern | Example | Patient ID field |
|---|---|---|
| `IM-XXXX-YYYY.jpeg` | `IM-0115-0001.jpeg` | `XXXX` = scan ID |
| `NORMAL2-IM-XXXX-YYYY.jpeg` | `NORMAL2-IM-1427-0001.jpeg` | `XXXX` = scan ID |
| `IM-XXXX-YYYY-ZZZZ.jpeg` (extra segment) | `IM-0429-0001-0001.jpeg` | `XXXX` = scan ID |

- **Unique NORMAL scan IDs (IM-XXXX):** 1,113
- **Unmatched NORMAL filenames:** 144 (all follow the `IM-XXXX-YYYY-ZZZZ` 4-segment variant — still contain a usable scan ID)

### PNEUMONIA filenames

| Pattern | Example | Patient ID field |
|---|---|---|
| `personNNNN_bacteria_NNN.jpeg` | `person1000_bacteria_2931.jpeg` | `NNNN` = person ID |
| `personNNNN_virus_NNN.jpeg` | `person1000_virus_1681.jpeg` | `NNNN` = person ID |

- **Unique PNEUMONIA person IDs:** 1,674
- **Sub-type extractable:** yes — `bacteria` vs `virus` is encoded in the filename

### Patient-level grouping feasibility

Patient-level grouping **is possible** for both classes using the scan/person ID embedded in filenames. This is important to prevent data leakage (same patient appearing in both train and test). The data-splitting step should respect these IDs.

---

## 9. Duplicate Image Check (MD5 Hashes)

| Metric | Value |
|---|---|
| Unique MD5 hashes | 5,824 |
| Duplicate groups | **30** |
| Files involved in duplicates | **62** |
| Extra (redundant) copies | 32 |

### Sample duplicate groups

| Hash prefix | Files |
|---|---|
| `dcc3ab0e` | `train/NORMAL/NORMAL2-IM-0587-0001-0001.jpeg`, `NORMAL2-IM-0587-0001-0002.jpeg` |
| `edbf5c28` | `train/PNEUMONIA/person1159_virus_1944.jpeg`, `_1945.jpeg` |
| `b0ef364a` | `train/PNEUMONIA/person124_virus_246.jpeg`, `_247.jpeg` |
| `de69d5b4` | `train/PNEUMONIA/person1372_bacteria_3501.jpeg`, `_3502.jpeg`, `_3503.jpeg` (triplet) |

**Pattern:** All 30 duplicate groups are within `train/` only. Consecutive-numbered scans of the same image (e.g., `_bacteria_3501`, `_bacteria_3502`).  
**Impact:** 32 redundant images in `train/PNEUMONIA` (and 1 in `train/NORMAL`).  
**Recommendation:** De-duplicate by MD5 hash before splitting (keep one copy per group). This is a data-splitting step decision.

---

## 10. Image Dimension Summary

| Dimension | Min | Max |
|---|---|---|
| Width (px) | 384 | 2,916 |
| Height (px) | 127 | 2,713 |

All images have **variable sizes** — preprocessing must resize to a fixed target (e.g., 224×224 for MobileNetV2).

### Color Mode Breakdown

| Mode | Count | Notes |
|---|---|---|
| `L` (Grayscale) | 5,573 | 95.2% — expected for X-rays |
| `RGB` | 283 | 4.8% — need to convert to grayscale or replicate to 3-channel |

**IMPORTANT:** 283 images are stored as RGB despite being X-rays. Preprocessing must normalise all images to a consistent channel format before training (either all-grayscale or all-3-channel).

---

## Summary of QA Findings

| # | Finding | Severity | Recommended Action |
|---|---|---|---|
| 1 | Dataset root confirmed | OK | Use `data/raw/archive/chest_xray/` |
| 2 | `__MACOSX/` artefact folder present | Info | Exclude from all glob patterns |
| 3 | Nested `chest_xray/chest_xray/` duplicate | Medium | Exclude; use only top-level splits |
| 4 | Class imbalance: 2.89:1 (PNEUMONIA:NORMAL) | **High** | Use class weights or oversampling in training |
| 5 | `val/` split has only 16 images | **High** | Create custom val split from `train/` |
| 6 | 0 corrupt images | OK | No action needed |
| 7 | 100% `.jpeg` extension | OK | No format handling needed |
| 8 | Patient IDs extractable from filenames | OK | Use for patient-level split to avoid leakage |
| 9 | 30 duplicate groups (62 files) in `train/` | Medium | De-duplicate by MD5 during splitting step |
| 10 | Variable image sizes (384–2916 px) | Info | Resize to 224×224 in preprocessing |
| 11 | 283 RGB images (rest are grayscale) | Medium | Normalise channel count in preprocessing |
