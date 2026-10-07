# ChestXR — Dataset QA Report

**Date:** 2026-10-04  
**Member:** 1 (Dataset QA)  
**Scan tool:** Python · Pillow · MD5 (hashlib)  
**Dataset root scanned:** `data/raw/archive/chest_xray/`  
**Folders excluded from scan:** `__MACOSX/`, `chest_xray/` (nested ZIP artefact)

---

## 1. Dataset Root

| Item | Value |
|---|---|
| Confirmed root | `data/raw/archive/chest_xray/` |
| Root exists | Yes |
| Top-level entries used | `train/`, `val/`, `test/` |
| Top-level entries skipped | `__MACOSX/` (macOS artefact), `chest_xray/` (nested duplicate) |

```
data/raw/archive/chest_xray/
├── train/
│   ├── NORMAL/        (1 341 images)
│   └── PNEUMONIA/     (3 875 images)
├── val/
│   ├── NORMAL/        (8 images)
│   └── PNEUMONIA/     (8 images)
└── test/
    ├── NORMAL/        (234 images)
    └── PNEUMONIA/     (390 images)
```

---

## 2. Image Counts per Split

| Split | NORMAL | PNEUMONIA | Split Total |
|---|---|---|---|
| **train** | 1 341 | 3 875 | **5 216** |
| **val** | 8 | 8 | **16** |
| **test** | 234 | 390 | **624** |
| **GRAND TOTAL** | **1 583** | **4 273** | **5 856** |

**Sanity check vs. expected:** counts match exactly.

### Class imbalance

| Metric | Value |
|---|---|
| PNEUMONIA share (overall) | 73.0 % |
| NORMAL share (overall) | 27.0 % |
| Train imbalance ratio (PNEUMONIA : NORMAL) | **2.89 : 1** |

> **Action required (training step):** Apply class weights or oversampling to compensate.

### Val set size

> **Action required (splitting step):** The provided `val/` split contains only 16 images — far too small for reliable validation monitoring. A proper validation set must be carved out of `train/` during data splitting.

---

## 3. Total Image Count

**5 856 images** (confirmed across all three splits).

---

## 4. File Extensions Found

| Extension | Count |
|---|---|
| `.jpeg` | 5 856 |

All images use a single, consistent extension. No format normalisation needed.

---

## 5. Corrupt / Unreadable Images

| Check | Result |
|---|---|
| Files scanned | 5 856 |
| Pillow `verify()` failures | **0** |
| Pillow `load()` (full decode) failures | **0** |
| Corrupt / unreadable | **None** |

All 5 856 images are intact and fully decodable.

---

## 6. Image Dimensions & Class-Wise Properties

### Overall Dimension Range

| Axis | Minimum | Maximum |
|---|---|---|
| Width (px) | 384 | 2 916 |
| Height (px) | 127 | 2 713 |

All images have **variable sizes**. Preprocessing must resize to a fixed target (224 × 224 for MobileNetV2).

### Class-Wise Image Statistics

Analyzing NORMAL and PNEUMONIA images separately reveals significant structural and channel differences:

| Metric | NORMAL (1 583 images) | PNEUMONIA (4 273 images) |
|---|---|---|
| **Color Modes** | 1 583 `L` (100.0 %) · 0 `RGB` (0.0 %) | 3 990 `L` (93.4 %) · 283 `RGB` (6.6 %) |
| **Width (min / max)** | 912 px / 2 916 px | 384 px / 2 772 px |
| **Width (mean / median)** | 1 686 px / 1 654 px | 1 195 px / 1 160 px |
| **Height (min / max)** | 496 px / 2 713 px | 127 px / 2 304 px |
| **Height (mean / median)** | 1 379 px / 1 323 px | 820 px / 776 px |
| **Aspect Ratio W/H (min / max)** | 0.877 / 2.107 | 0.835 / 3.379 |
| **Aspect Ratio W/H (mean / median)** | 1.249 / 1.235 | 1.515 / 1.490 |

#### RGB Mode Concentration Analysis
- **Finding:** All 283 RGB images in the entire dataset belong strictly to the **PNEUMONIA** class (all located within `train/PNEUMONIA`). There are **zero** RGB images in `NORMAL`.
- **Implication / Shortcut Risk:** If color channels are fed naively into a neural network, a 3-channel filter could trivially classify any 3-channel image as PNEUMONIA without learning radiographic features.
- **Mitigation (Preprocessing):** Every image must be standardized to a uniform channel representation (either converted to single-channel grayscale or uniformly broadcasted across 3 channels) during preprocessing.

#### Common Dimensions per Class

- **NORMAL (top occurrences):**
  - Extremely dispersed resolutions: no single dimension occurs more than twice (e.g., 1472×1171, 1480×1228, 1364×1148, 1616×1318, 1284×1013 each appear exactly 2 times).
- **PNEUMONIA (top occurrences):**
  - 1 072 × 648 (7 images)
  - 1 008 × 704 (7 images)
  - 1 216 × 872 (6 images)
  - 1 080 × 728 (6 images)
  - 992 × 592 (6 images)

### Unusually Small Images (< 100 px in any dimension)

**None found.** Smallest image is 384 × 127 px (`train/PNEUMONIA/person407_virus_811.jpeg`).

### Extreme Aspect Ratios

Images vary from narrow portrait ratios (~0.84) to wide panoramic strips (~3.38). All raw images are preserved as-is without modification.

#### Top 10 Widest Aspect Ratios (Most Landscape)

| Split | Class | Filename | Width | Height | Aspect Ratio (W/H) |
|---|---|---|---|---|---|
| train | PNEUMONIA | `person325_virus_661.jpeg` | 446 | 132 | **3.379** |
| train | PNEUMONIA | `person1684_bacteria_4462.jpeg` | 494 | 151 | **3.272** |
| train | PNEUMONIA | `person325_virus_659.jpeg` | 445 | 140 | **3.179** |
| train | PNEUMONIA | `person1732_bacteria_4564.jpeg` | 492 | 156 | **3.154** |
| train | PNEUMONIA | `person1719_bacteria_4541.jpeg` | 557 | 177 | **3.147** |
| train | PNEUMONIA | `person1689_bacteria_4474.jpeg` | 437 | 144 | **3.035** |
| train | PNEUMONIA | `person407_virus_811.jpeg` | 384 | 127 | **3.024** |
| train | PNEUMONIA | `person1669_bacteria_4422.jpeg` | 482 | 163 | **2.957** |
| train | PNEUMONIA | `person354_bacteria_1632.jpeg` | 460 | 157 | **2.930** |
| train | PNEUMONIA | `person1716_bacteria_4533.jpeg` | 564 | 194 | **2.907** |

#### Top 10 Tallest Aspect Ratios (Most Portrait)

| Split | Class | Filename | Width | Height | Aspect Ratio (W/H) |
|---|---|---|---|---|---|
| train | PNEUMONIA | `person1147_virus_1917.jpeg` | 1 624 | 1 944 | **0.835** |
| train | NORMAL | `NORMAL2-IM-0818-0001-0002.jpeg` | 1 679 | 1 914 | **0.877** |
| train | NORMAL | `NORMAL2-IM-1038-0001.jpeg` | 2 309 | 2 601 | **0.888** |
| train | NORMAL | `NORMAL2-IM-0994-0001.jpeg` | 2 373 | 2 663 | **0.891** |
| train | NORMAL | `NORMAL2-IM-1341-0001.jpeg` | 2 134 | 2 376 | **0.898** |
| train | NORMAL | `NORMAL2-IM-1160-0001.jpeg` | 2 091 | 2 326 | **0.899** |
| train | NORMAL | `NORMAL2-IM-0995-0001-0002.jpeg` | 1 692 | 1 878 | **0.901** |
| train | NORMAL | `NORMAL2-IM-0567-0001.jpeg` | 1 962 | 2 169 | **0.905** |
| train | NORMAL | `NORMAL2-IM-1285-0001.jpeg` | 1 962 | 2 169 | **0.905** |
| train | NORMAL | `NORMAL2-IM-1024-0001.jpeg` | 1 974 | 2 175 | **0.908** |

---

## 7. Patient ID Extraction & Full Validation

### ID Extraction Rules

- **PNEUMONIA:** `personNNNN_(bacteria|virus)_*.jpeg` → patient ID = `NNNN` (e.g., `person123_bacteria_456.jpeg` → ID `123`).
  - Regex: `^person(\d+)_(bacteria|virus)_` (handles trailing suffixes such as `person447_virus_921_1.jpeg` → ID `447`).
- **NORMAL:** `(NORMAL2?-)?IM-(\d+)-*.jpeg` → ID based on the IM number (e.g., `IM-0115-0001.jpeg` → `0115`, `NORMAL2-IM-0095-0001.jpeg` → `0095`).
  - Regex: `^(?:NORMAL2?-)?IM-(\d+)`

> [!IMPORTANT]
> **Clinical Identity Disclaimer:**  
> For PNEUMONIA, `personXXXX` represents a verified patient identifier assigned in the Guangzhou Women and Children’s Medical Center dataset.  
> For NORMAL, filename IDs extracted from `IM-XXXX` or `NORMAL2-IM-XXXX` are an **operational proxy / assumption** based on image accession prefixes, **not a clinically confirmed patient identifier**.

### Validation Results (All 5 856 Images)

| Check | Result | Status |
|---|---|---|
| Total files checked | **5 856** | Expected |
| Successfully assigned IDs | **5 856** | 100.0 % coverage |
| Missing / unmatched IDs | **0** | Perfect match |

Non-standard filenames like `person447_virus_921_1.jpeg` are correctly parsed and assigned to patient ID `447`.

### Unique Patient IDs by Split

| Split | Class | Total Files | Unique IDs |
|---|---|---|---|
| **train** | NORMAL | 1 341 | 1 000 |
| **train** | PNEUMONIA | 3 875 | 1 635 |
| **val** | NORMAL | 8 | 8 |
| **val** | PNEUMONIA | 8 | 7 |
| **test** | NORMAL | 234 | 202 |
| **test** | PNEUMONIA | 390 | 202 |

---

## 8. Patient Overlap Across Splits

To evaluate data leakage in the existing directory structure, patient IDs were intersected across all three pairs of splits:

| Split Pair | NORMAL Overlapping IDs | PNEUMONIA Overlapping IDs | Total Overlapping IDs |
|---|---|---|---|
| **train vs val** | **0** | **0** | **0** |
| **train vs test** | **94** | **170** | **264** |
| **val vs test** | **0** | **0** | **0** |

### Detailed Findings

1. **train vs val:** **0 overlapping IDs**. The existing validation set does not share any patients with the train set (though it contains only 16 images in total).
2. **val vs test:** **0 overlapping IDs**. Clean separation.
3. **train vs test:** **264 overlapping patient IDs!**
   - **NORMAL (94 overlapping IDs):** Out of 202 unique NORMAL IDs in `test/`, **94 (46.5 %)** are also present in `train/`.  
     *Overlapping IDs:* `0117`, `0129`, `0131`, `0135`, `0141`, `0145`, `0195`, `0199`, `0201`, `0206`, `0207`, `0210`, `0213`, `0217`, `0219`, `0221`, `0222`, `0229`, `0237`, `0238`, `0241`, `0249`, `0251`, `0256`, `0272`, `0273`, `0274`, `0275`, `0276`, `0277`, `0278`, `0279`, `0280`, `0282`, `0283`, `0285`, `0286`, `0288`, `0289`, `0290`, `0292`, `0294`, `0297`, `0300`, `0301`, `0302`, `0303`, `0304`, `0305`, `0307`, and 44 more.
   - **PNEUMONIA (170 overlapping IDs):** Out of 202 unique PNEUMONIA IDs in `test/`, **170 (84.2 %)** are also present in `train/`.  
     *Overlapping IDs:* `1`, `3`, `8`, `10`, `11`, `14`, `15`, `16`, `17`, `18`, `19`, `20`, `21`, `22`, `23`, `24`, `25`, `26`, `28`, `29`, `30`, `31`, `32`, `33`, `34`, `35`, `36`, `37`, `38`, `39`, `40`, `41`, `43`, `44`, `45`, `46`, `47`, `48`, `49`, `50`, `51`, `52`, `53`, `54`, `55`, `56`, `57`, `59`, `60`, `61`, and 120 more.

> [!WARNING]
> **Severe Kaggle Benchmark Data Leakage:**  
> The standard Kaggle split has massive patient-level contamination between `train` and `test`. Over 84% of pneumonia test patients appear in the training data. This explains why naive models achieve inflated test scores (>95%) while failing to generalize clinically.

---

## 9. Exact Duplicate Images (MD5) & Handling Policy

| Metric | Value |
|---|---|
| Total images scanned | 5 856 |
| Unique MD5 hashes | 5 824 |
| Duplicate groups | **30** |
| Files involved in duplicates | **62** |
| Redundant (extra) copies | **32** |
| Same-split duplicates | **30 groups** (24 in train, 6 in test) |
| Cross-split duplicates | **None** |

### Verified Duplicate Groups (Only confirmed MD5 matches reported)

| MD5 (first 8) | Verified Files |
|---|---|
| `dcc3ab0e` | `train/NORMAL/NORMAL2-IM-0587-0001-0001.jpeg`, `NORMAL2-IM-0587-0001-0002.jpeg` |
| `edbf5c28` | `train/PNEUMONIA/person1159_virus_1944.jpeg`, `_1945.jpeg` |
| `b0ef364a` | `train/PNEUMONIA/person124_virus_246.jpeg`, `_247.jpeg` |
| `3eaf3d1e` | `train/PNEUMONIA/person1261_virus_2147.jpeg`, `_2148.jpeg` |
| `ee1a4aa1` | `train/PNEUMONIA/person1312_bacteria_3317.jpeg`, `_3318.jpeg` |
| `b7d8223d` | `train/PNEUMONIA/person1343_bacteria_3413.jpeg`, `_3414.jpeg` |
| `077fce04` | `train/PNEUMONIA/person1349_bacteria_3436.jpeg`, `_3437.jpeg` |
| `de69d5b4` | `train/PNEUMONIA/person1372_bacteria_3501.jpeg`, `_3502.jpeg`, `_3503.jpeg` |
| `4a8541c3` | `train/PNEUMONIA/person1430_bacteria_3695.jpeg`, `_3696.jpeg` |
| `ec767162` | `train/PNEUMONIA/person1481_bacteria_3862.jpeg`, `_3863.jpeg` |
| `b6e7fa96` | `train/PNEUMONIA/person1481_bacteria_3864.jpeg`, `_3865.jpeg` |
| `23cd1cda` | `train/PNEUMONIA/person1481_bacteria_3866.jpeg`, `_3867.jpeg` |
| `684d201b` | `train/PNEUMONIA/person1496_bacteria_3909.jpeg`, `_3910.jpeg` |
| `061c1fac` | `train/PNEUMONIA/person1619_bacteria_4267.jpeg`, `_4268.jpeg` |
| `78ff1c62` | `train/PNEUMONIA/person1628_bacteria_4297.jpeg`, `_4298.jpeg` |
| `56dea5c1` | `train/PNEUMONIA/person162_virus_321.jpeg`, `_322.jpeg` |
| `99e09ba8` | `train/PNEUMONIA/person258_bacteria_1207.jpeg`, `_1208.jpeg`, `_1209.jpeg` |
| `9cadd725` | `train/PNEUMONIA/person264_bacteria_1230.jpeg`, `_1231.jpeg` |
| `d9a3798f` | `train/PNEUMONIA/person266_bacteria_1237.jpeg`, `_1238.jpeg` |
| `c7a269a8` | `train/PNEUMONIA/person30_bacteria_156.jpeg`, `_157.jpeg` |
| `23d69873` | `train/PNEUMONIA/person357_virus_734.jpeg`, `_735.jpeg` |
| `9355430a` | `train/PNEUMONIA/person401_virus_797.jpeg`, `_798.jpeg` |
| `6438db6b` | `train/PNEUMONIA/person688_virus_1281.jpeg`, `_1282.jpeg` |
| `8b643949` | `train/PNEUMONIA/person834_bacteria_2747.jpeg`, `_2748.jpeg` |
| `f9528b24` | `test/NORMAL/NORMAL2-IM-0095-0001.jpeg`, `NORMAL2-IM-0096-0001.jpeg` |
| `f430f335` | `test/NORMAL/NORMAL2-IM-0173-0001-0001.jpeg`, `NORMAL2-IM-0173-0001-0002.jpeg` |
| `05c43c85` | `test/NORMAL/NORMAL2-IM-0246-0001-0001.jpeg`, `NORMAL2-IM-0246-0001-0002.jpeg` |
| `62f9117f` | `test/PNEUMONIA/person128_bacteria_606.jpeg`, `_607.jpeg` |
| `fe703b7a` | `test/PNEUMONIA/person134_bacteria_643.jpeg`, `_644.jpeg` |
| `adf8b53d` | `test/PNEUMONIA/person1661_virus_2872.jpeg`, `_2873.jpeg` |

*Note:* Only the 30 groups / 62 files listed above have been verified as exact pixel-identical duplicates via MD5 hash comparison. Unverified consecutive filenames are NOT assumed identical.

### Duplicate Handling Strategy
- **Raw dataset remains untouched:** No duplicate image files are deleted, renamed, or moved in `data/raw/archive/chest_xray/`.
- **Handling via Split Manifests:** Redundant duplicate copies will be excluded during CSV creation (`make_splits.py`), retaining only the first canonical file path per unique MD5 hash.
- **Official Test Set Integrity:** The official test set files remain unaltered on disk.

---

## 10. Read-Only Confirmation

During the entire Dataset QA process, the repository and dataset state has been strictly preserved:
- [x] **No files moved**
- [x] **No files renamed**
- [x] **No files deleted**
- [x] **No raw images modified**

---

## 11. Summary Table

| # | Check | Result | Action Required |
|---|---|---|---|
| 1 | Dataset root | `data/raw/archive/chest_xray/` confirmed | None |
| 2 | Image counts | Match expected (5 856 total) | None |
| 3 | Total images | **5 856** | None |
| 4 | Extensions | 100 % `.jpeg` | None |
| 5 | Corrupt images | **0** | None |
| 6a | Dimension range | 384–2 916 × 127–2 713 px (variable) | Resize to 224×224 in preprocessing |
| 6b | Unusually small | None | None |
| 6c | Color modes | 5 573 grayscale + 283 RGB (100 % RGB in PNEUMONIA) | Standardize channel mode in preprocessing |
| 6d | Extreme aspect ratios | Ratios span 0.835 to 3.379 | Standardize resize / padding in preprocessing |
| 7 | Patient IDs | 5 856 / 5 856 successfully extracted (0 missing) | Ready for patient-level grouping |
| 8a | Patient overlap: train vs val | **0 IDs** | None |
| 8b | Patient overlap: train vs test | **264 IDs** (94 NORMAL, 170 PNEUMONIA) | Severe leakage in Kaggle split; decide strategy |
| 8c | Patient overlap: val vs test | **0 IDs** | None |
| 9a | Same-split duplicates | **30 groups / 62 files** in train + test | Filter out redundant copies in CSV |
| 9b | Cross-split duplicates | **None** | None |
| 10 | Raw dataset status | **100 % read-only confirmed** | None |
| — | Val set size | Only 16 images in default val/ | Re-carve proper validation set from train |
| — | Class imbalance | 2.89 : 1 (PNEUMONIA : NORMAL) | Use class weights in training loss |

---

## Split Readiness

### Patient Overlap Summary
The QA analysis proves that:
1. **Existing `val/` is statistically useless:** Only 16 images exist in `val/` (8 NORMAL, 8 PNEUMONIA).
2. **Existing `train/` and `test/` have severe patient contamination:** 264 patient IDs (94 NORMAL, 170 PNEUMONIA) are shared between `train/` and `test/`. Over 84% of test pneumonia patients have training images.

### Readiness Verdict
**The dataset is READY for patient-level splitting script implementation (`make_splits.py`), subject to architectural alignment on test set handling.**

Patient ID extraction is 100% complete and validated across all 5,856 images with zero syntax misses, and MD5 duplicates are mapped.

### Issues to Resolve Before Creating `make_splits.py`

1. **Test Set Splitting Policy (Benchmark vs. Pure Patient Stratification):**
   - **Approach 1 (Official Test Benchmark):** Keep `test/` identical to the official Kaggle test set to evaluate directly against published literature, and carve a patient-grouped validation set (~15–20%) exclusively out of `train/` (with train duplicates dropped). In this approach, train vs. val has **0 patient overlap**, while train vs. test leakage is clearly reported as an inherent benchmark artifact.
   - **Approach 2 (Zero-Leakage Re-split):** Pool all 5,856 images, drop the 32 duplicate copies, and perform a fresh 70/15/15 patient-stratified split ensuring **0 patient overlap across train, val, and test**.
   - **Approach 3 (Dual Splits):** Generate `official_clean` (official test set + clean train/val) and `patient_strict` (leakage-free 3-way split) to allow reporting both figures in the final project paper.
2. **Duplicate Exclusion in CSVs:**
   - 32 redundant duplicate copies (from the 30 groups) must be excluded from the generated split manifests, leaving the raw image directory unmodified.
3. **Channel Standardization in Preprocessing:**
   - The 283 RGB images are exclusively in PNEUMONIA. Preprocessing pipeline must ensure channel normalization so the model cannot use color channels as a spurious predictive feature.

