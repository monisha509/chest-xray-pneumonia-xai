"""
ChestXR — Group-ID Based Stratified Data Splitting
=================================================
Pipeline:
  Original train + val (5,232 images)
        ↓
  MD5 deduplicate train/val (drops redundant copies)
        ↓
  Remove train/val group IDs overlapping test (prevents train-test leakage)
        ↓
  Remove approved EXCLUDE_FILES (extreme aspect-ratio strips)
        ↓
  Group-ID stratified split (80% train / 20% val at group ID level)
        ↓
  Generated manifests in data/splits/:
    - train.csv
    - val.csv
    - test.csv (624 official test images preserved)
    - excluded_images.csv (transparent audit trail of all exclusions)

Zero raw images or files are modified, moved, renamed, or deleted.
All groupings represent ID numbers / group IDs from file metadata
and do NOT constitute verified clinical patient identities.
"""

import csv
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    EXCLUDE_FILES,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SPLITS_DIR,
    TRAIN_RATIO,
)

# Regex extraction for group IDs
RE_NORMAL = re.compile(r"^(?:NORMAL2?-)?IM-(\d+)", re.IGNORECASE)
RE_PNEUMONIA = re.compile(r"^person(\d+)_(bacteria|virus)_", re.IGNORECASE)


def extract_metadata(filename: str, class_name: str):
    """Extract group_id and sub_type from filename."""
    if class_name == "PNEUMONIA":
        m = RE_PNEUMONIA.match(filename)
        if m:
            return m.group(1), m.group(2).lower()
        return "UNKNOWN", "pneumonia"
    else:  # NORMAL
        m = RE_NORMAL.match(filename)
        if m:
            return m.group(1), "normal"
        return "UNKNOWN", "normal"


def compute_md5(filepath: Path) -> str:
    """Compute MD5 hash in chunks."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    exclude_set = set(EXCLUDE_FILES)

    # ── 1. Process Official Test Set ──────────────────────────────────────────
    test_records = []
    test_group_ids = {"NORMAL": set(), "PNEUMONIA": set()}

    for class_name in ["NORMAL", "PNEUMONIA"]:
        test_dir = RAW_DATA_DIR / "test" / class_name
        label = 0 if class_name == "NORMAL" else 1
        for img_path in sorted(test_dir.glob("*.jpeg")):
            gid, sub_type = extract_metadata(img_path.name, class_name)
            test_group_ids[class_name].add(gid)
            rel_to_proj = img_path.relative_to(PROJECT_ROOT).as_posix()
            test_records.append({
                "filepath": rel_to_proj,
                "filename": img_path.name,
                "group_id": gid,
                "patient_id": gid,
                "class_name": class_name,
                "label": label,
                "sub_type": sub_type,
            })

    # ── 2. Collect Original Train + Val ───────────────────────────────────────
    train_val_pool = []
    for split in ["train", "val"]:
        for class_name in ["NORMAL", "PNEUMONIA"]:
            folder = RAW_DATA_DIR / split / class_name
            label = 0 if class_name == "NORMAL" else 1
            for img_path in sorted(folder.glob("*.jpeg")):
                gid, sub_type = extract_metadata(img_path.name, class_name)
                rel_split_path = f"{split}/{class_name}/{img_path.name}"
                rel_to_proj = img_path.relative_to(PROJECT_ROOT).as_posix()
                train_val_pool.append({
                    "full_path": img_path,
                    "rel_split_path": rel_split_path,
                    "filepath": rel_to_proj,
                    "filename": img_path.name,
                    "group_id": gid,
                    "patient_id": gid,
                    "class_name": class_name,
                    "label": label,
                    "sub_type": sub_type,
                })

    # ── 3. MD5 Deduplication in Train/Val ──────────────────────────────────────
    seen_hashes = {}
    duplicate_map = {}  # rel_split_path -> (canonical_path, hash)

    for item in train_val_pool:
        h = compute_md5(item["full_path"])
        item["md5"] = h
        if h in seen_hashes:
            duplicate_map[item["rel_split_path"]] = (seen_hashes[h]["rel_split_path"], h)
        else:
            seen_hashes[h] = item

    # ── 4. Audit & Filter Exclusions ──────────────────────────────────────────
    excluded_records = []
    clean_pool = []

    for item in train_val_pool:
        rel_path = item["rel_split_path"]
        gid = item["group_id"]
        cls = item["class_name"]

        # 1: Redundant MD5 duplicate
        if rel_path in duplicate_map:
            canonical_path, h = duplicate_map[rel_path]
            excluded_records.append({
                "filepath": item["filepath"],
                "filename": item["filename"],
                "group_id": gid,
                "patient_id": gid,
                "class_name": cls,
                "reason": "duplicate_md5",
                "details": f"Pixel-identical copy of {canonical_path} (MD5: {h[:8]})",
            })
        # 2: Test group overlap (prevents train-test leakage)
        elif gid in test_group_ids[cls]:
            excluded_records.append({
                "filepath": item["filepath"],
                "filename": item["filename"],
                "group_id": gid,
                "patient_id": gid,
                "class_name": cls,
                "reason": "test_group_overlap",
                "details": f"Group ID {gid} appears in test set (excluded to prevent train-test leakage)",
            })
        # 3: Approved extreme aspect ratio strip
        elif rel_path in exclude_set:
            excluded_records.append({
                "filepath": item["filepath"],
                "filename": item["filename"],
                "group_id": gid,
                "patient_id": gid,
                "class_name": cls,
                "reason": "extreme_aspect_ratio",
                "details": "Severely cropped horizontal strip artifact",
            })
        # Retain clean image
        else:
            clean_pool.append(item)

    # ── 5. Group-ID Stratified Train/Val Split ────────────────────────────────
    groups_by_class = {
        "NORMAL": defaultdict(list),
        "PNEUMONIA": defaultdict(list),
    }
    for item in clean_pool:
        groups_by_class[item["class_name"]][item["group_id"]].append(item)

    train_records = []
    val_records = []
    rng = random.Random(RANDOM_SEED)

    for class_name in ["NORMAL", "PNEUMONIA"]:
        gids = sorted(list(groups_by_class[class_name].keys()))
        rng.shuffle(gids)

        n_train = int(len(gids) * TRAIN_RATIO)
        train_gids = set(gids[:n_train])
        val_gids = set(gids[n_train:])

        for gid in sorted(list(train_gids)):
            for item in groups_by_class[class_name][gid]:
                train_records.append({
                    "filepath": item["filepath"],
                    "filename": item["filename"],
                    "group_id": item["group_id"],
                    "patient_id": item["patient_id"],
                    "class_name": item["class_name"],
                    "label": item["label"],
                    "sub_type": item["sub_type"],
                })

        for gid in sorted(list(val_gids)):
            for item in groups_by_class[class_name][gid]:
                val_records.append({
                    "filepath": item["filepath"],
                    "filename": item["filename"],
                    "group_id": item["group_id"],
                    "patient_id": item["patient_id"],
                    "class_name": item["class_name"],
                    "label": item["label"],
                    "sub_type": item["sub_type"],
                })

    # Sort deterministically by filepath
    train_records.sort(key=lambda x: x["filepath"])
    val_records.sort(key=lambda x: x["filepath"])
    test_records.sort(key=lambda x: x["filepath"])
    excluded_records.sort(key=lambda x: x["filepath"])

    # ── 6. Write CSV Manifests ────────────────────────────────────────────────
    split_cols = ["filepath", "filename", "group_id", "patient_id", "class_name", "label", "sub_type"]
    exclude_cols = ["filepath", "filename", "group_id", "patient_id", "class_name", "reason", "details"]

    def write_csv(filename, records, fieldnames):
        out_path = SPLITS_DIR / filename
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    write_csv("train.csv", train_records, split_cols)
    write_csv("val.csv", val_records, split_cols)
    write_csv("test.csv", test_records, split_cols)
    write_csv("excluded_images.csv", excluded_records, exclude_cols)

    return train_records, val_records, test_records, excluded_records


def main():
    print("=" * 70)
    print("ChestXR — Group-ID Based Data Splitting Pipeline")
    print(f"Dataset root       : {RAW_DATA_DIR}")
    print(f"Splits destination : {SPLITS_DIR}")
    print(f"Random seed        : {RANDOM_SEED} | Ratio: {TRAIN_RATIO:.0%}/{(1-TRAIN_RATIO):.0%}")
    print("=" * 70)

    train_records, val_records, test_records, excluded_records = run_pipeline()

    # ── Summary Calculations ──────────────────────────────────────────────────
    reasons_breakdown = Counter(r["reason"] for r in excluded_records)

    train_groups = {"NORMAL": set(), "PNEUMONIA": set()}
    val_groups   = {"NORMAL": set(), "PNEUMONIA": set()}
    test_groups  = {"NORMAL": set(), "PNEUMONIA": set()}

    for r in train_records:
        train_groups[r["class_name"]].add(r["group_id"])
    for r in val_records:
        val_groups[r["class_name"]].add(r["group_id"])
    for r in test_records:
        test_groups[r["class_name"]].add(r["group_id"])

    # Leakage checks
    print("\n" + "=" * 70)
    print("FINAL LEAKAGE ASSERTIONS & SANITY CHECKS")
    print("=" * 70)

    for cls in ["NORMAL", "PNEUMONIA"]:
        tr_val = train_groups[cls] & val_groups[cls]
        tr_te  = train_groups[cls] & test_groups[cls]
        va_te  = val_groups[cls] & test_groups[cls]
        print(f"  [{cls}] group_id overlap train vs val  : {len(tr_val)} -> {'PASS (0 overlap)' if not tr_val else 'FAIL'}")
        print(f"  [{cls}] group_id overlap train vs test : {len(tr_te)} -> {'PASS (0 overlap)' if not tr_te else 'FAIL'}")
        print(f"  [{cls}] group_id overlap val vs test   : {len(va_te)} -> {'PASS (0 overlap)' if not va_te else 'FAIL'}")

    total_accounted = len(train_records) + len(val_records) + len(excluded_records) + len(test_records)
    print(f"\n  Total images accounted for: {total_accounted} / 5856 "
          f"({'PASS (exact match)' if total_accounted == 5856 else 'FAIL'})")

    print("\nSummary Counts:")
    print(f"  TRAIN : NORMAL={len([r for r in train_records if r['class_name']=='NORMAL']):4d}, "
          f"PNEUMONIA={len([r for r in train_records if r['class_name']=='PNEUMONIA']):4d}, "
          f"TOTAL={len(train_records):4d} | "
          f"Unique Group IDs: NORMAL={len(train_groups['NORMAL'])}, PNEUMONIA={len(train_groups['PNEUMONIA'])}")
    print(f"  VAL   : NORMAL={len([r for r in val_records if r['class_name']=='NORMAL']):4d}, "
          f"PNEUMONIA={len([r for r in val_records if r['class_name']=='PNEUMONIA']):4d}, "
          f"TOTAL={len(val_records):4d} | "
          f"Unique Group IDs: NORMAL={len(val_groups['NORMAL'])}, PNEUMONIA={len(val_groups['PNEUMONIA'])}")
    print(f"  TEST  : NORMAL={len([r for r in test_records if r['class_name']=='NORMAL']):4d}, "
          f"PNEUMONIA={len([r for r in test_records if r['class_name']=='PNEUMONIA']):4d}, "
          f"TOTAL={len(test_records):4d} | "
          f"Unique Group IDs: NORMAL={len(test_groups['NORMAL'])}, PNEUMONIA={len(test_groups['PNEUMONIA'])}")
    print(f"  EXCLUDED: TOTAL={len(excluded_records):4d} | "
          f"duplicate_md5={reasons_breakdown['duplicate_md5']}, "
          f"test_group_overlap={reasons_breakdown['test_group_overlap']}, "
          f"extreme_aspect_ratio={reasons_breakdown['extreme_aspect_ratio']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
