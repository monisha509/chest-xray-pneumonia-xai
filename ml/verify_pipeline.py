"""
ChestXR — Preprocessing Verification & Visualization
=====================================================
Executes the three specific pipeline tests:
  1. shuffle=False, augment=False -> first batch identical across epochs
  2. shuffle=False, augment=True  -> first batch different across epochs
  3. shuffle=True,  augment=False -> batch ordering changes across epochs

Verifies val.csv comparison and generates two visual PNG grids:
  - 16 preprocessed training images with labels
  - 8 augmented versions of the same image
"""

import csv
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.preprocessing import (
    PROJECT_ROOT,
    preprocess_image,
    load_dataset,
    get_training_augmentation,
    compute_class_weights,
)

DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

GRID_16_PATH = DOCS_DIR / "preprocessed_training_grid.png"
GRID_AUG_PATH = DOCS_DIR / "augmented_samples_grid.png"


def run_pipeline_tests():
    print("=" * 70)
    print("STEP 1: THREE SPECIFIC PIPELINE TESTS")
    print("=" * 70)

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"

    # Test 1: shuffle=False, augment=False -> first batch identical across two epochs
    print("\n[Test 1] shuffle=False, augment=False:")
    ds1 = load_dataset(train_csv, batch_size=8, shuffle=False, augment=False)
    epoch1_b0 = next(iter(ds1))[0].numpy()
    epoch2_b0 = next(iter(ds1))[0].numpy()
    test1_identical = np.array_equal(epoch1_b0, epoch2_b0)
    print(f"  First batch identical across 2 epochs: {test1_identical}")
    assert test1_identical, "Test 1 Failed: Expected identical batches without shuffle and augmentation."
    print("  -> TEST 1 PASSED: Deterministic and frozen as expected.")

    # Test 2: shuffle=False, augment=True -> first batch different across two epochs
    print("\n[Test 2] shuffle=False, augment=True:")
    ds2 = load_dataset(train_csv, batch_size=8, shuffle=False, augment=True)
    epoch1_b0_aug = next(iter(ds2))[0].numpy()
    epoch2_b0_aug = next(iter(ds2))[0].numpy()
    test2_different = not np.array_equal(epoch1_b0_aug, epoch2_b0_aug)
    diff2 = np.max(np.abs(epoch1_b0_aug - epoch2_b0_aug))
    print(f"  First batch different across 2 epochs: {test2_different} (max pixel diff: {diff2:.2f})")
    assert test2_different, "Test 2 Failed: Expected different batches with fresh augmentation."
    print("  -> TEST 2 PASSED: Fresh augmentation generated on each epoch.")

    # Test 3: shuffle=True, augment=False -> batch ordering changes across epochs
    print("\n[Test 3] shuffle=True, augment=False:")
    ds3 = load_dataset(train_csv, batch_size=8, shuffle=True, seed=42, augment=False)
    epoch1_b0_shuf = next(iter(ds3))[0].numpy()
    epoch2_b0_shuf = next(iter(ds3))[0].numpy()
    test3_different = not np.array_equal(epoch1_b0_shuf, epoch2_b0_shuf)
    print(f"  First batch ordering changed across 2 epochs: {test3_different}")
    assert test3_different, "Test 3 Failed: Expected different batch ordering across epochs."
    print("  -> TEST 3 PASSED: Reshuffle_each_iteration=True reorders batches each epoch.")

    # Shared preprocessing comparison test confirmation
    print("\n" + "=" * 70)
    print("STEP 2: CONFIRM SHARED-PREPROCESSING COMPARISON USES val.csv")
    print("=" * 70)
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"
    with open(val_csv, "r", encoding="utf-8") as f:
        first_val_row = next(csv.DictReader(f))
    first_val_path = first_val_row["filepath"]
    expected_tensor = preprocess_image(first_val_path)

    val_ds = load_dataset(val_csv, batch_size=4, shuffle=False, augment=False)
    actual_first_tensor = next(iter(val_ds))[0][0].numpy()

    is_identical = np.array_equal(actual_first_tensor, expected_tensor)
    print(f"  Source CSV manifest verified : {val_csv.relative_to(PROJECT_ROOT)}")
    print(f"  First validation image path  : {first_val_path}")
    print(f"  Exact tensor match           : {is_identical}")
    assert is_identical, "Shared preprocessing comparison failed on val.csv"
    print("  -> CONFIRMATION PASSED: Exactly uses data/splits/val.csv with bitwise parity.")


def generate_png_grid_16():
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING PNG GRID OF 16 PREPROCESSED TRAINING IMAGES")
    print("=" * 70)

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    normal_samples = []
    pneumonia_samples = []

    with open(train_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["class_name"] == "NORMAL" and len(normal_samples) < 8:
                normal_samples.append(r)
            elif r["class_name"] == "PNEUMONIA" and len(pneumonia_samples) < 8:
                pneumonia_samples.append(r)
            if len(normal_samples) == 8 and len(pneumonia_samples) == 8:
                break

    # Interleave 8 NORMAL and 8 PNEUMONIA
    samples = []
    for n, p in zip(normal_samples, pneumonia_samples):
        samples.extend([n, p])

    # 4 columns x 4 rows
    cols, rows = 4, 4
    tile_w, tile_h = 224, 224
    header_h = 32
    padding = 8

    canvas_w = cols * tile_w + (cols + 1) * padding
    canvas_h = rows * (tile_h + header_h) + (rows + 1) * padding

    grid_img = Image.new("RGB", (canvas_w, canvas_h), color=(20, 24, 33))
    draw = ImageDraw.Draw(grid_img)
    font = ImageFont.load_default()

    for idx, item in enumerate(samples):
        c = idx % cols
        r = idx // cols
        x = padding + c * (tile_w + padding)
        y = padding + r * (tile_h + header_h + padding)

        # Preprocess image
        arr = preprocess_image(item["filepath"])
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        tile_img = Image.fromarray(arr_uint8)

        # Draw header banner
        cls_name = item["class_name"]
        lbl = item["label"]
        bg_color = (24, 80, 50) if lbl == "0" else (100, 35, 35)
        text_color = (120, 240, 160) if lbl == "0" else (255, 140, 140)
        draw.rectangle([x, y, x + tile_w, y + header_h], fill=bg_color)
        title_text = f"#{idx+1}: {cls_name} (label={lbl})"
        draw.text((x + 8, y + 8), title_text, fill=text_color, font=font)

        # Paste image
        grid_img.paste(tile_img, (x, y + header_h))

    grid_img.save(GRID_16_PATH, "PNG")
    print(f"  Saved 16-image grid to: {GRID_16_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Dimensions: {canvas_w} x {canvas_h} px")


def generate_png_grid_augmented_8():
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING PNG OF 8 AUGMENTED VERSIONS OF SAME IMAGE")
    print("=" * 70)

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f))
    sample_path = first_row["filepath"]
    base_arr = preprocess_image(sample_path)  # shape (224, 224, 3)

    # Convert to tensor batch (1, 224, 224, 3)
    import tensorflow as tf
    base_tensor = tf.convert_to_tensor(base_arr[np.newaxis, ...], dtype=tf.float32)

    aug_layer = get_training_augmentation()

    # Generate 1 original + 7 augmented versions
    versions = [("Original (Unaugmented)", base_arr)]
    for i in range(1, 8):
        aug_tensor = aug_layer(base_tensor, training=True)
        aug_arr = aug_tensor.numpy()[0]
        versions.append((f"Augmented Variant #{i}", aug_arr))

    # Layout: 4 columns x 2 rows
    cols, rows = 4, 2
    tile_w, tile_h = 224, 224
    header_h = 32
    padding = 8

    canvas_w = cols * tile_w + (cols + 1) * padding
    canvas_h = rows * (tile_h + header_h) + (rows + 1) * padding

    grid_img = Image.new("RGB", (canvas_w, canvas_h), color=(20, 24, 33))
    draw = ImageDraw.Draw(grid_img)
    font = ImageFont.load_default()

    for idx, (title, arr) in enumerate(versions):
        c = idx % cols
        r = idx // cols
        x = padding + c * (tile_w + padding)
        y = padding + r * (tile_h + header_h + padding)

        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        tile_img = Image.fromarray(arr_uint8)

        bg_color = (30, 60, 90) if idx == 0 else (45, 45, 60)
        text_color = (130, 210, 255) if idx == 0 else (220, 220, 230)
        draw.rectangle([x, y, x + tile_w, y + header_h], fill=bg_color)
        draw.text((x + 8, y + 8), title, fill=text_color, font=font)

        grid_img.paste(tile_img, (x, y + header_h))

    grid_img.save(GRID_AUG_PATH, "PNG")
    print(f"  Saved 8-augmentation grid to: {GRID_AUG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Dimensions: {canvas_w} x {canvas_h} px")


def main():
    run_pipeline_tests()
    generate_png_grid_16()
    generate_png_grid_augmented_8()
    print("\n" + "=" * 70)
    print("ALL TESTS AND VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
