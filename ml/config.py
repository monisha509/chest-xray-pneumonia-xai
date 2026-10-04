"""
ChestXR — Project Configuration
==============================
Central configuration for data paths, splitting parameters, and exclusions.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "archive" / "chest_xray"
SPLITS_DIR   = DATA_DIR / "splits"

# Random Seed & Split Ratio
RANDOM_SEED = 42
TRAIN_RATIO = 0.80  # 80% train / 20% val at group_id level

# All 7 extreme-aspect-ratio strip images inspected and confirmed for exclusion
EXCLUDE_FILES = [
    "train/PNEUMONIA/person325_virus_661.jpeg",
    "train/PNEUMONIA/person1684_bacteria_4462.jpeg",
    "train/PNEUMONIA/person325_virus_659.jpeg",
    "train/PNEUMONIA/person1732_bacteria_4564.jpeg",
    "train/PNEUMONIA/person1719_bacteria_4541.jpeg",
    "train/PNEUMONIA/person1689_bacteria_4474.jpeg",
    "train/PNEUMONIA/person407_virus_811.jpeg",
]
