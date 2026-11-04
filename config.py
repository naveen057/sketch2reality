"""Central settings. Edit CATEGORIES to change what the model can recognise."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models_saved"
METRICS_FILE = MODELS_DIR / "metrics.json"

# Quick Draw category names (must match the dataset file names, no spaces).
CATEGORIES = [
    "airplane", "apple", "bicycle", "bird", "book",
    "car", "cat", "chair", "clock", "cup",
    "dog", "fish", "flower", "guitar", "house",
    "moon", "star", "sun", "tree", "umbrella",
]

SAMPLES_PER_CLASS = 10000   # lower this (e.g. 3000) for a faster first run
IMG_SIZE = 28               # Quick Draw numpy bitmaps are 28x28
