import argparse

from config import CATEGORIES
from core.models import SketchCNN
from core.train_utils import fit

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--samples", type=int, default=None, help="samples per class (default: config)")
    a = p.parse_args()
    fit(SketchCNN(len(CATEGORIES)), "cnn", a.epochs, a.lr, a.samples)
