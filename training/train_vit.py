import argparse

from config import CATEGORIES
from core.models import TinyViT
from core.train_utils import fit

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--samples", type=int, default=None, help="samples per class (default: config)")
    a = p.parse_args()
    fit(TinyViT(len(CATEGORIES)), "vit", a.epochs, a.lr, a.samples)
