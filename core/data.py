import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CATEGORIES, DATA_DIR, SAMPLES_PER_CLASS


def load_dataset(samples_per_class=None, seed=42):
    """Returns X (N,1,28,28) float in [0,1] and y (N,) int64."""
    samples_per_class = samples_per_class or SAMPLES_PER_CLASS
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for idx, name in enumerate(CATEGORIES):
        path = DATA_DIR / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python -m training.download_data")
        arr = np.load(path, mmap_mode="r")
        n = min(samples_per_class, len(arr))
        sel = np.sort(rng.choice(len(arr), n, replace=False))
        xs.append(np.asarray(arr[sel]).reshape(-1, 28, 28))
        ys.append(np.full(n, idx))
    X = np.concatenate(xs).astype(np.float32) / 255.0
    y = np.concatenate(ys).astype(np.int64)
    perm = rng.permutation(len(X))
    return torch.from_numpy(X[perm]).unsqueeze(1), torch.from_numpy(y[perm])


def make_loaders(batch_size=256, val_frac=0.1, samples_per_class=None):
    X, y = load_dataset(samples_per_class)
    n_val = int(len(X) * val_frac)
    train = TensorDataset(X[n_val:], y[n_val:])
    val = TensorDataset(X[:n_val], y[:n_val])
    return DataLoader(train, batch_size, shuffle=True), DataLoader(val, batch_size)


def augment(x, max_shift=2):
    """Cheap augmentation: random shift, so hand-drawn canvas input generalises better."""
    dx, dy = np.random.randint(-max_shift, max_shift + 1, size=2)
    return torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
