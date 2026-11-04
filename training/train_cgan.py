"""Train a class-conditional GAN that draws 28x28 sketches of each category."""
import argparse

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from config import CATEGORIES, MODELS_DIR
from core.data import load_dataset
from core.models import CGANDiscriminator, CGANGenerator
from core.train_utils import get_device


def save_grid(G, device, z_dim, path, per_class=8):
    G.eval()
    n = len(CATEGORIES)
    with torch.no_grad():
        y = torch.arange(n, device=device).repeat_interleave(per_class)
        imgs = G(torch.randn(len(y), z_dim, device=device), y).cpu().numpy()[:, 0]
    imgs = ((imgs + 1) * 127.5).astype(np.uint8).reshape(n, per_class, 28, 28)
    grid = imgs.transpose(0, 2, 1, 3).reshape(n * 28, per_class * 28)
    Image.fromarray(grid).save(path)
    G.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--samples", type=int, default=5000, help="samples per class")
    a = p.parse_args()

    device = get_device()
    X, y = load_dataset(a.samples)
    X = X * 2 - 1
    dl = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True, drop_last=True)

    n_cls, z_dim = len(CATEGORIES), 100
    G, D = CGANGenerator(n_cls, z_dim).to(device), CGANDiscriminator(n_cls).to(device)
    opt_g = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    MODELS_DIR.mkdir(exist_ok=True)

    for ep in range(1, a.epochs + 1):
        for real, labels in dl:
            real, labels = real.to(device), labels.to(device)
            b = real.size(0)
            z = torch.randn(b, z_dim, device=device)
            fake = G(z, labels)

            opt_d.zero_grad()
            d_loss = bce(D(real, labels), torch.ones(b, 1, device=device)) + \
                     bce(D(fake.detach(), labels), torch.zeros(b, 1, device=device))
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            g_loss = bce(D(fake, labels), torch.ones(b, 1, device=device))
            g_loss.backward()
            opt_g.step()
        print(f"epoch {ep:02d}/{a.epochs}  D {d_loss.item():.3f}  G {g_loss.item():.3f}")

    torch.save({"state_dict": {k: v.cpu() for k, v in G.state_dict().items()},
                "categories": CATEGORIES, "z_dim": z_dim}, MODELS_DIR / "cgan_generator.pt")
    save_grid(G, device, z_dim, MODELS_DIR / "cgan_samples.png")
    print("Saved models_saved/cgan_generator.pt and cgan_samples.png (one row per category)")


if __name__ == "__main__":
    main()
