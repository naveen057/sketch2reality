import json
import time

import torch
import torch.nn as nn

from config import CATEGORIES, METRICS_FILE, MODELS_DIR
from core.data import augment, make_loaders


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def update_metrics(name, values):
    MODELS_DIR.mkdir(exist_ok=True)
    data = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    data[name] = values
    METRICS_FILE.write_text(json.dumps(data, indent=2))


def fit(model, name, epochs=8, lr=1e-3, samples_per_class=None):
    device = get_device()
    print(f"Training '{name}' on {device}")
    train_dl, val_dl = make_loaders(samples_per_class=samples_per_class)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, start = 0.0, time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = augment(x).to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        acc = evaluate(model, val_dl, device)
        print(f"epoch {epoch:02d}/{epochs}  loss {running / len(train_dl):.3f}  val_acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(
                {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                 "categories": CATEGORIES, "val_acc": acc},
                MODELS_DIR / f"{name}.pt",
            )

    update_metrics(name, {
        "val_acc": round(best, 4),
        "params": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(time.time() - start),
        "epochs": epochs,
    })
    print(f"Done. Best val_acc {best:.4f}, saved to models_saved/{name}.pt")
