"""FastAPI backend: recognition, generation, description, comparison + serves the frontend."""
import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import generation
from config import CATEGORIES, METRICS_FILE, MODELS_DIR, ROOT
from core.models import SketchCNN, TinyViT
from core.preprocess import decode_image, sketch_to_tensor, to_data_url

app = FastAPI(title="Sketch2Reality")

CLASSIFIERS = {}


def load_classifiers():
    for name, cls in (("cnn", SketchCNN), ("vit", TinyViT)):
        path = MODELS_DIR / f"{name}.pt"
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            model = cls(len(ckpt["categories"]))
            model.load_state_dict(ckpt["state_dict"])
            CLASSIFIERS[name] = model.eval()
            print(f"Loaded {name} (val_acc={ckpt.get('val_acc', 0):.3f})")
        else:
            print(f"[warn] {path.name} not found - run training.train_{name}")


load_classifiers()


class RecognizeIn(BaseModel):
    image: str


class GenerateIn(BaseModel):
    image: str
    label: str | None = None
    prompt: str | None = None


def top_k(model, x, k=3):
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.topk(k)
    return [{"label": CATEGORIES[i], "confidence": round(c, 4)} for c, i in zip(conf.tolist(), idx.tolist())]


@app.get("/api/health")
def health():
    return {
        "classifiers": list(CLASSIFIERS),
        "diffusion_enabled": generation.ENABLE_DIFFUSION,
        "caption_enabled": generation.ENABLE_CAPTION,
        "cgan_available": generation.cgan_available(),
        "categories": CATEGORIES,
    }


@app.get("/api/metrics")
def metrics():
    return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}


@app.post("/api/recognize")
def recognize(body: RecognizeIn):
    if not CLASSIFIERS:
        raise HTTPException(503, "No trained models found. Run the training scripts first (see README).")
    x = sketch_to_tensor(decode_image(body.image))
    if x is None:
        raise HTTPException(400, "The canvas is empty. Draw something first.")
    return {"predictions": {name: top_k(m, x) for name, m in CLASSIFIERS.items()}}


@app.post("/api/generate")
def generate(body: GenerateIn):
    sketch = decode_image(body.image)
    label = body.label
    if label not in CATEGORIES:
        if not CLASSIFIERS:
            raise HTTPException(503, "No trained models found, so the sketch can't be labelled.")
        x = sketch_to_tensor(sketch)
        if x is None:
            raise HTTPException(400, "The canvas is empty. Draw something first.")
        label = top_k(next(iter(CLASSIFIERS.values())), x, 1)[0]["label"]

    prompt = (body.prompt or "").strip() or f"a realistic photo of a {label}, high quality, detailed"

    if generation.ENABLE_DIFFUSION:
        try:
            out, ctrl = generation.diffusion_generate(sketch, prompt)
        except Exception as exc:
            raise HTTPException(500, f"Diffusion failed: {exc}")
        score = generation.structure_score(ctrl, out)
        mode = "diffusion"
    elif generation.cgan_available():
        out, score, mode = generation.cgan_generate(label), None, "cgan"
    else:
        raise HTTPException(
            503,
            "No generator available. Enable diffusion (ENABLE_DIFFUSION=1) or train the cGAN "
            "(python -m training.train_cgan). See README.",
        )

    return {
        "label": label,
        "mode": mode,
        "prompt": prompt,
        "image": to_data_url(out),
        "caption": generation.describe(out, label),
        "structure_score": score,
    }


FRONTEND = ROOT / "frontend"


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
