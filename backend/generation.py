"""Sketch -> image generation, captioning and sketch/image comparison.

Heavy models are optional and loaded lazily:
  ENABLE_DIFFUSION=1  Stable Diffusion + ControlNet (scribble). Needs a GPU for sensible speed.
  ENABLE_CAPTION=1    BLIP image captioning.
Without them the app falls back to the conditional GAN and template descriptions.
"""
import os

import numpy as np
import torch
from PIL import Image, ImageFilter

from config import CATEGORIES, MODELS_DIR
from core.models import CGANGenerator
from core.preprocess import ink_map, square_crop

ENABLE_DIFFUSION = os.getenv("ENABLE_DIFFUSION", "0") == "1"
ENABLE_CAPTION = os.getenv("ENABLE_CAPTION", "0") == "1"

_state = {"pipe": None, "captioner": None, "cgan": None, "cgan_checked": False}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------- sketch preparation ----------
def control_image(sketch: Image.Image, size: int) -> Image.Image:
    """White strokes on black, centred and scaled, thickened - the format ControlNet scribble expects."""
    sq = square_crop(ink_map(sketch), margin_frac=0.1)
    if sq is None:
        raise ValueError("blank sketch")
    img = Image.fromarray(sq.clip(0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    img = img.filter(ImageFilter.MaxFilter(5))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / max(arr.max(), 1) * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


# ---------- diffusion ----------
def _get_pipe():
    if _state["pipe"] is None:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

        dtype = torch.float16 if _device() == "cuda" else torch.float32
        cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=dtype
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        _state["pipe"] = pipe.to(_device())
    return _state["pipe"]


def diffusion_generate(sketch: Image.Image, prompt: str):
    size = 512 if _device() == "cuda" else 384
    ctrl = control_image(sketch, size)
    steps = 25 if _device() == "cuda" else 15
    out = _get_pipe()(
        prompt, image=ctrl, num_inference_steps=steps, guidance_scale=7.5,
        negative_prompt="blurry, low quality, deformed, text, watermark",
    ).images[0]
    return out, ctrl


# ---------- conditional GAN fallback ----------
def cgan_available() -> bool:
    if not _state["cgan_checked"]:
        _state["cgan_checked"] = True
        path = MODELS_DIR / "cgan_generator.pt"
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            g = CGANGenerator(len(ckpt["categories"]), ckpt["z_dim"])
            g.load_state_dict(ckpt["state_dict"])
            _state["cgan"] = g.eval()
    return _state["cgan"] is not None


def cgan_generate(label: str) -> Image.Image:
    g = _state["cgan"]
    idx = CATEGORIES.index(label)
    with torch.no_grad():
        out = g(torch.randn(1, g.z_dim), torch.tensor([idx]))[0, 0].numpy()
    arr = ((out + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(255 - arr).resize((256, 256), Image.BICUBIC)  # dark strokes on white
    return img.convert("RGB")


# ---------- captioning ----------
def describe(image: Image.Image, label: str) -> str:
    if ENABLE_CAPTION:
        try:
            if _state["captioner"] is None:
                from transformers import BlipForConditionalGeneration, BlipProcessor
                name = "Salesforce/blip-image-captioning-base"
                _state["captioner"] = (BlipProcessor.from_pretrained(name),
                                       BlipForConditionalGeneration.from_pretrained(name))
            proc, model = _state["captioner"]
            inputs = proc(image.convert("RGB"), return_tensors="pt")
            ids = model.generate(**inputs, max_new_tokens=30)
            return proc.decode(ids[0], skip_special_tokens=True).capitalize() + "."
        except Exception as exc:  # fall through to template
            print("Captioning failed, using template:", exc)
    return f"A generated picture based on your sketch of a {label}."


# ---------- comparison ----------
def _edge_vec(img: Image.Image, size=48) -> np.ndarray:
    g = img.convert("L").resize((size, size), Image.LANCZOS).filter(ImageFilter.FIND_EDGES)
    g = g.filter(ImageFilter.GaussianBlur(1.5))
    v = np.asarray(g, dtype=np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def structure_score(control: Image.Image, generated: Image.Image) -> float:
    """Rough 0-100 score: how well edges of the output line up with the sketch strokes."""
    a = np.asarray(control.convert("L").resize((48, 48), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.5)),
                   dtype=np.float32).ravel()
    a -= a.mean()
    a /= (np.linalg.norm(a) + 1e-6)
    sim = float(np.dot(a, _edge_vec(generated)))
    return round(max(0.0, sim) * 100, 1)
