import base64
import io

import numpy as np
import torch
from PIL import Image


def decode_image(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data_url)))


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ink_map(img: Image.Image) -> np.ndarray:
    """Dark strokes on a light canvas -> float array, strokes high (0..255)."""
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        img = bg
    return 255.0 - np.asarray(img.convert("L"), dtype=np.float32)


def square_crop(ink: np.ndarray, margin_frac=0.12):
    """Crop to the drawing, centre it in a square with margin. Returns None if blank."""
    ys, xs = np.where(ink > 25)
    if len(xs) == 0:
        return None
    crop = ink[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    sq[oy: oy + h, ox: ox + w] = crop
    m = max(2, int(side * margin_frac))
    return np.pad(sq, m)


def sketch_to_tensor(img: Image.Image):
    """Canvas PNG -> (1,1,28,28) tensor matching Quick Draw bitmaps (white strokes, black bg)."""
    sq = square_crop(ink_map(img))
    if sq is None:
        return None
    small = Image.fromarray(sq.clip(0, 255).astype(np.uint8)).resize((28, 28), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    if arr.max() > 0:
        arr /= arr.max()
    return torch.from_numpy(arr)[None, None]
