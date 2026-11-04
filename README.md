# Sketch2Reality

Draw on a canvas. A CNN and a Vision Transformer recognise the sketch, and a generator turns it into an image you can compare with your drawing.

| Feature | How it works |
|---|---|
| Draw with mouse/touch | HTML canvas, pen, eraser, undo, clear |
| Recognise sketches | CNN + Vision Transformer trained on Google Quick Draw (20 categories) |
| Sketch to image | Stable Diffusion + ControlNet scribble (GPU), or a conditional GAN (CPU-friendly) |
| Image description | BLIP captioning (optional), template text otherwise |
| Compare | Sketch, generated image, label, caption and a sketch-match score side by side |

## Project layout
```
config.py              categories, sample counts, paths
core/                  models.py (CNN, ViT, cGAN), data.py, preprocess.py, train_utils.py
training/              download_data.py, train_cnn.py, train_vit.py, train_cgan.py
backend/               main.py (FastAPI), generation.py (diffusion, cGAN, caption, score)
frontend/index.html    the web app (served by FastAPI)
data/                  Quick Draw .npy files (downloaded)
models_saved/          trained weights + metrics.json (created by training)
```

## Setup (run every command from the project root)

**1. Requirements:** Python 3.10 or newer, about 2 GB free disk, internet for the first run.

**2. Create a virtual environment and install packages**
```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**3. Download the dataset** (about 20 files of roughly 100 MB each)
```bash
python -m training.download_data
```

**4. Train the recognisers** (run a quick test first with fewer samples)
```bash
python -m training.train_cnn --samples 3000 --epochs 3     # quick test, about 1-2 min on CPU
python -m training.train_cnn                                # full run, about 90-95% accuracy
python -m training.train_vit                                # full run, slower than the CNN on CPU
```
Weights go to `models_saved/cnn.pt` and `vit.pt`; accuracy figures go to `metrics.json` and appear in the app.

**5. Train the conditional GAN** (needed for generation unless you use diffusion)
```bash
python -m training.train_cgan
```
Check `models_saved/cgan_samples.png`: each row is one category.

**6. Start the app**
```bash
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000 and draw.

## Optional: realistic images with Stable Diffusion + ControlNet
This needs a GPU (NVIDIA, 6 GB+ VRAM) for reasonable speed. On a CPU it works but takes minutes per image.
```bash
pip install -r requirements-optional.txt
# Mac/Linux
ENABLE_DIFFUSION=1 ENABLE_CAPTION=1 uvicorn backend.main:app
# Windows PowerShell
$env:ENABLE_DIFFUSION="1"; $env:ENABLE_CAPTION="1"; uvicorn backend.main:app
```
The first generation downloads about 5 GB of model weights from Hugging Face. No GPU? Run the same steps on Google Colab or Kaggle and expose the app with a tunnel, or keep using the cGAN.
If you installed a CPU-only PyTorch, install the CUDA build from https://pytorch.org/get-started/locally/ first.

## API
| Endpoint | Purpose |
|---|---|
| `POST /api/recognize` `{image}` | top-3 predictions from each model |
| `POST /api/generate` `{image, label?, prompt?}` | generated image, caption, sketch-match score |
| `GET /api/metrics` | validation accuracy, parameters, train time |
| `GET /api/health` | which models and generators are loaded |

## Customising
- **Categories:** edit `CATEGORIES` in `config.py` (names must match the Quick Draw list), then re-download and retrain.
- **Accuracy:** raise `SAMPLES_PER_CLASS` or `--epochs`.
- **Image size:** classifiers use 28x28 like the dataset; drawings are cropped, centred and resized the same way (`core/preprocess.py`).

## Troubleshooting
- **"No trained models found"**: run steps 3 and 4, then restart uvicorn.
- **"No generator available"**: train the cGAN (step 5) or enable diffusion.
- **Wrong predictions on easy drawings**: draw big, use thick strokes, one object, and keep to the 20 categories.
- **CUDA out of memory**: close other GPU apps, or run without `ENABLE_DIFFUSION`.
- **Model downloads fail**: check your internet connection and retry; Hugging Face downloads resume.

## Limits worth knowing
Quick Draw has fixed categories, so the recogniser cannot name anything outside them. The cGAN produces low-resolution sketches, not photos. The sketch-match score is a rough edge-overlap measure, not a quality rating.
"# sketch2reality" 
"# sketch2reality" 
