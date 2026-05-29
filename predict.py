"""
=============================================================
  predict.py — Predict Alzheimer stage from MRI images
  ─────────────────────────────────────────────────────
  Uses the Hybrid CNN + ViT + MLP model (PyTorch)

  Usage:
    python predict.py <image_path>              # single image
    python predict.py <image1> <image2> ...     # multiple images
    python predict.py <folder_path>             # all images in folder
=============================================================
"""

import os
import sys
import random
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from PIL import Image
from torchvision import transforms

from hybrid_model import AlzheimerHybridModel

# ── Config ──────────────────────────────────
MODEL_PATH = "alzheimer_hybrid_model.pth"
IMG_SIZE = 224
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
STAGE_INFO = {
    "NonDemented":       "No signs of dementia detected.",
    "VeryMildDemented":  "Very mild cognitive decline -- early warning stage.",
    "MildDemented":      "Mild cognitive impairment -- medical follow-up recommended.",
    "ModerateDemented":  "Moderate dementia -- clinical intervention required.",
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Helpers ─────────────────────────────────
def load_and_preprocess(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)        # (1, 3, 224, 224)


def predict_single(model, image_path: str, device: torch.device):
    img = load_and_preprocess(image_path).to(device)

    # Simulated metadata (age=70, MMSE=25 normalized)
    metadata = torch.tensor([[0.70, 0.83]], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(img, metadata)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    print(f"\n  [Image]  : {os.path.basename(image_path)}")
    print(f"  [Result] : {label}  ({confidence:.1f}% confidence)")
    print(f"  [Info]   : {STAGE_INFO[label]}")
    print("  +--------------------------------------+")
    for i, cname in enumerate(CLASS_NAMES):
        bar = "#" * int(probs[i] * 30)
        print(f"  | {cname:<20s} {probs[i]*100:5.1f}% {bar}")
    print("  +--------------------------------------+")

    if confidence < 60:
        print("  [!] Low confidence -- consider clinical review.")


def collect_images(args: list) -> list:
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            for f in sorted(os.listdir(arg)):
                if f.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(arg, f))
        elif os.path.isfile(arg) and arg.lower().endswith(IMAGE_EXTS):
            paths.append(arg)
        else:
            print(f"  [!] Skipping: {arg}")
    return paths


# ── Main ────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    print("=" * 50)
    print("  Alzheimer Hybrid Model Predictor")
    print("  (CNN + Vision Transformer + MLP)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Loading model: {MODEL_PATH}")

    model = AlzheimerHybridModel(num_classes=4, metadata_features=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"  [OK] Model loaded\n")

    images = collect_images(sys.argv[1:])
    if not images:
        print("  No valid images found.")
        sys.exit(1)

    print(f"  Processing {len(images)} image(s)...")
    for img_path in images:
        predict_single(model, img_path, device)

    print("\n" + "=" * 50)
    print(f"  Done -- {len(images)} prediction(s).")
    print("=" * 50)


if __name__ == "__main__":
    main()
