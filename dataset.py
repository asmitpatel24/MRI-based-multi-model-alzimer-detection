"""
=============================================================
  dataset.py — PyTorch Dataset for Alzheimer MRI + Metadata
  ─────────────────────────────────────────────────────────
  - Loads images from class-subdirectory structure
  - Augmentation for training (RandomResizedCrop, ColorJitter,
    HorizontalFlip, Rotation, CutOut)
  - Simulated clinical metadata (age, MMSE) for Kaggle prototype
  - Class-weighted sampler for imbalanced data
=============================================================
"""

import os
import random
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ── Config ──────────────────────────────────
IMG_SIZE = 224
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Simulated MMSE score ranges per class (for Kaggle prototype)
# In production with ADNI, real values would come from CSV
MMSE_RANGES = {
    "NonDemented":       (27, 30),
    "VeryMildDemented":  (21, 26),
    "MildDemented":      (15, 20),
    "ModerateDemented":  (10, 14),
}

AGE_RANGE = (55, 90)


# ── CutOut Transform ───────────────────────
class CutOut:
    """Randomly mask out a square patch from the image."""

    def __init__(self, size: int = 40):
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


# ── Dataset Class ──────────────────────────
class AlzheimerDataset(Dataset):
    """
    PyTorch dataset for Alzheimer MRI classification.

    Returns:
        image   : (3, 224, 224) tensor
        metadata: (2,) tensor  [normalized_age, normalized_mmse]
        label   : int class index
    """

    def __init__(
        self,
        root_dir: str,
        is_training: bool = True,
        img_size: int = IMG_SIZE,
    ):
        self.root_dir = root_dir
        self.is_training = is_training
        self.img_size = img_size

        # Discover samples
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        class_dirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}

        for cname in class_dirs:
            cdir = os.path.join(root_dir, cname)
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cdir, fname), self.class_to_idx[cname]))

        # Transforms
        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.2, saturation=0.15
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                CutOut(size=40),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Simulate clinical metadata  (age, MMSE)
        class_name = CLASS_NAMES[label]
        mmse_lo, mmse_hi = MMSE_RANGES[class_name]
        mmse = random.uniform(mmse_lo, mmse_hi) / 30.0       # normalize to [0, 1]
        age = random.uniform(*AGE_RANGE) / 100.0              # normalize to [0, 1]
        metadata = torch.tensor([age, mmse], dtype=torch.float32)

        return img, metadata, label


# ── Helpers ─────────────────────────────────
def compute_class_weights(dataset: AlzheimerDataset) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    labels = [s[1] for s in dataset.samples]
    counter = Counter(labels)
    total = len(labels)
    n_cls = len(counter)
    weights = torch.zeros(n_cls)
    for cls_idx, count in counter.items():
        weights[cls_idx] = total / (n_cls * count)
    return weights


def get_weighted_sampler(dataset: AlzheimerDataset) -> WeightedRandomSampler:
    """Sampler that oversamples minority classes (ModerateDemented)."""
    labels = [s[1] for s in dataset.samples]
    counter = Counter(labels)
    total = len(labels)
    class_weight = {c: total / cnt for c, cnt in counter.items()}
    sample_weights = [class_weight[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels))


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Create train/val DataLoaders + class weights."""
    train_ds = AlzheimerDataset(train_dir, is_training=True)
    val_ds = AlzheimerDataset(val_dir, is_training=False)

    print(f"  [train] {len(train_ds)} images  |  [val] {len(val_ds)} images")

    sampler = get_weighted_sampler(train_ds)
    class_weights = compute_class_weights(train_ds)
    print(f"  Class weights: {dict(zip(CLASS_NAMES, class_weights.tolist()))}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, class_weights


# ── Quick test ──────────────────────────────
if __name__ == "__main__":
    ds = AlzheimerDataset("data/data/train", is_training=True)
    img, meta, label = ds[0]
    print(f"Samples : {len(ds)}")
    print(f"Image   : {img.shape}")
    print(f"Metadata: {meta}")
    print(f"Label   : {label} ({CLASS_NAMES[label]})")
    print("[OK] Dataset works")
