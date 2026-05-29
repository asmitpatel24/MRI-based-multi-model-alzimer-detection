"""
=============================================================
  train.py — Training Pipeline for Hybrid CNN + ViT + MLP
  ─────────────────────────────────────────────────────────
  Phase 1: Freeze CNN + ViT backbones, train fusion + MLP
  Phase 2: Unfreeze top 30%, fine-tune end-to-end
  
  Usage:
    python train.py                     # full training
    python train.py --epochs 1          # smoke test
    python train.py --batch-size 8      # smaller batch
=============================================================
"""

import os
import sys
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)
import seaborn as sns
from tqdm import tqdm

from hybrid_model import AlzheimerHybridModel
from dataset import create_dataloaders, CLASS_NAMES


# ── Cosine Annealing with Warmup ───────────
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ── Training one epoch ─────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, _, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if device.type == "cuda":
            # using AMP for faster and memory efficient training
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision for CPU
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        lr = scheduler.step()
        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return running_loss / total, correct / total


# ── Validation ─────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, _, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted")

    return running_loss / total, acc, bal_acc, w_f1, all_preds, all_labels


# ── Full evaluation with report ────────────
def full_evaluation(model, loader, device):
    _, acc, bal_acc, w_f1, preds, labels = validate(
        model, loader, nn.CrossEntropyLoss(), device
    )
    print(f"\n  Accuracy          : {acc*100:.2f}%")
    print(f"  Balanced Accuracy : {bal_acc*100:.2f}%")
    print(f"  Weighted F1       : {w_f1*100:.2f}%")
    print("\n" + classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4
    ))

    # Confusion matrix with attractive dark tech theme
    cm = confusion_matrix(labels, preds)
    
    plt.style.use('dark_background')
    sns.set_theme(style="ticks", rc={"axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117", "text.color": "#e6edf3"})

    fig, ax = plt.subplots(figsize=(11, 8))
    heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 16, "weight": "bold"},
                linewidths=3, linecolor='#0d1117', ax=ax,
                cbar_kws={'shrink': .85, 'label': 'Number of Samples'})
                
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Number of Samples', size=14, color='#8b949e', weight='bold')
    cbar.ax.tick_params(labelsize=12, colors='#c9d1d9')

    plt.xlabel("Predicted Mode", fontsize=16, fontweight='bold', color='#a2aabc', labelpad=15)
    plt.ylabel("True Disease Stage", fontsize=16, fontweight='bold', color='#a2aabc', labelpad=15)
    plt.title("Confusion Matrix — Hybrid Architecture", fontsize=20, fontweight='bold', pad=25, color='#ffffff')
    plt.xticks(rotation=30, ha='right', fontsize=13, color='#c9d1d9')
    plt.yticks(rotation=0, fontsize=13, color='#c9d1d9')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight', facecolor='#0d1117', pad_inches=0.3)
    plt.close()
    
    # Reset seaborn theme back to default for other plots
    sns.reset_orig()
    print("  -> aesthetic confusion_matrix.png saved")

    return {"accuracy": acc, "balanced_accuracy": bal_acc, "weighted_f1": w_f1}


# ── Plot training curves ──────────────────
def plot_history(history):
    metrics = ["loss", "accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m in zip(axes, metrics):
        ax.plot(history[f"train_{m}"], label="train")
        ax.plot(history[f"val_{m}"], label="val")
        ax.set_title(m.upper())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()
    print("  -> training_history.png saved")


# ── Main training pipeline ─────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Alzheimer Hybrid Model")
    parser.add_argument("--train-dir", default="data/data/train")
    parser.add_argument("--val-dir", default="data/data/val")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total epochs for Phase 1 (head training)")
    parser.add_argument("--ft-epochs", type=int, default=20,
                        help="Total epochs for Phase 2 (fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ft-lr", type=float, default=1e-5)
    parser.add_argument("--save-path", default="alzheimer_hybrid_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True # Auto-tuner for fastest convolution algorithms
    print("=" * 60)
    print("  Alzheimer Hybrid CNN + ViT + MLP  Trainer")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Phase 1    : {args.epochs} epochs  (lr={args.lr})")
    print(f"  Phase 2    : {args.ft_epochs} epochs  (lr={args.ft_lr})")
    print()

    # ── Data ──
    # Optimal workers limit on windows usually shouldn't exceed CPU cores
    train_loader, val_loader, class_weights = create_dataloaders(
        args.train_dir, args.val_dir, args.batch_size, num_workers=min(4, os.cpu_count() or 1),
    )

    # ── Model ──
    model = AlzheimerHybridModel(num_classes=4)
    model.to(device)
    total, trainable = model.count_params()
    print(f"  Model params: {total:,} total")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))
    history = defaultdict(list)
    best_val_acc = 0.0

    # ════════════════════════════════════════
    # PHASE 1 — Train fusion head + MLP only
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 1 — Train Head (backbones frozen)")
    print("=" * 60)

    model.freeze_backbones()
    _, trainable = model.count_params()
    print(f"  Trainable: {trainable:,} params\n")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    steps_per_epoch = len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=steps_per_epoch * 2,
        total_steps=steps_per_epoch * args.epochs,
        max_lr=args.lr,
    )

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        v_loss, v_acc, v_bal, v_f1, _, _ = validate(
            model, val_loader, criterion, device
        )
        history["train_loss"].append(t_loss)
        history["train_accuracy"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(v_acc)

        print(f"  Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={t_loss:.4f}  train_acc={t_acc:.3f}  "
              f"val_acc={v_acc:.3f}  bal_acc={v_bal:.3f}  f1={v_f1:.3f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"    -> Saved best model ({v_acc*100:.1f}%)")

    # ════════════════════════════════════════
    # PHASE 2 — Fine-tune with unfrozen top layers
    # ════════════════════════════════════════
    if args.ft_epochs > 0:
        print("\n" + "=" * 60)
        print("  PHASE 2 — Fine-tune (top 30% unfrozen)")
        print("=" * 60)

        model.load_state_dict(torch.load(args.save_path, weights_only=True))
        model.unfreeze_backbones(unfreeze_fraction=0.3)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.ft_lr, weight_decay=1e-4,
        )
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=steps_per_epoch * 2,
            total_steps=steps_per_epoch * args.ft_epochs,
            max_lr=args.ft_lr,
        )

        for epoch in range(1, args.ft_epochs + 1):
            t_loss, t_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, scaler
            )
            v_loss, v_acc, v_bal, v_f1, _, _ = validate(
                model, val_loader, criterion, device
            )
            history["train_loss"].append(t_loss)
            history["train_accuracy"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_accuracy"].append(v_acc)

            print(f"  Epoch {epoch:02d}/{args.ft_epochs}  "
                  f"train_loss={t_loss:.4f}  train_acc={t_acc:.3f}  "
                  f"val_acc={v_acc:.3f}  bal_acc={v_bal:.3f}  f1={v_f1:.3f}")

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(model.state_dict(), args.save_path)
                print(f"    -> Saved best model ({v_acc*100:.1f}%)")

    # ── Final evaluation ──
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60)
    model.load_state_dict(torch.load(args.save_path, weights_only=True))
    results = full_evaluation(model, val_loader, device)

    plot_history(history)

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Model saved to    : {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
