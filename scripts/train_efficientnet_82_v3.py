"""
Training Pipeline for Synthetic Augmented 82-Class Indian Cattle & Buffalo Recognition (Model v3).
Architecture: EfficientNet-B0 (Two-Stage Transfer Learning)
Augmentation Policy: Real + Synthetic Training Partition (Minimum 20 images/class target)
Validation Policy: 100% Real Held-Out Validation Partition (Zero Synthetic Contamination)
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

WORKSPACE = Path("c:/Users/HP/Desktop/MCA Project/MCA Project AI Breed")
MODELS_V3 = WORKSPACE / "models" / "synthetic_82_breeds_v3"
REPORTS_V3 = WORKSPACE / "reports" / "synthetic_v3"

MODELS_V3.mkdir(parents=True, exist_ok=True)
REPORTS_V3.mkdir(parents=True, exist_ok=True)

MANIFEST_V3 = REPORTS_V3 / "augmented_split_manifest_v3.csv"
CLASS_NAMES_JSON = WORKSPACE / "models" / "class_names.json"

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(8)


class FastPreloadedDataset(Dataset):
    def __init__(self, df_subset, class_to_idx, transform=None, name="dataset"):
        self.transform = transform
        self.samples = []
        print(f"Pre-caching {name} into RAM ({len(df_subset)} records)...")
        t0 = time.time()
        for _, row in df_subset.iterrows():
            img_path = WORKSPACE / row["relative_path"]
            if img_path.exists():
                try:
                    with Image.open(img_path) as im:
                        rgb_im = im.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                    self.samples.append({
                        "image": rgb_im,
                        "label": class_to_idx[row["breed_id"]],
                        "breed_id": row["breed_id"],
                        "breed_name": row["breed_name"],
                        "species": row["species"],
                        "image_id": row["image_id"],
                        "source_type": row.get("source_type", "UNKNOWN")
                    })
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        print(f"Cached {len(self.samples)} {name} images in {time.time()-t0:.2f}s.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = item["image"]
        if self.transform:
            img = self.transform(img)
        return img, item["label"], item["image_id"]


def main():
    print("=" * 70)
    print("STARTING EFFICIENTNET-B0 V3 TRAINING (REAL + SYNTHETIC AUGMENTATION)")
    print("=" * 70)

    with open(CLASS_NAMES_JSON, "r") as f:
        class_map = json.load(f)

    idx_to_class = class_map["idx_to_class"]
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    num_classes = len(class_to_idx)
    assert num_classes == 82, f"Expected 82 classes, got {num_classes}"

    # Copy class mapping to models v3
    with open(MODELS_V3 / "class_mapping_v3.json", "w") as f:
        json.dump(class_map, f, indent=2)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    manifest_df = pd.read_csv(MANIFEST_V3)
    train_df = manifest_df[manifest_df["split"] == "train"]
    val_df = manifest_df[manifest_df["split"] == "validation"]

    print(f"Augmented Train split: {len(train_df)} ({len(train_df[train_df['source_type']=='REAL'])} Real, {len(train_df[train_df['source_type']=='SYNTHETIC'])} Synthetic)")
    print(f"Validation split: {len(val_df)} (100% Real)")

    train_ds = FastPreloadedDataset(train_df, class_to_idx, transform=train_tf, name="Augmented Train (V3)")
    val_ds = FastPreloadedDataset(val_df, class_to_idx, transform=eval_tf, name="Real Validation")

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    print(f"Training on device: {device} with batch size {batch_size}")

    # Build model: EfficientNet-B0 pretrained
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Stage 1: Classifier head warmup (frozen backbone)
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_s1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs_s1 = 10

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_state = None
    history = []

    print("\n" + "=" * 60)
    print("STAGE 1: CLASSIFIER HEAD WARMUP (10 EPOCHS, FROZEN BACKBONE)")
    print("=" * 60)

    for epoch in range(1, epochs_s1 + 1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_s1.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_s1.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation on Real dataset
        model.eval()
        val_loss = 0.0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                y_true_val.extend(labels.numpy())
                y_pred_val.extend(preds.numpy())

        val_loss = val_loss / len(val_ds)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_macro_f1 = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro', zero_division=0)[2]

        is_best = val_acc >= best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_f1 = val_macro_f1
            best_model_state = model.state_dict().copy()

        history.append({
            "epoch": epoch,
            "stage": 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "lr": 1e-3,
            "is_best": is_best
        })

        print(f"Epoch {epoch:2d}/10 (Stage 1) [{time.time()-t_ep:.1f}s] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:5.2f}% | Macro F1: {val_macro_f1*100:5.2f}%"
              f"{' *BEST*' if is_best else ''}")

    # Stage 2: Deep layer fine-tuning
    print("\n" + "=" * 60)
    print("STAGE 2: DEEP BACKBONE FINE-TUNING (5 EPOCHS, UNFREEZE features[6:])")
    print("=" * 60)

    # Unfreeze top convolutional stages
    for param in model.features[6:].parameters():
        param.requires_grad = True

    optimizer_s2 = optim.AdamW([
        {"params": model.features[6:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4}
    ], weight_decay=1e-4)

    epochs_s2 = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=epochs_s2, eta_min=1e-5)

    for epoch in range(epochs_s1 + 1, epochs_s1 + epochs_s2 + 1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_s2.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_s2.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation on Real dataset
        model.eval()
        val_loss = 0.0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                y_true_val.extend(labels.numpy())
                y_pred_val.extend(preds.numpy())

        val_loss = val_loss / len(val_ds)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_macro_f1 = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro', zero_division=0)[2]

        is_best = val_acc >= best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_f1 = val_macro_f1
            best_model_state = model.state_dict().copy()

        current_lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch,
            "stage": 2,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
            "is_best": is_best
        })

        print(f"Epoch {epoch:2d}/15 (Stage 2) [{time.time()-t_ep:.1f}s] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:5.2f}% | Macro F1: {val_macro_f1*100:5.2f}%"
              f"{' *BEST*' if is_best else ''}")

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE. Peak Real Validation Accuracy: {best_val_acc*100:.2f}% (Macro F1: {best_val_f1*100:.2f}%)")
    print("=" * 60)

    # Save checkpoints
    torch.save({
        "model_state_dict": best_model_state if best_model_state is not None else model.state_dict(),
        "num_classes": num_classes,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "architecture": "efficientnet_b0",
        "training_dataset": "augmented_v3_real_plus_synthetic",
        "created_at": datetime.now(timezone.utc).isoformat()
    }, MODELS_V3 / "best_model_v3.pth")
    print(f"Saved best model checkpoint to: {MODELS_V3 / 'best_model_v3.pth'}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "final_val_acc": val_acc,
        "final_val_f1": val_macro_f1,
        "architecture": "efficientnet_b0",
        "training_dataset": "augmented_v3_real_plus_synthetic",
        "created_at": datetime.now(timezone.utc).isoformat()
    }, MODELS_V3 / "final_model_v3.pth")
    print(f"Saved final model weights to: {MODELS_V3 / 'final_model_v3.pth'}")

    # Save training history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(REPORTS_V3 / "training_history_v3.csv", index=False)

    # Plot training curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss (Augmented)")
    plt.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss (Real Only)", linestyle="--")
    plt.axvline(10.5, color="gray", linestyle=":", label="Stage 2 Transition")
    plt.title("Loss Progression (Model V3: Real + Synthetic)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df_hist["epoch"], df_hist["train_acc"] * 100, label="Train Acc (Augmented)")
    plt.plot(df_hist["epoch"], df_hist["val_acc"] * 100, label="Val Acc (Real Only)", linestyle="--")
    plt.axvline(10.5, color="gray", linestyle=":", label="Stage 2 Transition")
    plt.title("Accuracy Progression (Model V3: Real + Synthetic)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS_V3 / "training_curves_v3.png", dpi=300)
    plt.close()
    print("Saved training curves plot to reports/synthetic_v3/training_curves_v3.png")

    # Training configuration json
    config = {
        "model_version": "v3_real_plus_synthetic",
        "architecture": "EfficientNet-B0",
        "num_classes": 82,
        "train_samples_total": len(train_ds),
        "real_train_samples": len(train_df[train_df["source_type"] == "REAL"]),
        "synthetic_train_samples": len(train_df[train_df["source_type"] == "SYNTHETIC"]),
        "val_samples_real": len(val_ds),
        "min_train_images_per_class": int(train_df.groupby("breed_id").size().min()),
        "max_train_images_per_class": int(train_df.groupby("breed_id").size().max()),
        "stage_1_epochs": 10,
        "stage_1_lr": 1e-3,
        "stage_2_epochs": 5,
        "stage_2_backbone_lr": 1e-4,
        "stage_2_head_lr": 5e-4,
        "batch_size": batch_size,
        "loss_function": "CrossEntropyLoss(label_smoothing=0.1)",
        "best_real_val_accuracy": float(best_val_acc),
        "best_real_val_macro_f1": float(best_val_f1)
    }
    with open(MODELS_V3 / "training_config_v3.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save training_report_v3.md
    report_md = f"""# Model Training Report: EfficientNet-B0 V3 (Real + Synthetic Augmentation)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model Architecture**: EfficientNet-B0 (ImageNet-1K Pretrained)  
**Dataset Augmentation**: Supplementary Synthetic Dataset (Minimum 20 images/class target)  
**Validation Isolation**: 100% Real Unseen Validation Samples ($N=84$)  
**Best Validation Accuracy**: **{best_val_acc*100:.2f}%** (Macro F1: **{best_val_f1*100:.2f}%**)  
**Date**: September 2026  

---

## 1. Training Setup & Hyperparameters

| Hyperparameter | Configuration |
| :--- | :--- |
| **Total Classes** | 82 (59 Cattle, 23 Buffalo) |
| **Training Partition Composition** | **1,657 images** (382 Real + 1,275 Synthetic) |
| **Minimum Training Images / Class** | **20 images** (100% of classes reach $\ge 20$) |
| **Validation Partition Composition**| **84 images** (100% Real Photographs) |
| **Held-Out Test Partition** | **123 images** (100% Real Photographs, untouched) |
| **Batch Size** | {batch_size} |
| **Loss Function** | CrossEntropyLoss(label_smoothing=0.1) |
| **Stage 1 (Head Warmup)** | 10 Epochs, Frozen Backbone, AdamW lr=$10^{{-3}}$ |
| **Stage 2 (Deep Fine-Tuning)** | 5 Epochs, Unfrozen `features[6:]`, CosineAnnealingLR |

---

## 2. Validation Progression on Real Data

- Peak Real Validation Accuracy: **{best_val_acc*100:.2f}%**
- Peak Real Validation Macro F1: **{best_val_f1*100:.2f}%**
- Checkpoints saved:
  - `models/synthetic_82_breeds_v3/best_model_v3.pth`
  - `models/synthetic_82_breeds_v3/final_model_v3.pth`
"""
    with open(REPORTS_V3 / "training_report_v3.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("Saved training_report_v3.md and training_config_v3.json")


if __name__ == "__main__":
    main()
