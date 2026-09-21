"""
Training Pipeline for Expanded 82-Class Indian Cattle & Buffalo Recognition (Model v2).
Architecture: EfficientNet-B0 (Two-Stage Transfer Learning)
Outputs:
- models/expanded_82_breeds_v2/best_model_v2.pth
- models/expanded_82_breeds_v2/final_model_v2.pth
- models/expanded_82_breeds_v2/class_mapping_v2.json
- models/expanded_82_breeds_v2/training_config_v2.json
- reports/dataset_expansion_v2/training_history_v2.csv
- reports/dataset_expansion_v2/training_report_v2.md
- reports/dataset_expansion_v2/training_curves_v2.png
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

WORKSPACE = Path(__file__).resolve().parent.parent
DATASET_V2 = WORKSPACE / "dataset" / "expanded_82_breeds_v2"
SPLITS_V2 = DATASET_V2 / "splits"
MODELS_V2 = WORKSPACE / "models" / "expanded_82_breeds_v2"
REPORTS_V2 = WORKSPACE / "reports" / "dataset_expansion_v2"

MODELS_V2.mkdir(parents=True, exist_ok=True)
REPORTS_V2.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLITS_V2 / "train.csv"
VAL_CSV = SPLITS_V2 / "validation.csv"
TEST_CSV = SPLITS_V2 / "test.csv"
CLASS_NAMES_JSON = WORKSPACE / "models" / "class_names.json"

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(4)


class FastPreloadedDataset(Dataset):
    def __init__(self, csv_file, class_to_idx, transform=None):
        self.transform = transform
        self.samples = []
        df = pd.read_csv(csv_file)
        print(f"Pre-caching {csv_file.name} into RAM ({len(df)} records)...")
        t0 = time.time()
        for _, row in df.iterrows():
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
                        "image_id": row["image_id"]
                    })
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        print(f"Cached {len(self.samples)} images in {time.time()-t0:.2f}s.")

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
    print("STARTING EFFICIENTNET-B0 TRAINING PIPELINE (DATASET v2, 82 CLASSES)")
    print("=" * 70)

    with open(CLASS_NAMES_JSON, "r") as f:
        class_map = json.load(f)

    idx_to_class = class_map["idx_to_class"]
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    num_classes = len(class_to_idx)
    assert num_classes == 82, f"Expected 82 classes, got {num_classes}"

    # Copy class mapping to models v2
    with open(MODELS_V2 / "class_mapping_v2.json", "w") as f:
        json.dump(class_map, f, indent=2)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = FastPreloadedDataset(TRAIN_CSV, class_to_idx, transform=train_tf)
    val_ds = FastPreloadedDataset(VAL_CSV, class_to_idx, transform=eval_tf)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"\nDataLoader initialized: {len(train_ds)} train, {len(val_ds)} val, batch size {batch_size}")

    device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Build model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # Class-aware loss: Label smoothing to prevent overconfident collapse on long-tail classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Stage 1: Freeze backbone
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

        # Validation
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

        print(f"Epoch {epoch:2d}/10 (Stage 1) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% Macro-F1: {val_macro_f1*100:5.2f}% {'[BEST]' if is_best else ''}")

    # Stage 2: Deep layer unfreezing (features[6:])
    print("\n" + "=" * 60)
    print("STAGE 2: DEEP LAYER FINE-TUNING (5 EPOCHS, COSINE DECAY)")
    print("=" * 60)

    for param in model.features[6:].parameters():
        param.requires_grad = True

    optimizer_s2 = optim.AdamW([
        {'params': model.features[6:].parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-4)

    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=5, eta_min=1e-6)
    epochs_s2 = 5

    for epoch in range(epochs_s1 + 1, epochs_s1 + epochs_s2 + 1):
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

        scheduler_s2.step()
        train_loss = running_loss / total
        train_acc = correct / total

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

        current_lr = scheduler_s2.get_last_lr()[0]
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

        print(f"Epoch {epoch:2d}/15 (Stage 2) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% Macro-F1: {val_macro_f1*100:5.2f}% {'[BEST]' if is_best else ''}")

    # Save checkpoints
    final_model_state = model.state_dict().copy()
    torch.save(best_model_state, MODELS_V2 / "best_model_v2.pth")
    torch.save(final_model_state, MODELS_V2 / "final_model_v2.pth")
    print(f"\nSaved best model to {MODELS_V2 / 'best_model_v2.pth'} (Best Val Acc: {best_val_acc*100:.2f}%)")
    print(f"Saved final model to {MODELS_V2 / 'final_model_v2.pth'}")

    # Save history CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(REPORTS_V2 / "training_history_v2.csv", index=False)
    print("Saved reports/dataset_expansion_v2/training_history_v2.csv")

    # Save configuration JSON
    config_v2 = {
        "model_name": "EfficientNet-B0 (Expanded 82-Breeds v2)",
        "backbone": "efficientnet_b0",
        "weights": "DEFAULT (ImageNet-1K)",
        "num_classes": 82,
        "input_resolution": [224, 224, 3],
        "training_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "batch_size": batch_size,
        "stage_1_epochs": epochs_s1,
        "stage_2_epochs": epochs_s2,
        "loss_function": "CrossEntropyLoss(label_smoothing=0.1)",
        "best_validation_accuracy": float(best_val_acc),
        "best_validation_macro_f1": float(best_val_f1),
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }
    with open(MODELS_V2 / "training_config_v2.json", "w") as f:
        json.dump(config_v2, f, indent=2)
    print("Saved models/expanded_82_breeds_v2/training_config_v2.json")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    plt.axvline(x=10.5, color="grey", linestyle="--", label="Stage 2 Transition")
    plt.title("Training & Validation Loss (v2)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df_hist["epoch"], df_hist["train_acc"] * 100, label="Train Accuracy", color="#2ca02c", lw=2)
    plt.plot(df_hist["epoch"], df_hist["val_acc"] * 100, label="Val Accuracy", color="#d62728", lw=2)
    plt.axvline(x=10.5, color="grey", linestyle="--", label="Stage 2 Transition")
    plt.title("Training & Validation Accuracy (v2)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS_V2 / "training_curves_v2.png", dpi=300)
    plt.close()
    print("Saved reports/dataset_expansion_v2/training_curves_v2.png")

    # Save training report markdown
    training_report_md = f"""# Model Training Report: EfficientNet-B0 v2 (82 Breeds)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model Checkpoint**: `models/expanded_82_breeds_v2/best_model_v2.pth`  
**Training Date**: September 2026  
**Status**: Completed  

---

## 1. Training Configuration & Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Base Architecture** | EfficientNet-B0 (Torchvision ImageNet Pretrained) |
| **Input Shape** | 224 x 224 x 3 (RGB) |
| **Classifier Head** | Dropout(0.2) + Linear(1280, 82) |
| **Loss Function** | CrossEntropyLoss(label_smoothing=0.1) |
| **Stage 1 (Warmup)** | 10 Epochs, Frozen Backbone, AdamW lr=$10^{{-3}}$, weight_decay=$10^{{-4}}$ |
| **Stage 2 (Fine-Tuning)** | 5 Epochs, Unfrozen `features[6:]`, AdamW (backbone lr=$10^{{-4}}$, head lr=$5\\times 10^{{-4}}$), CosineAnnealingLR |
| **Batch Size** | 16 |
| **Training Samples** | {len(train_ds)} |
| **Validation Samples** | {len(val_ds)} |

---

## 2. Validation Progression

| Milestone | Accuracy | Macro F1 | Epoch |
| :--- | :---: | :---: | :---: |
| **Stage 1 Best** | {max(r['val_acc'] for r in history[:10])*100:.2f}% | {max(r['val_macro_f1'] for r in history[:10])*100:.2f}% | {np.argmax([r['val_acc'] for r in history[:10]]) + 1} |
| **Stage 2 Final** | {history[-1]['val_acc']*100:.2f}% | {history[-1]['val_macro_f1']*100:.2f}% | 15 |
| **Overall Best Validation** | **{best_val_acc*100:.2f}%** | **{best_val_f1*100:.2f}%** | - |

---

## 3. Training Artifacts
- Checkpoint: `models/expanded_82_breeds_v2/best_model_v2.pth`
- Final Weights: `models/expanded_82_breeds_v2/final_model_v2.pth`
- Training Curves: `reports/dataset_expansion_v2/training_curves_v2.png`
- History CSV: `reports/dataset_expansion_v2/training_history_v2.csv`
"""
    with open(REPORTS_V2 / "training_report_v2.md", "w", encoding="utf-8") as f:
        f.write(training_report_md)
    print("Saved reports/dataset_expansion_v2/training_report_v2.md")
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
