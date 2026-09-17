"""
Optimized High-Performance Training Pipeline for 82-Class Indian Cattle and Buffalo Breeds
Model: EfficientNet-B0 (Transfer Learning)
Optimization: In-Memory Image Pre-caching + Two-Stage Fine-Tuning
Checkpoints:
  - models/efficientnet_b0_82_breeds_best.pth
  - models/efficientnet_b0_82_breeds_final.pth
Class mapping:
  - models/class_names.json
Reports:
  - reports/training_history.csv
  - reports/training_configuration.json
  - reports/training_report.md
"""

import os
import sys
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# Path Configuration
WORKSPACE = Path(__file__).resolve().parent.parent
DATASET_ROOT = WORKSPACE / "dataset"
SPLITS_DIR = DATASET_ROOT / "splits"
MODELS_DIR = WORKSPACE / "models"
REPORTS_DIR = WORKSPACE / "reports"

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "validation.csv"
TEST_CSV = SPLITS_DIR / "test.csv"
CLASS_MAPPING_PATH = MODELS_DIR / "class_names.json"
BEST_MODEL_PATH = MODELS_DIR / "efficientnet_b0_82_breeds_best.pth"
FINAL_MODEL_PATH = MODELS_DIR / "efficientnet_b0_82_breeds_final.pth"
HISTORY_CSV_PATH = REPORTS_DIR / "training_history.csv"
CONFIG_JSON_PATH = REPORTS_DIR / "training_configuration.json"
TRAINING_REPORT_PATH = REPORTS_DIR / "training_report.md"

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.set_num_threads(4)


class CachedBreedDataset(Dataset):
    """Caches pre-resized PIL images in RAM for maximum I/O throughput."""
    def __init__(self, csv_file, dataset_root, class_to_idx, transform=None):
        self.dataset_root = Path(dataset_root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        print(f"Pre-caching dataset from {Path(csv_file).name} into RAM...")
        t0 = time.time()
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.dataset_root / row["relative_path"]
                if img_path.exists():
                    try:
                        with Image.open(img_path) as im:
                            # Pre-resize to 256x256 to eliminate disk bottlenecks and huge RAM usage
                            resized_im = im.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                        self.samples.append({
                            "image_id": row["image_id"],
                            "image": resized_im,
                            "breed_id": row["breed_id"],
                            "breed_name": row["breed_name"],
                            "species": row["species"],
                            "label": self.class_to_idx[row["breed_id"]]
                        })
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        print(f"Loaded {len(self.samples)} images in {time.time()-t0:.2f}s.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = item["image"]
        if self.transform:
            img = self.transform(img)
        return img, item["label"], item["image_id"]


def build_class_mapping():
    breeds = set()
    breed_info = {}
    
    for csv_file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not csv_file.exists():
            continue
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                b_id = row["breed_id"]
                breeds.add(b_id)
                if b_id not in breed_info:
                    breed_info[b_id] = {
                        "breed_name": row["breed_name"],
                        "species": row["species"]
                    }

    sorted_breeds = sorted(list(breeds))
    class_to_idx = {b_id: idx for idx, b_id in enumerate(sorted_breeds)}
    idx_to_class = {idx: b_id for idx, b_id in enumerate(sorted_breeds)}
    idx_to_breed_name = {idx: breed_info[b_id]["breed_name"] for idx, b_id in enumerate(sorted_breeds)}
    idx_to_species = {idx: breed_info[b_id]["species"] for idx, b_id in enumerate(sorted_breeds)}

    mapping_payload = {
        "num_classes": len(sorted_breeds),
        "classes": sorted_breeds,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "idx_to_breed_name": idx_to_breed_name,
        "idx_to_species": idx_to_species
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_PATH, mode="w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2)

    print(f"Saved class mapping with {len(sorted_breeds)} classes to {CLASS_MAPPING_PATH}")
    return class_to_idx, mapping_payload


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def extract_features(model_backbone, dataloader, device):
    """Pre-extracts 1280-dim feature vectors when backbone is frozen."""
    model_backbone.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            # Pass through convolutional features + avgpool + flatten
            feats = model_backbone.features(images)
            feats = model_backbone.avgpool(feats)
            feats = torch.flatten(feats, 1)
            all_features.append(feats.cpu())
            all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def run_training():
    print("=" * 70)
    print("AI-POWERED BREED RECOGNITION: OPTIMIZED EFFICIENTNET-B0 82-CLASS TRAINING")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"Compute Device: {device} (Thread pool: {torch.get_num_threads()})")

    class_to_idx, mapping_payload = build_class_mapping()
    num_classes = len(class_to_idx)
    print(f"Total Target Classes: {num_classes}")

    train_tf, eval_tf = get_transforms()

    train_ds = CachedBreedDataset(TRAIN_CSV, DATASET_ROOT, class_to_idx, transform=train_tf)
    val_ds = CachedBreedDataset(VAL_CSV, DATASET_ROOT, class_to_idx, transform=eval_tf)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize EfficientNet-B0 with ImageNet weights
    print("\nLoading pretrained EfficientNet-B0 backbone...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    # Loss function with label smoothing for regularization on fine-grained classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    epochs_stage1 = 10
    epochs_stage2 = 5
    history = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    # -------------------------------------------------------------
    # STAGE 1: Fast Feature-Extraction Training of Head
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"STAGE 1: Training Classification Head ({epochs_stage1} Epochs, Frozen Backbone)")
    print("=" * 50)

    print("Extracting feature representations using frozen backbone...")
    t_feat = time.time()
    # Create deterministic loader for feature extraction
    train_eval_ds = CachedBreedDataset(TRAIN_CSV, DATASET_ROOT, class_to_idx, transform=eval_tf)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_feats, train_y = extract_features(model, train_eval_loader, device)
    val_feats, val_y = extract_features(model, val_loader, device)
    print(f"Features extracted in {time.time()-t_feat:.2f}s (Train: {train_feats.shape}, Val: {val_feats.shape})")

    # Classification head
    classifier_head = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    ).to(device)

    optimizer_stage1 = torch.optim.AdamW(classifier_head.parameters(), lr=1e-3, weight_decay=1e-2)

    # Dataset of pre-extracted features
    from torch.utils.data import TensorDataset
    feat_train_ds = TensorDataset(train_feats, train_y)
    feat_train_loader = DataLoader(feat_train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs_stage1 + 1):
        t0 = time.time()
        classifier_head.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in feat_train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer_stage1.zero_grad()
            outputs = classifier_head(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage1.step()

            running_loss += loss.item() * feats.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        tr_loss = running_loss / total
        tr_acc = correct / total

        # Validation
        classifier_head.eval()
        with torch.no_grad():
            v_outputs = classifier_head(val_feats.to(device))
            v_loss = criterion(v_outputs, val_y.to(device)).item()
            _, v_preds = torch.max(v_outputs, 1)
            val_acc = (v_preds == val_y.to(device)).sum().item() / val_y.size(0)
            val_loss = v_loss

        elapsed = time.time() - t0
        current_lr = optimizer_stage1.param_groups[0]["lr"]

        print(f"Stage 1 Epoch [{epoch:02d}/{epochs_stage1:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | "
              f"Time: {elapsed:4.2f}s")

        history.append({
            "epoch": epoch,
            "stage": 1,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "learning_rate": current_lr
        })

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            # Assemble full model
            model.classifier = classifier_head
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": num_classes,
                "stage": 1
            }, BEST_MODEL_PATH)
            print(f"  --> Checkpoint saved (New Best Val Acc: {best_val_acc*100:.2f}%)")

    # Re-attach trained classifier head to model
    model.classifier = classifier_head

    # -------------------------------------------------------------
    # STAGE 2: End-to-End Fine-Tuning
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"STAGE 2: End-to-End Network Fine-Tuning ({epochs_stage2} Epochs, Cosine Annealing)")
    print("=" * 50)

    for param in model.parameters():
        param.requires_grad = True

    optimizer_stage2 = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=epochs_stage2, eta_min=1e-6)

    def full_train_epoch():
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer_stage2.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage2.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return running_loss / total, correct / total

    @torch.no_grad()
    def full_eval_epoch():
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return running_loss / total, correct / total

    for s2_epoch in range(1, epochs_stage2 + 1):
        epoch = epochs_stage1 + s2_epoch
        t0 = time.time()
        tr_loss, tr_acc = full_train_epoch()
        val_loss, val_acc = full_eval_epoch()
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = optimizer_stage2.param_groups[0]["lr"]
        print(f"Stage 2 Epoch [{s2_epoch:02d}/{epochs_stage2:02d}] (Total {epoch:02d}) "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | "
              f"Time: {elapsed:4.1f}s")

        history.append({
            "epoch": epoch,
            "stage": 2,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "learning_rate": current_lr
        })

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": num_classes,
                "stage": 2
            }, BEST_MODEL_PATH)
            print(f"  --> Checkpoint saved (New Best Val Acc: {best_val_acc*100:.2f}%)")

    # Save final model state
    torch.save({
        "epoch": epochs_stage1 + epochs_stage2,
        "model_state_dict": model.state_dict(),
        "val_acc": history[-1]["val_acc"],
        "val_loss": history[-1]["val_loss"],
        "num_classes": num_classes,
        "stage": 2
    }, FINAL_MODEL_PATH)

    print(f"\nFinal model saved to: {FINAL_MODEL_PATH}")
    print(f"Best model saved to:  {BEST_MODEL_PATH} (Epoch {best_epoch}, Best Val Acc: {best_val_acc*100:.2f}%)")

    # Save training history CSV
    with open(HISTORY_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to: {HISTORY_CSV_PATH}")

    # Save training configuration JSON
    config = {
        "architecture": "EfficientNet-B0",
        "num_classes": num_classes,
        "image_size": 224,
        "batch_size": batch_size,
        "random_seed": RANDOM_SEED,
        "epochs_stage1": epochs_stage1,
        "epochs_stage2": epochs_stage2,
        "total_epochs": epochs_stage1 + epochs_stage2,
        "optimizer_stage1": "AdamW (lr=1e-3, weight_decay=1e-2)",
        "optimizer_stage2": "AdamW (features_lr=1e-4, classifier_lr=5e-4, weight_decay=1e-2)",
        "scheduler": "CosineAnnealingLR (T_max=5, eta_min=1e-6)",
        "loss_function": "CrossEntropyLoss(label_smoothing=0.05)",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    with open(CONFIG_JSON_PATH, mode="w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to: {CONFIG_JSON_PATH}")

    # Write training report markdown
    report_md = f"""# EfficientNet-B0 82-Class Training Report

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model Architecture**: EfficientNet-B0 Transfer Learning  
**Target Classes**: 82 (59 Indian Cattle Breeds + 23 Indian Buffalo Breeds)  
**Training Date**: {config['timestamp_utc']}  

---

## 1. Executive Summary

| Parameter | Value |
|---|---|
| **Backbone** | EfficientNet-B0 (Pretrained on ImageNet-1K) |
| **Input Resolution** | 224 × 224 RGB |
| **Total Classes** | 82 |
| **Training Samples** | {len(train_ds)} |
| **Validation Samples** | {len(val_ds)} |
| **Batch Size** | {batch_size} |
| **Total Epochs** | {epochs_stage1 + epochs_stage2} (Stage 1: {epochs_stage1}, Stage 2: {epochs_stage2}) |
| **Best Epoch** | Epoch {best_epoch} |
| **Best Validation Accuracy** | **{best_val_acc*100:.2f}%** |
| **Best Validation Loss** | **{best_val_loss:.4f}** |
| **Final Checkpoint Path** | `{FINAL_MODEL_PATH.name}` |
| **Best Checkpoint Path** | `{BEST_MODEL_PATH.name}` |

---

## 2. Two-Stage Training Protocol

1. **Stage 1 (Feature Extraction, Epochs 1–{epochs_stage1})**:
   - Frozen convolutional features backbone.
   - Classification head (`Linear(1280, 82)`) trained with AdamW ($lr = 10^{{-3}}$, weight decay = $10^{{-2}}$).
   - Fast convergence of top-layer decision boundary without destroying pretrained feature representations.

2. **Stage 2 (Full Fine-Tuning, Epochs {epochs_stage1 + 1}–{epochs_stage1 + epochs_stage2})**:
   - Complete network unlocked for gradient backpropagation.
   - Differential learning rate ($10^{{-4}}$ on convolutional backbone, $5 \\times 10^{{-4}}$ on classifier head).
   - Cosine Annealing schedule decaying learning rate smoothly toward $10^{{-6}}$.
   - Label smoothing regularizer ($\alpha = 0.05$) to mitigate overconfidence across 82 fine-grained classes.

---

## 3. Training & Validation Progression

| Epoch | Stage | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|---|---|---|---|---|---|---|
"""
    for h in history:
        report_md += f"| {h['epoch']} | {h['stage']} | {h['train_loss']:.4f} | {h['train_acc']*100:.1f}% | {h['val_loss']:.4f} | {h['val_acc']*100:.1f}% | {h['learning_rate']:.2e} |\n"

    report_md += f"""
---
*Model weights saved to `{BEST_MODEL_PATH.name}` are ready for unbiased unseen test evaluation.*
"""

    with open(TRAINING_REPORT_PATH, mode="w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Training report saved to: {TRAINING_REPORT_PATH}")


if __name__ == "__main__":
    run_training()
