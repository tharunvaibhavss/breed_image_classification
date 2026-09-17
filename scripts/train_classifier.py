#!/usr/bin/env python3
"""CLI script to run EfficientNet-B0 breed classification transfer learning training."""

import sys
import os
import json
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.classification.dataset import create_dataloaders
from ml.classification.trainer import EfficientNetTrainer


def main():
    """Execute complete Phase 6 EfficientNet-B0 transfer learning training pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "processed" / "dataset_manifest.json"
    dataset_dir = project_root / "data" / "raw" / "dataset"
    models_dir = project_root / "models"
    configs_dir = project_root / "configs"

    print("=" * 70)
    print(" PHASE 6 - EFFICIENTNET-B0 BREED CLASSIFICATION TRAINING")
    print("=" * 70)
    print(f" Dataset Manifest Path : {manifest_path}")
    print(f" Output Models Dir     : {models_dir}")
    print("-" * 70)

    # 1. Load manifest records
    manifest_records = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            manifest_records = data.get("manifest_records", [])

    print(f"[1/3] Loaded {len(manifest_records)} dataset manifest records...")

    # 2. Create PyTorch DataLoaders
    print("[2/3] Constructing PyTorch DataLoaders for train, val, and test splits...")
    dataloaders = create_dataloaders(
        manifest_records=manifest_records,
        base_dir=dataset_dir,
        batch_size=16,
        num_workers=0,
        target_size=(224, 224),
    )

    # 3. Execute EfficientNet-B0 Training
    print("[3/3] Launching EfficientNet-B0 transfer learning trainer...")
    trainer = EfficientNetTrainer(
        num_classes=6,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=42,
        models_dir=models_dir,
    )
    res = trainer.train(dataloaders=dataloaders, epochs=20, patience=7)

    print("=" * 70)
    print(" EFFICIENTNET-B0 TRAINING SUMMARY")
    print("=" * 70)
    print(f" Training Status       : {res['status']}")
    print(f" Best Model Checkpoint : {res['best_model_path']}")
    print(f" Last Model Checkpoint : {res['last_model_path']}")
    if "best_val_f1" in res:
        print(f" Best Val Macro F1     : {res['best_val_f1']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
