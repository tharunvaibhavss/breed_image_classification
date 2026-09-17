# Deep Learning Model Training Guide

This guide covers the dataset validation, YOLO animal detector annotation verification, Albumentations data augmentation, EfficientNet-B0 breed classification transfer learning, and MLflow experiment tracking workflows.

---

## 1. Dataset Inspection & Split Generation

Before starting model training, inspect the raw dataset structure and generate group-aware 70/15/15 train/val/test splits to prevent data leakage:

```bash
# Run dataset inspection and hash deduplication
python scripts/inspect_dataset.py

# Build dataset manifest and split directories
python -m ml.preprocessing.manifest_manager
```

Processed manifest and reports will be saved at:
- `data/processed/dataset_report.json`
- `data/processed/dataset_manifest.json`
- `docs/dataset_visualization.png`

---

## 2. Albumentations Data Augmentation Pipeline

Training augmentation pipeline is defined in [`ml/preprocessing/albumentations_pipeline.py`](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/ml/preprocessing/albumentations_pipeline.py). It includes:

- Horizontal flip ($p=0.5$)
- Small rotation ($\pm 10^\circ$)
- Brightness & contrast variations ($\pm 15\%$)
- Scale and translation shifts ($\pm 5\%$)
- Mild Gaussian noise ($\sigma \le 10.0$)
- Deterministic resize ($224 \times 224$) for validation and testing.

To generate visualization sample comparisons:
```bash
python -m ml.preprocessing.augmentation_visualizer
```

---

## 3. YOLO Animal Detector Training

The YOLO animal detector identifies cattle (`0`) and buffalo (`1`) bounding box ROIs without classifying breeds.

### Step 1: Verify Label Annotations
```bash
python scripts/train_yolo.py --verify-only
```
If annotations are incomplete, the verifier stops training safely and reports missing label files.

### Step 2: Launch YOLO Training
```bash
python scripts/train_yolo.py --epochs 50 --batch 16 --img-size 640
```
Trained checkpoints and metrics will be saved at `runs/detect/train/weights/best.pt`.

---

## 4. EfficientNet-B0 Breed Classification Training

The breed classifier uses PyTorch transfer learning on pretrained `torchvision.models.efficientnet_b0`.

```bash
# Launch EfficientNet-B0 training script
python scripts/train_classifier.py --epochs 25 --batch-size 32 --lr 0.0003
```

### Key Training Hyperparameters:
- **Optimizer**: `AdamW` (weight decay $= 1\times 10^{-4}$)
- **Learning Rate Scheduler**: `CosineAnnealingLR` ($T_{\max}=25, \eta_{\min}=1\times 10^{-6}$)
- **Early Stopping**: Patience $= 7$ epochs based on validation loss
- **Reproducibility**: Seed $= 42$ (`torch.manual_seed(42)`)

---

## 5. MLflow Experiment Management

Training metrics and checkpoints are logged to MLflow under standardized experiment names:

```bash
# Launch MLflow tracking UI server
mlflow ui --port 5000
```

Experiment Names:
- YOLO Detection: `indian-cattle-buffalo/yolo-animal-detection`
- Breed Classification: `indian-cattle-buffalo/efficientnet-breed-classification`
