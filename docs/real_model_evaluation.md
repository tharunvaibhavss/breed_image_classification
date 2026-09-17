# Real Model Evaluation Report

## 1. Evaluation Methodology & Constraints

This report documents the real-data evaluation of the trained **EfficientNet-B0** classifier (`models/efficientnet_best.pth`) and **YOLO** detector on real dataset images.

### Strict Audit Guidelines Applied:
- **No Fabricated Results**: Metrics are derived strictly from empirical evaluation runs.
- **No Synthetic Data**: Mock/synthetic image fixtures used during unit testing are excluded from real-data performance tables.
- **Unseen Test Set Only**: Training and validation split images are strictly excluded.

---

## 2. Dataset Status & Evaluation Execution

- **Target Dataset Path**: `data/raw/dataset/`
- **Real Images Found**: `0`
- **Real Test Set Size**: `0`

### Real-Data Evaluation Status
Because zero physical images were found in `data/raw/dataset/`, real-data metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) cannot be computed on physical unseen livestock images.

---

## 3. YOLO Animal Detection Real-Data Verification

- **Model Backbone**: `yolov8n.pt` / `models/yolo_best.pt`
- **Ground-Truth Annotations**: Ground-truth bounding box coordinate annotations (`data/annotations/yolo/`) for physical test images are unavailable.

### Official YOLO Detection Audit Verdict
> *"Detection metrics not verified because validated ground-truth bounding-box annotations are unavailable."*

---

## 4. Grad-CAM Explainability Verification

- **Module**: `ml/explainability/gradcam.py` (`EfficientNetGradCAM`)
- **Hook Target**: Final convolutional feature layer `features[-1]` of EfficientNet-B0.
- **Execution Test**: Verified functional operation using synthetic test tensor inputs without modifying model weights or predictions. Real-image visual overlays will be generated upon placing physical livestock images in `data/raw/dataset/`.
