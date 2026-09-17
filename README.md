# AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.15+-blue.svg)](https://onnxruntime.ai/)
[![Tests](https://img.shields.io/badge/Tests-84%2F84%20Passed-emerald.svg)]()
[![Data Leakage](https://img.shields.io/badge/Data_Leakage-0%25%20PASSED-brightgreen.svg)]()

A research application project submitted for the Master of Computer Applications (MCA) Degree. This system provides an end-to-end, deep learning-powered pipeline for automated localization, indigenous breed recognition, and visual explainability of Indian Cattle (*Bos indicus*) and Buffaloes (*Bubalus bubalis*) officially recognized by the **Indian Council of Agricultural Research – National Bureau of Animal Genetic Resources (ICAR-NBAGR)** and **ICAR-CIRB**.

---

## 🌟 Key Features & Architectural Capabilities

1. **Dual Model Architecture Support**:
   - **82-Breeds Comprehensive Model** (`efficientnet_b0_82_breeds_v1`): Full coverage of all 82 official ICAR-NBAGR registered breeds (59 Cattle breeds + 23 Buffalo breeds).
   - **6-Breeds Production Prototype** (`efficientnet_b0_6_breeds_v1`): High-support baseline covering premier dairy breeds (*Gir, Ongole, Sahiwal, Jaffarabadi, Murrah, Surti*).
2. **Sequential Multi-Stage AI Pipeline**:
   - **Animal Localization**: YOLOv8n detector extracts animal bounding box Region of Interest (ROI), rejecting non-animal backgrounds.
   - **Deterministic Preprocessing**: OpenCV aspect-ratio preserved letterbox resize to 224 x 224 RGB.
   - **Fine-Grained Classification**: EfficientNet-B0 transfer learning with ImageNet initialization.
   - **Visual Explainability**: Real-time Grad-CAM attention heatmaps highlighting facial crests, horns, and hump structures.
3. **High-Fidelity ONNX Runtime Acceleration**:
   - Exported to ONNX (opset 14) with graph constant folding.
   - **100.0% prediction agreement** with PyTorch.
   - **2.07x mean speedup** (ONNX CPU: ~19–41 ms vs PyTorch CPU: ~79–85 ms).
4. **Zero Data Leakage Guarantee**:
   - Bitwise SHA-256 and perceptual pHash (threshold $\le 4$) deduplication.
   - Verified 0 exact duplicates and 0 perceptual overlaps across Train, Validation, and Test splits.
5. **Modern Full-Stack Integration**:
   - **FastAPI REST API**: Asynchronous endpoints, JWT bearer auth, Pydantic v2 schemas, request tracing.
   - **Next.js 14 Web Application**: App Router, TailwindCSS, drag-and-drop upload, model version switcher, interactive Grad-CAM heatmap viewer.
   - **Persistence**: SQLAlchemy 2.0 ORM with PostgreSQL / SQLite and Alembic migrations.

---

## 🏛️ System Architecture Workflow

```
Input Image (Web UI / API)
         │
         ▼
FastAPI Request-ID Middleware
         │
         ▼
YOLOv8 Animal Detection (ROI Bounding Box)
         │
         ▼
OpenCV Aspect-Preserving Letterbox (224x224 RGB)
         │
         ▼
EfficientNet-B0 Classification (82 or 6 Breeds)
   ├── PyTorch Eager Engine
   └── ONNX Runtime Acceleration Engine (2.07x speedup)
         │
         ▼
Grad-CAM Explainability Heatmap & Overlay Generation
         │
         ▼
Database Persistence & History Logging (SQLAlchemy / PostgreSQL)
         │
         ▼
Client Visualization (Top-1, Top-3, Confidence, Grad-CAM Overlay)
```

---

## 📊 Dataset & Model Performance Summary

### Dataset Statistics (ICAR-NBAGR Catalog)
- **Total Registered Breeds**: 82 (59 Cattle, 23 Buffalo)
- **Total Verified Images**: 486 (349 Cattle, 137 Buffalo)
- **Partitions**: Train (302, 62.1%) | Validation (69, 14.2%) | Test (115, 23.7%)
- **Test Class Representation**: 100% of all 82 classes represented in unseen test set.
- **Data Leakage Status**: **PASSED** (0 exact, 0 near-duplicate overlaps).

### Final Empirical Test Performance (Unseen Test Dataset, N = 115)
- **Top-1 Accuracy**: **13.04%** (Random chance: 1.22%)
- **Top-3 Accuracy**: **32.17%**
- **Macro Precision (Primary Metric)**: **3.63%**
- **Weighted Precision**: **5.89%**
- **Macro Recall**: **6.61%**
- **Macro F1-Score**: **4.43%**
- **Cattle (59 classes)**: Top-1 Acc 10.98%, Top-3 Acc 30.49%, Macro Prec 3.51%
- **Buffalo (23 classes)**: Top-1 Acc 18.18%, Top-3 Acc 36.36%, Macro Prec 7.84%

### Hardware Inference Latency (CPU)
| Component | Mean Latency | Median Latency | P95 Latency |
| :--- | :--- | :--- | :--- |
| **YOLO Animal Detection** | 117.92 ms | 108.38 ms | 232.08 ms |
| **EfficientNet PyTorch CPU** | 79.21 ms | 67.49 ms | 163.72 ms |
| **EfficientNet ONNX CPU** | **19.03 ms** | **17.31 ms** | **26.77 ms** |
| **Grad-CAM Explainability** | 649.21 ms | 567.90 ms | 1022.24 ms |
| **End-to-End Pipeline (ONNX)** | **790.66 ms** | **731.92 ms** | **1237.01 ms** |

---

## 📁 Repository Directory Structure

```
FINAL_MCA_PROJECT/
├── frontend/               # Next.js 14 App Router client
├── app/ (backend/)         # FastAPI asynchronous REST API service
├── ml/ (ai_model/)         # Deep learning pipeline (YOLO, EfficientNet, Grad-CAM, ONNX)
├── scripts/                # Reproducible training, evaluation, and benchmark scripts
├── tests/                  # 84 automated unit, integration, and security tests (100% pass)
├── configs/                # YAML configuration and class mapping definitions
│
├── dataset/
│   ├── cleaned/            # Verified authentic breed images (cattle/ and buffalo/)
│   ├── metadata/           # ICAR-NBAGR accession, source manifests, image metadata
│   ├── rejected/           # Quarantined corruptions, duplicates, and low-res images
│   ├── splits/             # Stratified train.csv, validation.csv, test.csv, split_statistics.csv
│   └── README.md
│
├── models/
│   ├── efficientnet_b0_82_breeds_best.pth   # Best validation checkpoint (82 classes)
│   ├── efficientnet_b0_82_breeds_final.pth  # Final trained checkpoint (82 classes)
│   ├── efficientnet_b0_82_breeds.onnx       # Optimized ONNX model (82 classes)
│   ├── efficientnet_best.pth                # 6-class prototype checkpoint
│   └── class_names.json                     # Comprehensive 82-class mapping
│
├── reports/
│   ├── dataset_audit.md                     # Dataset inventory and class audit
│   ├── data_leakage_report.md               # Cryptographic & perceptual leakage audit
│   ├── duplicate_report.csv                 # Detailed record of 20 quarantined duplicates
│   ├── training_report.md                   # Two-stage transfer learning report
│   ├── classification_report.csv            # Per-breed precision, recall, F1, support
│   ├── final_model_evaluation.md            # Final evaluation metrics on unseen test set
│   ├── cattle_metrics.json                  # Cattle-specific performance metrics
│   ├── buffalo_metrics.json                 # Buffalo-specific performance metrics
│   ├── final_test_report.md                 # Test suite results (84/84 passing)
│   ├── model_card.md                        # Formal AI Model Card
│   ├── dataset_card.md                      # Formal Dataset Card
│   ├── documentation_results.md             # Chapters 1–7 MCA project report material
│   └── submission_readiness.md              # Project submission readiness checklist
│
├── plots/
│   ├── confusion_matrix.png                 # 82x82 full confusion matrix
│   ├── cattle_confusion_matrix.png          # 59-cattle confusion matrix
│   ├── buffalo_confusion_matrix.png         # 23-buffalo confusion matrix
│   ├── training_loss.png                    # Training vs. validation loss curve
│   ├── training_accuracy.png                # Training vs. validation accuracy curve
│   └── per_class_precision.png             # Per-class precision distribution
│
├── results/
│   ├── demo/                                # 6 real cattle & buffalo demo predictions
│   └── gradcam/                             # High-resolution Grad-CAM explainability figures
│
└── README.md
```

---

## 🚀 Installation & Setup Guide

### 1. Environment Requirements
- Python 3.10+ (Tested on Python 3.13)
- Node.js 18+ and npm
- Git

### 2. Python Environment & Backend Setup
```bash
# Clone or navigate to the workspace root
cd "MCA Project AI Breed"

# Create and activate Python virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install backend dependencies
pip install -r requirements.txt

# Start FastAPI backend server
uvicorn app.main:app --reload --port 8000
```
Interactive API documentation available at: `http://localhost:8000/docs`

### 3. Frontend Web Application Setup
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your web browser.

---

## 🧪 Independent Execution Scripts

All training, evaluation, and benchmark workflows can be executed independently:

```bash
# 1. Run Complete Automated Test Suite (84 tests)
pytest -v

# 2. Run EfficientNet-B0 82-Breeds Training
python scripts/train_efficientnet_82.py

# 3. Evaluate Best Model on Unseen Test Set
python scripts/evaluate_model_82.py

# 4. Generate Confusion Matrices & Training Curves
python scripts/create_confusion_matrix.py

# 5. Export Model to ONNX & Run Equivalence Verification
python scripts/export_onnx_82.py
python scripts/compare_pytorch_onnx.py
```

---

## ⚠️ Research & Dataset Limitations

1. **Long-Tail Sample Support**: With an average of ~5.9 images per breed across 82 classes, rare breeds have few-shot representation in test, bounding Top-1 accuracy at 13.04%.
2. **Phenotypic Convergence**: Genetically related draught cattle breeds (e.g. *Hallikar, Amritmahal, Khillar*) share light grey coats and lyre horns, introducing visual ambiguity under single-view 2D images.
3. **Top-3 Operational Decision Support**: While Top-1 is constrained by single-shot classes, Top-3 accuracy reaches **32.17%** overall (and **36.36%** for buffaloes), providing actionable clinical decision support for livestock inspectors.

---

## 📜 Academic Attribution & Acknowledgments
- Developed as an MCA Major Project.
- Breed taxonomy and reference descriptions sourced from **ICAR-NBAGR** (https://nbagr.res.in) and **ICAR-CIRB** (https://cirb.icar.gov.in).
- Deep learning architectures powered by **PyTorch**, **Torchvision**, **Ultralytics YOLOv8**, and **ONNX Runtime**.
