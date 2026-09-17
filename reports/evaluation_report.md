# Comprehensive Model Evaluation & Benchmark Report: 82 Indian Cattle and Buffalo Breeds

**Project Title**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Evaluation Scope**: 82 ICAR-NBAGR Registered Breeds (59 Cattle Breeds + 23 Buffalo Breeds)  
**Model**: EfficientNet-B0 (Transfer Learning with Two-Stage Fine-Tuning)  
**Test Partition**: Strictly Unseen Test Dataset (`dataset/splits/test.csv`, 115 images across all 82 classes)  
**Data Leakage Audit**: **PASSED** (Cryptographic overlap: 0, Perceptual overlap pHash $\le$ 4: 0)  

---

## 1. Executive Summary of Results

All reported metrics are strictly calculated from actual forward-pass predictions on the isolated, unseen test dataset. Zero metrics are fabricated or estimated.

| Metric | Scientific Score | Percentage | Notes |
|---|---|---|---|
| **Macro Precision (Primary Metric)** | **0.0363** | **3.63%** | Unweighted average precision across all 82 fine-grained classes |
| **Overall Accuracy (Top-1)** | **0.1304** | **13.04%** | Exact match between Rank 1 prediction and ground truth |
| **Top-3 Accuracy** | **0.3217** | **32.17%** | Ground truth breed present in Top-3 predicted candidates |
| **Macro Recall** | **0.0661** | **6.61%** | Unweighted average sensitivity across all 82 classes |
| **Macro F1-Score** | **0.0443** | **4.43%** | Harmonic mean of Macro Precision and Macro Recall |
| **Weighted Precision** | **0.0589** | **5.89%** | Support-weighted average precision |
| **Weighted Recall** | **0.1304** | **13.04%** | Support-weighted average recall |
| **Weighted F1-Score** | **0.0768** | **7.68%** | Support-weighted harmonic mean |
| **Total Test Samples** | **115** | 100% | Every single one of the 82 classes has test representation |
| **Total Target Classes** | **82** | 100% | 59 Cattle + 23 Buffalo |

---

## 2. Species-Specific Evaluation

### 2.1 Cattle Partition (59 Official Breeds)
- **Total Test Samples**: 82 images
- **Top-1 Accuracy**: 10.98%
- **Top-3 Accuracy**: 30.49%
- **Macro Precision**: 3.51%
- **Macro Recall**: 5.58%
- **Macro F1-Score**: 4.08%
- **Cross-Species Misclassifications (Cattle $\rightarrow$ Buffalo)**: 13 / 82 (15.85%)

### 2.2 Buffalo Partition (23 Official Breeds)
- **Total Test Samples**: 33 images
- **Top-1 Accuracy**: 18.18%
- **Top-3 Accuracy**: 36.36%
- **Macro Precision**: 7.84%
- **Macro Recall**: 7.04%
- **Macro F1-Score**: 6.85%
- **Cross-Species Misclassifications (Buffalo $\rightarrow$ Cattle)**: 7 / 33 (21.21%)

---

## 3. Data Leakage Verification Summary

A comprehensive cryptographic and perceptual hash audit was executed on the dataset partitions:
- **Exact Hash Collisions (MD5 / SHA-256)**:
  - Train vs. Test: **0**
  - Train vs. Validation: **0**
  - Validation vs. Test: **0**
- **Perceptual Hash Overlaps ($pHash \le 4$ Hamming Distance)**:
  - Train vs. Test: **0**
  - Train vs. Validation: **0**
  - Validation vs. Test: **0**
- **Deduplication Pre-Processing**:
  - 20 duplicate downloads quarantined to `dataset/rejected/duplicates/`
- **Audit Verdict**: **PASSED** (Complete isolation of evaluation sets).

---

## 4. Inference Engine Benchmarking (PyTorch vs ONNX Runtime)

The best model checkpoint (`models/efficientnet_b0_82_breeds_best.pth`) was exported to ONNX format (`models/efficientnet_b0_82_breeds.onnx`, Opset 14, constant-folded) and validated on all 115 unseen test samples:

| Benchmark Criterion | PyTorch (CPU) | ONNX Runtime (CPU) | Improvement / Parity |
|---|---|---|---|
| **Top-1 Class Agreement** | 115 / 115 | 115 / 115 | **100.00% Exact Agreement** |
| **Max Probability Difference** | - | - | **$1.68 \times 10^{-6}$** |
| **Mean Probability Difference**| - | - | **$1.67 \times 10^{-8}$** |
| **Mean Inference Latency** | 84.92 ms | 41.10 ms | **2.07× Speedup** |
| **Median Inference Latency** | 79.95 ms | 29.11 ms | **2.75× Speedup** |
| **95th Percentile Latency (P95)**| 108.45 ms | 98.20 ms | **1.10× Speedup** |
| **Model Disk Size** | 15.98 MB | 15.68 MB | **1.88% Size Reduction** |

---

## 5. Artifact and Model Inventory

| Item Description | Storage Location |
|---|---|
| **Trained Best PyTorch Model** | `models/efficientnet_b0_82_breeds_best.pth` |
| **Trained Final PyTorch Model** | `models/efficientnet_b0_82_breeds_final.pth` |
| **Exported ONNX Model** | `models/efficientnet_b0_82_breeds.onnx` |
| **Canonical Class Mapping (82 Classes)** | `models/class_names.json` |
| **Test Predictions Manifest** | `reports/test_predictions.csv` |
| **Per-Breed Classification Report** | `reports/classification_report.csv` |
| **Error Analysis Manifest** | `reports/error_analysis.csv` |
| **Full Confusion Matrix CSV** | `reports/confusion_matrix.csv` |
| **Confusion Matrix Plot** | `plots/confusion_matrix.png` |
| **Training & Validation Loss Plots** | `plots/training_loss.png`, `plots/validation_loss.png` |
| **Training & Validation Accuracy Plots** | `plots/training_accuracy.png`, `plots/validation_accuracy.png` |
| **Per-Class Precision Plot** | `plots/per_class_precision.png` |
| **Data Leakage Verification Report** | `reports/data_leakage_report.md` |
| **ONNX Validation Report** | `reports/onnx_validation_report.md` |
| **Machine-Readable Metrics JSON** | `reports/metrics.json` |

---

## 6. Reproducible Execution Scripts

- `scripts/train_efficientnet_82.py`: Two-stage transfer learning pipeline with memory pre-caching.
- `scripts/evaluate_model_82.py`: Evaluates best checkpoint on strictly unseen test partition (`test.csv`).
- `scripts/create_confusion_matrix.py`: Generates full $82 \times 82$ confusion matrix and training curves.
- `scripts/export_onnx_82.py`: Exports PyTorch weights to ONNX format with opset 14.
- `scripts/compare_pytorch_onnx.py`: Benchmarks PyTorch vs ONNX Runtime on the unseen test dataset.
