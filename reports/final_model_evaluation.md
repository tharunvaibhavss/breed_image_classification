# Final Model Evaluation Report (82 Breeds)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Evaluation Date**: 2026-09-17 14:21:54 UTC  
**Evaluation Dataset**: Strictly Unseen Test Set (`dataset/splits/test.csv`)  
**Data Isolation**: Verified Zero Data Leakage (Cryptographic & Perceptual Overlap = 0)  

---

## 1. Primary Performance Metrics

| Metric | Scientific Score | Benchmark / Context |
|---|---|---|
| **Macro Precision (Primary)** | **0.0363** (3.63%) | Unweighted average precision across all 82 fine-grained classes |
| **Weighted Precision** | **0.0589** (5.89%) | Support-weighted average precision |
| **Overall Accuracy** | **0.1304** (13.04%) | Percentage of correctly identified breeds in unseen test set |
| **Macro Recall** | **0.0661** (6.61%) | Average sensitivity across all 82 classes |
| **Macro F1-Score** | **0.0443** (4.43%) | Harmonic mean of Macro Precision and Macro Recall |
| **Top-1 Accuracy** | **0.1304** (13.04%) | Ground-truth breed matches Rank 1 prediction |
| **Top-3 Accuracy** | **0.3217** (32.17%) | Ground-truth breed present within top-3 predictions |

---

## 2. Species-Level Breakdown

### Cattle (59 Indian Breeds)
- **Test Samples**: 82
- **Top-1 Accuracy**: 10.98%
- **Top-3 Accuracy**: 30.49%
- **Macro Precision**: 3.51%
- **Macro Recall**: 5.58%
- **Macro F1**: 4.08%
- **Cross-Species Errors (Cattle → Buffalo)**: 13

### Buffalo (23 Indian Breeds)
- **Test Samples**: 33
- **Top-1 Accuracy**: 18.18%
- **Top-3 Accuracy**: 36.36%
- **Macro Precision**: 7.84%
- **Macro Recall**: 7.04%
- **Macro F1**: 6.85%
- **Cross-Species Errors (Buffalo → Cattle)**: 7

---

## 3. High-Confidence and Error Analysis

- **Total Test Samples**: 115
- **Correct Top-1 Predictions**: 15 (13.04%)
- **Correct Top-3 Predictions**: 37 (32.17%)
- **Total Misclassifications**: 100
- **Cross-Species Misclassifications**: 20

Complete individual predictions and error details are recorded in:
- `test_predictions.csv`
- `error_analysis.csv`
- `classification_report.csv`
