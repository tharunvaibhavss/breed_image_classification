# Multi-Architecture Model Ensemble Study

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Campaign**: High-Accuracy Optimization V2 (Phase 16)  
**Date**: September 2026  

---

## 1. Ensemble Architecture & Weighting

To combine complementary feature representations without label leakage, predictions from the top three performing distinct backbones were blended via calibrated probability averaging:
- **Model 1 (Weight: 0.45)**: Optimized EfficientNet-B0 (Morphology-Safe Augmentation)
- **Model 2 (Weight: 0.30)**: DenseNet121 (Dense Feature Reuse)
- **Model 3 (Weight: 0.25)**: ResNet50 (Residual Bottleneck Representation)

---

## 2. Performance Comparison

| Model Configuration | Validation Top-1 Acc | Validation Macro F1 | Validation Top-3 Acc | Test Set Top-1 Acc | Test Set Top-3 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single Best (EfficientNet-B0)** | 54.76% | 39.52% | 71.43% | 36.59% | 50.41% |
| **Single Best (DenseNet121)** | 53.57% | 38.41% | 70.24% | 35.77% | 49.59% |
| **Multi-Architecture Ensemble** | **57.14%** | **42.30%** | **75.00%** | **38.21%** | **52.85%** |
| **Ensemble Advantage** | **+2.38%** | **+2.78%** | **+3.57%** | **+1.62%** | **+2.44%** |

Ensembling yields an empirical gain of +1.62% Top-1 accuracy and +2.44% Top-3 accuracy on the unseen real test set.
