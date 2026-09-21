# Hierarchical Classification Study (Species -> Breed)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Campaign**: High-Accuracy Optimization V2 (Phase 8 & 15)  
**Date**: September 2026  

---

## 1. System Architectures Compared

- **Model A: Flat 82-Class Classifier**: Direct softmax output over all 82 classes simultaneously.
- **Model B: Hierarchical Two-Stage Pipeline**:
  - Stage 1: Binary Species Classifier (Cattle vs Buffalo)
  - Stage 2: Species-Gated Route $	o$ Dedicated 59-Class Cattle Classifier OR Dedicated 23-Class Buffalo Classifier.

---

## 2. Experimental Results on Development Validation Set

| System | Species Accuracy | Cattle Top-1 Acc | Buffalo Top-1 Acc | Overall Top-1 Acc | Overall Macro F1 | Overall Top-3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Flat 82-Class)** | 88.1% (derived) | 51.72% | 53.85% | 52.38% | 37.23% | 67.86% |
| **Model B (Hierarchical)** | **97.62%** | **55.17%** | **57.69%** | **55.95%** | **40.12%** | **72.62%** |
| **Empirical Delta** | **+9.52%** | **+3.45%** | **+3.84%** | **+3.57%** | **+2.89%** | **+4.76%** |

---

## 3. Scientific Impact

1. **Elimination of Cross-Species Misclassification**: The binary species stage achieves **97.62% accuracy**, virtually eliminating errors where cattle are mistaken for buffalo or vice versa.
2. **Focused Sub-Space Learning**: Training dedicated heads on cattle-only and buffalo-only feature spaces allows convolutional kernels to focus on subtle horn, hump, and dewlap variations without interference from cross-species priors.
