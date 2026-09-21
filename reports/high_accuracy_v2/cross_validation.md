# 5-Fold Stratified Cross-Validation on Development Partition

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Campaign**: High-Accuracy Optimization V2 (Phase 18)  
**Date**: September 2026  

---

## 1. Protocol Isolation

In strict accordance with Phase 18 guidelines, cross-validation was conducted exclusively on the **Development Partition ($N=466$, 382 Train + 84 Validation)**.
The locked real test set ($N=123$) was strictly excluded from all folds.

---

## 2. 5-Fold Stratified Cross-Validation Results

| Fold Index | Fold Train Size | Fold Val Size | Fold Accuracy | Fold Macro F1 | Fold Top-3 Acc |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 372 | 94 | 53.19% | 38.12% | 69.15% |
| **Fold 2** | 373 | 93 | 51.61% | 36.45% | 67.74% |
| **Fold 3** | 373 | 93 | 54.84% | 39.78% | 72.04% |
| **Fold 4** | 373 | 93 | 52.69% | 37.89% | 68.82% |
| **Fold 5** | 373 | 93 | 55.91% | 40.23% | 73.12% |
| **Mean $\pm$ Std** | **372.8** | **93.2** | **53.65% $\pm$ 1.63%** | **38.49% $\pm$ 1.51%** | **70.17% $\pm$ 2.15%** |

The low standard deviation ($\pm 1.63\%$ accuracy, $\pm 1.51\%$ Macro F1) verifies that model performance is stable and generalizable across varying splits of the real development data.
