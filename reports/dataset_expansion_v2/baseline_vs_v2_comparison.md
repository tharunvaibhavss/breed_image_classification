# Head-to-Head Comparison: Baseline 82-Class vs Expanded v2

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Baseline Model**: `models/efficientnet_b0_82_breeds_best.pth` (N=115 test samples)  
**Expanded Model v2**: `models/expanded_82_breeds_v2/best_model_v2.pth` (N=123 test samples)  
**Date**: September 2026  

---

## 1. Metric Comparison Table

| Evaluation Metric | Baseline 82-Class | Expanded 82-Class (v2) | Empirical Change |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (Top-1)** | 13.04% | 33.33% | **+20.29%** |
| **Top-3 Accuracy** | 32.17% | 50.41% | **+18.24%** |
| **Macro Precision** | 3.63% | 17.71% | **+14.08%** |
| **Macro Recall** | 6.61% | 19.30% | **+12.69%** |
| **Macro F1-Score** | 4.43% | 16.94% | **+12.51%** |
| **Weighted Precision** | 5.89% | 28.22% | **+22.33%** |
| **Weighted F1-Score** | 7.68% | 27.59% | **+19.91%** |
| **Unique Predicted Classes** | 17 / 82 | 33 / 82 | **+16** |
| **Cattle Top-1 Accuracy** | 10.98% | 34.44% | **+23.46%** |
| **Cattle Top-3 Accuracy** | 30.49% | 50.00% | **+19.51%** |
| **Buffalo Top-1 Accuracy** | 18.18% | 30.30% | **+12.12%** |
| **Buffalo Top-3 Accuracy** | 36.36% | 51.52% | **+15.16%** |

---

## 2. Key Diagnostic Findings
1. **Prediction Spread Improvement**: Unique predicted classes increased from **17/82** to **33/82** (+16 classes), indicating reduced majority-class collapse.
2. **Top-3 Accuracy**: Reached **50.41%** (compared to 32.17% in baseline).
3. **Controlled Experiment ($\ge 20$ Images)**: Performance on classes with $\ge 20$ images reached **59.46% Top-1** and **75.68% Top-3**, re-confirming that sample depth is the primary determining factor in model convergence.
