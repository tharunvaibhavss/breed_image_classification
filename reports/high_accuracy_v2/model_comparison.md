# Comprehensive Model Evolution and Progression Summary

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Campaign**: High-Accuracy Optimization V2  
**Date**: September 2026  

---

## 1. Project Experimental Evolution (V1 $	o$ V2 $	o$ V3 $	o$ High-Accuracy V2)

| Iteration / Experiment | Training Strategy | Real Train | Synth Train | Real Val | Real Test | Test Top-1 Acc | Test Top-3 Acc | Test Macro F1 | Unique Pred |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline V1** | Real-Only Raw Baseline | 302 | 0 | 69 | 115 | 13.04% | 32.17% | 4.43% | 17 / 82 |
| **Expanded V2** | Cleaned Real Dataset | 382 | 0 | 84 | 123 | 33.33% | 50.41% | 16.94% | 33 / 82 |
| **Augmented V3** | Minimum 30 Synthetic Augmentation | 382 | 2,079 | 84 | 123 | 36.59% | 50.41% | 21.41% | 40 / 82 |
| **High-Accuracy V2 (Ensemble)** | **Multi-Arch Ensemble + Padded ROI** | **382** | **382** | **84** | **123** | **38.21%** | **52.85%** | **23.15%** | **44 / 82** |
| **Total Project Gain** | — | — | — | — | — | **+25.17%** | **+20.68%** | **+18.72%** | **+27 classes** |

---

## 2. Benchmark Against 92% Target

- **Aspirational Target**: 92.00%
- **Highest Empirically Verified Accuracy**: **38.21%** (Top-1), **52.85%** (Top-3)
- **Status**: **NOT ACHIEVED**
- **Explanation**: 92% Top-1 accuracy cannot be attained without rectifying the primary physical bottleneck: gathering an estimated 6,000–8,000 additional genuine real photographs to resolve the acute few-shot scarcity where 64.6% of breeds currently possess fewer than 5 real images.
