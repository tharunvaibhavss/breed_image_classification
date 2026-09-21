# Morphology-Preserving Augmentation Study

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Campaign**: High-Accuracy Optimization V2 (Phase 13)  
**Date**: September 2026  

---

## 1. Domain-Specific Augmentation Constraints

In livestock biometrics, naive heavy augmentations (such as extreme shearing, vertical flipping, aggressive perspective distortion, or severe color jitter) corrupt definitive breed hallmarks:
- Vertical flips invert anatomical orientation (legs on top, hump at bottom).
- Heavy perspective warping alters horn curvature angles.
- Intense hue shifts convert characteristic white-grey coats (e.g. Ongole, Hariana) into red/brown, triggering spurious misclassifications.

---

## 2. Controlled Regimes Evaluated

1. **Regime A (Baseline Augmentation)**: Random crop + standard horizontal flip ($p=0.5$).
2. **Regime B (Morphology-Safe Augmentation)**:
   - Subtle rotation ($\le 5^\circ$)
   - Micro-affine translation ($\pm 4\%$)
   - Mild scale variance ($0.96 \dots 1.04$)
   - Subtle illumination jitter (brightness $\pm 10\%$, contrast $\pm 10\%$)
   - Horizontal flip ($p=0.5$)
3. **Regime C (Aggressive Augmentation)**: Strong rotation ($\pm 25^\circ$), RandAugment, Cutout.

---

## 3. Results on Real Validation Set

| Augmentation Regime | Validation Top-1 Acc | Validation Macro F1 | Validation Top-3 | Impact on Morphology |
| :--- | :---: | :---: | :---: | :--- |
| **Regime A (Baseline)** | 52.38% | 37.23% | 67.86% | Minimal |
| **Regime B (Morphology-Safe)** | **54.76%** | **39.52%** | **71.43%** | **Preserves diagnostic traits** |
| **Regime C (Aggressive)** | 46.43% | 29.84% | 63.10% | Severe distortion |

Regime B is adopted across all production training pipelines.
