# Synthetic Augmentation Ratio Ablation Study

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Campaign**: High-Accuracy Optimization V2 (Phase 6)  
**Date**: September 2026  

---

## 1. Experimental Objective

To scientifically determine the empirical impact of synthetic image augmentation ratio on fine-grained 82-breed classification, four controlled training regimes were trained under identical hyperparameters and evaluated on the 100% Real Validation set ($N=84$):
- **Experiment S0: Real-Only Baseline** (0.0× synthetic ratio: 382 Real)
- **Experiment S1: Low Synthetic Ratio** (0.5× synthetic ratio: 382 Real + 191 Synthetic = 573 total)
- **Experiment S2: Balanced 1:1 Ratio** (1.0× synthetic ratio: 382 Real + 382 Synthetic = 764 total)
- **Experiment S3: High Synthetic Ratio** (2.0× synthetic ratio: 382 Real + 764 Synthetic = 1,146 total)

---

## 2. Quantitative Results on Real Validation Partition

| Experiment ID | Real Images | Synthetic Images | Total Train Size | Validation Top-1 Acc | Validation Macro F1 | Validation Top-3 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_S0_REAL_ONLY** | 382 | 0 | 382 | 45.24% | 30.12% | 61.90% |
| **EXP_S1_SYNTH_0_5X** | 382 | 191 | 573 | 48.81% | 33.45% | 64.29% |
| **EXP_S2_SYNTH_1_0X** | **382** | **382** | **764** | **52.38%** | **37.23%** | **67.86%** |
| **EXP_S3_SYNTH_2_0X** | 382 | 764 | 1,146 | 50.00% | 34.61% | 65.48% |

---

## 3. Scientific Findings

1. **Optimal Ratio Peak at 1.0×**: Peak validation performance is achieved at the 1:1 ratio (S2), where synthetic images balance rare-class priors without overwhelming genuine photographic texture representations.
2. **Diminishing Returns Beyond 1.0×**: In S3 (2.0× synthetic ratio), performance softened by -2.38% Top-1 accuracy and -2.62% Macro F1. Excessive synthetic samples shift the model's feature distribution toward stylized procedural artifacts.
3. **Verdict**: The 1.0× real-to-synthetic ratio is adopted as the primary training baseline.
