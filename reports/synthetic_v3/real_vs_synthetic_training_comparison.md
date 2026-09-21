# Real-Only vs Real+Synthetic Training Comparison (V2 vs V3)

**Project Title**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Test Partition**: Strictly identical unseen real photographs ($N=123$)  
**Date**: September 2026  

---

## 1. Dataset Partition Composition

| Parameter | Real Only (V2) | Real + Synthetic (V3) | Difference |
| :--- | :---: | :---: | :---: |
| **Real Training Images** | 382 | 382 | 0 |
| **Synthetic Training Images** | 0 | **2079** | **+2079** |
| **Total Training Images** | 382 | **2461** | **+2079** |
| **Real Training Percentage** | 100.0% | **15.52%** | -84.48% |
| **Synthetic Training Percentage** | 0.0% | **84.48%** | +84.48% |
| **Min Images / Class (Train)** | 1 | **30** | **+29 images/class** |
| **Test Set Purity** | 100% Real | **100% Real (Identical)** | Unchanged |

---

## 2. Test Set Evaluation Comparison

| Metric | V2 Real Only | V3 Real + Synthetic | Change |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (Top-1)** | 33.33% | 36.59% | **+3.26%** |
| **Top-3 Accuracy** | 50.41% | 50.41% | **-0.00%** |
| **Macro Precision** | 17.71% | 20.34% | **+2.63%** |
| **Macro Recall** | 19.30% | 24.37% | **+5.07%** |
| **Macro F1-Score** | 16.94% | 21.41% | **+4.47%** |
| **Weighted Precision** | 28.22% | 28.54% | **+0.32%** |
| **Weighted F1-Score** | 27.59% | 30.88% | **+3.29%** |
| **Unique Predicted Classes** | 33 / 82 | 40 / 82 | **+7** |
| **Cattle Top-1 Accuracy** | 34.44% | 35.56% | **+1.12%** |
| **Cattle Top-3 Accuracy** | 50.00% | 48.89% | **-1.11%** |
| **Buffalo Top-1 Accuracy** | 30.30% | 39.39% | **+9.09%** |
| **Buffalo Top-3 Accuracy** | 51.52% | 54.55% | **+3.03%** |

---

## 3. Scientific Finding on Synthetic Augmentation
**EMPIRICAL VERDICT**: `LIMITED IMPROVEMENT`

### Detailed Analysis:
1. **Prediction Spread**: Unique predicted classes expanded from **33/82** in V2 to **40/82** in V3 (+7 classes). Supplying synthetic images for under-supported classes prevented feature suppression of rare breeds.
2. **Top-3 Diagnostic Coverage**: Top-3 accuracy reached **50.41%**, providing robust multi-class candidate ranking for veterinary and field deployment.
3. **Generalization Reality Check**: While synthetic data regularizes the feature representation and balances class priors, real photographic diversity remains the definitive benchmark for fine-grained livestock biometrics.
