# Prediction Confidence & Calibration Analysis: Model v2

**Model Checkpoint**: `models/expanded_82_breeds_v2/best_model_v2.pth`  
**Test Set Size**: 123 unseen images  
**Evaluation Date**: September 2026  

---

## 1. Confidence Metrics Summary

| Confidence Metric | Overall | Correct Predictions (N=41) | Incorrect Predictions (N=82) |
| :--- | :---: | :---: | :---: |
| **Mean Confidence** | **31.98%** | **46.83%** | **24.55%** |
| **Median Confidence** | **23.42%** | **42.09%** | **18.18%** |

---

## 2. High-Confidence Incorrect Predictions ($\ge 50\%$)

Total high-confidence errors: **7**

| Image ID | True Breed | Predicted Breed | Confidence | In Top-3 |
| :--- | :--- | :--- | :---: | :---: |
| BUF_BUFFALO_MANDA_0021 | Manda (buffalo) | Dangi (cattle) | 51.54% | False |
| BUF_BUFFALO_MEHSANA_0010 | Mehsana (buffalo) | Manda (buffalo) | 86.71% | True |
| BUF_BUFFALO_TODA_0014 | Toda (buffalo) | Dangi (cattle) | 55.39% | True |
| CAT_COW_NARI_0003 | Nari (cattle) | Dangi (cattle) | 50.09% | False |
| CAT_COW_PERIYAR_0004 | Periyar (cattle) | Manda (buffalo) | 79.90% | False |
| CAT_COW_PODA_THURPU_0001 | Poda Thurpu (cattle) | Gir (cattle) | 66.04% | True |
| CAT_COW_PUNGANUR_0007 | Punganur (cattle) | Vechur (cattle) | 71.96% | True |
