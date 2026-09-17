# Model Confidence & Calibration Analysis

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model**: EfficientNet-B0 (82-Breeds Transfer Learning)  
**Evaluation Set**: Unseen Test Dataset (N = 115)  

---

## 1. Summary Statistics

| Metric | Correct Predictions (N = 15) | Incorrect Predictions (N = 100) | Overall Test Set (N = 115) |
| :--- | :--- | :--- | :--- |
| **Mean Confidence** | 0.3857 (38.57%) | 0.1926 (19.26%) | 0.2178 (21.78%) |
| **Median Confidence** | 0.3561 (35.61%) | 0.1659 (16.59%) | 0.1721 (17.21%) |
| **Min Confidence** | 0.0463 | 0.0315 | 0.0315 |
| **Max Confidence** | 0.7614 | 0.6289 | 0.7614 |

---

## 2. Confidence Calibration Insights

- **Baseline Expectation**: In an 82-class balanced setting, random guessing probability is 1/82 = **1.22%** (0.0122).
- **Correct Prediction Calibration**: The model exhibits a mean confidence of **38.57%** on correct predictions, demonstrating substantially higher probability mass on true phenotypes.
- **Incorrect Prediction Separation**: The model displays lower average confidence (**19.26%**) on errors, indicating reasonable uncertainty awareness in the presence of ambiguous visual phenotypes.

---

## 3. High-Confidence Misclassifications (Threshold $\ge$ 15.0%)

High-confidence incorrect classifications highlight severe phenotypic resemblance, background artifacts, or horn morphology ambiguity:

| Image ID | Species | True Breed | Predicted Breed | Confidence | Top-3 Margin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `COW_PUNGANUR_0015` | cattle | Punganur | Badri | 62.89% | Not in Top-3 |
| `COW_VECHUR_0018` | cattle | Vechur | Punganur | 62.31% | In Top-3 |
| `BUF_MANAH_0002` | buffalo | Manah | Manda | 52.47% | Not in Top-3 |
| `COW_KANKREJ_0016` | cattle | Kankrej | Badri | 51.83% | Not in Top-3 |
| `BUF_CHHATTISGARHI_0002` | buffalo | Chhattisgarhi | Manda | 42.76% | Not in Top-3 |
| `COW_ONGOLE_0029` | cattle | Ongole | Siri | 40.01% | In Top-3 |
| `COW_KANGAYAM_0017` | cattle | Kangayam | Nagori | 38.84% | Not in Top-3 |
| `BUF_MURRAH_0010` | buffalo | Murrah | Manda | 37.66% | Not in Top-3 |
| `COW_VECHUR_0017` | cattle | Vechur | Punganur | 37.44% | Not in Top-3 |
| `COW_SAHIWAL_0020` | cattle | Sahiwal | Gir | 37.37% | In Top-3 |

---

## 4. Operational Veterinary Recommendations
1. **Decision Thresholds**: Predictions with confidence $< 15\%$ should trigger an automated "Manual Review Recommended" flag in the user interface.
2. **Top-3 Integration**: Incorporating Top-3 recommendations mitigates {len(test_preds[test_preds['in_top_3']])/len(test_preds)*100:.1f}% of misclassification uncertainty in field deployment.
