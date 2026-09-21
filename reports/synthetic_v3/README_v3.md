# Synthetic Dataset Augmentation V3 Artifact Suite

This directory contains the complete artifact and scientific documentation suite for **Dataset Augmentation V3** of the MCA Major Research Project:
*"AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning"*.

---

## 1. Artifact Index

| Artifact Filename | Description |
| :--- | :--- |
| **`synthetic_dataset_plan_v3.md`** | Detailed scarcity distribution and per-breed target audit |
| **`synthetic_metadata_v3.csv`** | Provenance manifest for all 2,079 synthetic samples with SHA-256 hashes |
| **`synthetic_quality_report_v3.md`** | Image dimensions, RGB format, and deduplication quality audit |
| **`augmented_split_manifest_v3.csv`**| Combined partition manifest (2,461 Train, 84 Val, 123 Test) |
| **`leakage_report_v3.md`** | Cryptographic and perceptual isolation audit (Zero Leakage) |
| **`training_report_v3.md`** | Two-stage training configuration and real validation tracking |
| **`training_history_v3.csv`** | Epoch-by-epoch loss, accuracy, and learning rate progression |
| **`training_curves_v3.png`** | Training loss and validation accuracy curves |
| **`real_vs_synthetic_training_comparison.md`** | Head-to-head comparison table against V2 Real-Only baseline |
| **`final_model_evaluation_v3.md`** | Full test evaluation on 100% Real unseen test set |
| **`classification_report_v3.csv`** | Per-breed Precision, Recall, F1-Score, and Support |
| **`confusion_matrix_v3.png`** | Full 82-breed confusion matrix heatmap |
| **`prediction_distribution_v3.png`** | Frequency distribution of top predicted breeds |
| **`test_predictions_v3.csv`** | Sample-by-sample predictions on the 123 real test images |
| **`experiment_log_v3.csv`** | Reproducible experiment audit log across V1, V2, and V3 |
| **`README_v3.md`** | This summary index document |

---

## 2. Key Checkpoints & Models

| Checkpoint Name | File Path | Description |
| :--- | :--- | :--- |
| **Best Model V3** | `models/synthetic_82_breeds_v3/best_model_v3.pth` | Peak checkpoint on real validation data (52.38% Acc, 37.23% F1) |
| **Final Model V3** | `models/synthetic_82_breeds_v3/final_model_v3.pth` | Final epoch weights at the conclusion of Stage 2 fine-tuning |
| **Class Mapping V3** | `models/synthetic_82_breeds_v3/class_mapping_v3.json` | Exact 82-class index-to-breed mapping |
| **Training Config V3**| `models/synthetic_82_breeds_v3/training_config_v3.json` | Hyperparameters and dataset composition metadata |

---

## 3. Scientific Performance Summary

- **Real Training Images**: 382 (15.52%)
- **Synthetic Training Images**: 2079 (84.48%)
- **Total Training Images**: 2461 (Minimum 30 images/class target)
- **Real Unseen Test Samples**: 123 (100% Real, zero synthetic contamination)
- **Top-1 Accuracy**: **36.59%**
- **Top-3 Accuracy**: **50.41%**
- **Macro Precision**: **20.34%**
- **Macro F1-Score**: **21.41%**
- **Unique Predicted Classes**: **40 / 82**
- **Leakage Status**: **PASSED (0 overlap)**
- **Empirical Verdict**: `LIMITED IMPROVEMENT`
