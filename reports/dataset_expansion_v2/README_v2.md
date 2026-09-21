# Dataset Expansion v2 & Model v2 (82 Indian Cattle & Buffalo Breeds)

This directory contains the complete artifact and documentation suite for **Dataset Expansion v2** of the MCA Major Research Project:
*"AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning"*.

---

## 1. Directory Structure

```
reports/dataset_expansion_v2/
├── dataset_expansion_v2_report.md       # Comprehensive scientific research report
├── dataset_quality_report_v2.md         # Image dimensions, formats, and deduplication audit
├── breed_label_audit_v2.csv             # Morphological validation against ICAR-NBAGR standards
├── split_manifest_v2.csv                # Complete image partition manifest (Train/Val/Test)
├── leakage_report_v2.md                 # Cryptographic and perceptual leakage verification
├── under_supported_classes.csv          # Audit of rare breeds under the 50-image target
├── training_report_v2.md                # Two-stage fine-tuning setup and progression
├── training_history_v2.csv              # Epoch-by-epoch loss and accuracy metrics
├── training_curves_v2.png               # Loss and accuracy curve visualizations
├── baseline_vs_v2_comparison.md         # Detailed markdown comparison of Baseline 1 vs v2
├── baseline_vs_v2_comparison.csv        # Percentage-point comparison table
├── final_model_evaluation_v2.md         # Unseen test set evaluation summary
├── test_predictions_v2.csv              # Raw test prediction log for all 123 test samples
├── classification_report_v2.csv         # Per-breed Precision, Recall, F1, and Support
├── confusion_matrix_v2.csv              # Full 82x82 confusion matrix
├── confusion_matrix_v2.png              # High-resolution heatmap of confusion matrix
├── prediction_distribution_v2.png       # Bar chart of prediction frequency across classes
├── confidence_analysis_v2.md            # Calibration and high-confidence error analysis
├── onnx_validation_report_v2.md         # Numerical parity and latency benchmark report
├── onnx_validation_report_v2.json       # Machine-readable ONNX benchmark metrics
├── experiment_log_v2.csv                # Reproducible experiment log across all iterations
└── README_v2.md                         # This summary index document
```

---

## 2. Key Checkpoints & Models

| Checkpoint Name | File Path | Description |
| :--- | :--- | :--- |
| **Best Model v2** | `models/expanded_82_breeds_v2/best_model_v2.pth` | Best checkpoint selected via validation accuracy (50.00%) |
| **Final Model v2** | `models/expanded_82_breeds_v2/final_model_v2.pth` | Final epoch weights at the conclusion of Stage 2 fine-tuning |
| **Optimized ONNX** | `models/expanded_82_breeds_v2/efficientnet_b0_v2.onnx` | Graph-folded ONNX Runtime model (Opset 14, 22.86 ms CPU latency) |
| **Class Mapping v2** | `models/expanded_82_breeds_v2/class_mapping_v2.json` | Exact 82-class index-to-breed mapping |
| **Training Config** | `models/expanded_82_breeds_v2/training_config_v2.json` | Machine-readable hyperparameters and metadata |

---

## 3. Quick Performance Overview

- **Top-1 Accuracy**: **33.33%** (vs. 13.04% in Baseline 1, **+20.29%**)
- **Top-3 Accuracy**: **50.41%** (vs. 32.17% in Baseline 1, **+18.24%**)
- **Macro Precision**: **17.71%** (vs. 3.63% in Baseline 1, **+14.08%**)
- **Macro F1-Score**: **16.94%** (vs. 4.43% in Baseline 1, **+12.51%**)
- **Unique Predicted Classes**: **33 / 82** (vs. 17 / 82 in Baseline 1, **+16 classes**)
- **Data Leakage**: **0 Overlap (PASSED)**
- **Inference Speedup**: **3.58x faster** with ONNX Runtime CPU (22.86 ms vs 81.87 ms PyTorch CPU)
- **Status**: **IMPROVED BUT REQUIRES MORE DATA**
