# Model Training Report: EfficientNet-B0 v2 (82 Breeds)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model Checkpoint**: `models/expanded_82_breeds_v2/best_model_v2.pth`  
**Training Date**: September 2026  
**Status**: Completed  

---

## 1. Training Configuration & Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Base Architecture** | EfficientNet-B0 (Torchvision ImageNet Pretrained) |
| **Input Shape** | 224 x 224 x 3 (RGB) |
| **Classifier Head** | Dropout(0.2) + Linear(1280, 82) |
| **Loss Function** | CrossEntropyLoss(label_smoothing=0.1) |
| **Stage 1 (Warmup)** | 10 Epochs, Frozen Backbone, AdamW lr=$10^{-3}$, weight_decay=$10^{-4}$ |
| **Stage 2 (Fine-Tuning)** | 5 Epochs, Unfrozen `features[6:]`, AdamW (backbone lr=$10^{-4}$, head lr=$5\times 10^{-4}$), CosineAnnealingLR |
| **Batch Size** | 16 |
| **Training Samples** | 382 |
| **Validation Samples** | 84 |

---

## 2. Validation Progression

| Milestone | Accuracy | Macro F1 | Epoch |
| :--- | :---: | :---: | :---: |
| **Stage 1 Best** | 42.86% | 27.14% | 5 |
| **Stage 2 Final** | 47.62% | 30.81% | 15 |
| **Overall Best Validation** | **50.00%** | **29.88%** | - |

---

## 3. Training Artifacts
- Checkpoint: `models/expanded_82_breeds_v2/best_model_v2.pth`
- Final Weights: `models/expanded_82_breeds_v2/final_model_v2.pth`
- Training Curves: `reports/dataset_expansion_v2/training_curves_v2.png`
- History CSV: `reports/dataset_expansion_v2/training_history_v2.csv`
