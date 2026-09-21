# Model Training Report: EfficientNet-B0 V3 (Real + Synthetic Augmentation)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model Architecture**: EfficientNet-B0 (ImageNet-1K Pretrained)  
**Dataset Augmentation**: Supplementary Synthetic Dataset (Minimum 20 images/class target)  
**Validation Isolation**: 100% Real Unseen Validation Samples ($N=84$)  
**Best Validation Accuracy**: **52.38%** (Macro F1: **37.23%**)  
**Date**: September 2026  

---

## 1. Training Setup & Hyperparameters

| Hyperparameter | Configuration |
| :--- | :--- |
| **Total Classes** | 82 (59 Cattle, 23 Buffalo) |
| **Training Partition Composition** | **1,657 images** (382 Real + 1,275 Synthetic) |
| **Minimum Training Images / Class** | **20 images** (100% of classes reach $\ge 20$) |
| **Validation Partition Composition**| **84 images** (100% Real Photographs) |
| **Held-Out Test Partition** | **123 images** (100% Real Photographs, untouched) |
| **Batch Size** | 32 |
| **Loss Function** | CrossEntropyLoss(label_smoothing=0.1) |
| **Stage 1 (Head Warmup)** | 10 Epochs, Frozen Backbone, AdamW lr=$10^{-3}$ |
| **Stage 2 (Deep Fine-Tuning)** | 5 Epochs, Unfrozen `features[6:]`, CosineAnnealingLR |

---

## 2. Validation Progression on Real Data

- Peak Real Validation Accuracy: **52.38%**
- Peak Real Validation Macro F1: **37.23%**
- Checkpoints saved:
  - `models/synthetic_82_breeds_v3/best_model_v3.pth`
  - `models/synthetic_82_breeds_v3/final_model_v3.pth`
