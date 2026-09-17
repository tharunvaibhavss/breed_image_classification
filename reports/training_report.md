# EfficientNet-B0 82-Class Training Report

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model Architecture**: EfficientNet-B0 Transfer Learning  
**Target Classes**: 82 (59 Indian Cattle Breeds + 23 Indian Buffalo Breeds)  
**Training Date**: 2026-09-17 14:21:12 UTC  

---

## 1. Executive Summary

| Parameter | Value |
|---|---|
| **Backbone** | EfficientNet-B0 (Pretrained on ImageNet-1K) |
| **Input Resolution** | 224 × 224 RGB |
| **Total Classes** | 82 |
| **Training Samples** | 302 |
| **Validation Samples** | 69 |
| **Batch Size** | 16 |
| **Total Epochs** | 15 (Stage 1: 10, Stage 2: 5) |
| **Best Epoch** | Epoch 5 |
| **Best Validation Accuracy** | **33.33%** |
| **Best Validation Loss** | **3.2166** |
| **Final Checkpoint Path** | `efficientnet_b0_82_breeds_final.pth` |
| **Best Checkpoint Path** | `efficientnet_b0_82_breeds_best.pth` |

---

## 2. Two-Stage Training Protocol

1. **Stage 1 (Feature Extraction, Epochs 1–10)**:
   - Frozen convolutional features backbone.
   - Classification head (`Linear(1280, 82)`) trained with AdamW ($lr = 10^{-3}$, weight decay = $10^{-2}$).
   - Fast convergence of top-layer decision boundary without destroying pretrained feature representations.

2. **Stage 2 (Full Fine-Tuning, Epochs 11–15)**:
   - Complete network unlocked for gradient backpropagation.
   - Differential learning rate ($10^{-4}$ on convolutional backbone, $5 \times 10^{-4}$ on classifier head).
   - Cosine Annealing schedule decaying learning rate smoothly toward $10^{-6}$.
   - Label smoothing regularizer ($lpha = 0.05$) to mitigate overconfidence across 82 fine-grained classes.

---

## 3. Training & Validation Progression

| Epoch | Stage | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|---|---|---|---|---|---|---|
| 1 | 1 | 3.9798 | 16.6% | 3.6983 | 17.4% | 1.00e-03 |
| 2 | 1 | 2.8417 | 51.0% | 3.4683 | 27.5% | 1.00e-03 |
| 3 | 1 | 2.2462 | 60.9% | 3.3467 | 29.0% | 1.00e-03 |
| 4 | 1 | 1.8031 | 70.9% | 3.2777 | 30.4% | 1.00e-03 |
| 5 | 1 | 1.5068 | 75.8% | 3.2166 | 33.3% | 1.00e-03 |
| 6 | 1 | 1.2698 | 86.4% | 3.1764 | 30.4% | 1.00e-03 |
| 7 | 1 | 1.0992 | 93.7% | 3.1383 | 31.9% | 1.00e-03 |
| 8 | 1 | 0.9718 | 96.7% | 3.1401 | 31.9% | 1.00e-03 |
| 9 | 1 | 0.8655 | 97.7% | 3.1281 | 31.9% | 1.00e-03 |
| 10 | 1 | 0.7966 | 98.0% | 3.1290 | 31.9% | 1.00e-03 |
| 11 | 2 | 2.2266 | 56.0% | 3.1954 | 29.0% | 9.05e-05 |
| 12 | 2 | 1.5595 | 76.5% | 3.2061 | 30.4% | 6.58e-05 |
| 13 | 2 | 1.3309 | 83.1% | 3.1754 | 30.4% | 3.52e-05 |
| 14 | 2 | 1.1854 | 84.4% | 3.1788 | 30.4% | 1.05e-05 |
| 15 | 2 | 1.0957 | 85.8% | 3.1769 | 31.9% | 1.00e-06 |

---
*Model weights saved to `efficientnet_b0_82_breeds_best.pth` are ready for unbiased unseen test evaluation.*
