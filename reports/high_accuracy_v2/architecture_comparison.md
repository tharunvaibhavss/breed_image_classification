# Model Architecture Comparison Study

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Campaign**: High-Accuracy Optimization V2 (Phase 10)  
**Date**: September 2026  

---

## 1. Experimental Setup

Four distinct deep learning vision backbones representing differing inductive biases were trained on the identical S2 development partition and evaluated on the locked Real Validation set:
1. **EfficientNet-B0**: Compound-scaled lightweight inverted residual architecture.
2. **ResNet50**: Classic residual learning with bottleneck blocks.
3. **DenseNet121**: Dense feature concatenation maximizing feature reuse across layers.
4. **ConvNeXt-Tiny**: Modernized pure-CNN architecture incorporating Vision Transformer design choices (inverted bottleneck, depthwise separable 7x7 convolutions).

---

## 2. Validation Set Performance & Latency Benchmark

| Architecture | Model Parameters | CPU Latency (ms) | Validation Top-1 Acc | Validation Macro F1 | Validation Top-3 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | 5.3M | **17.7 ms** | 52.38% | 37.23% | 67.86% |
| **ResNet50** | 25.6M | 48.2 ms | 47.62% | 32.18% | 64.29% |
| **DenseNet121** | 7.9M | 36.4 ms | **53.57%** | **38.41%** | **70.24%** |
| **ConvNeXt-Tiny** | 28.6M | 54.1 ms | 48.81% | 33.74% | 65.48% |

---

## 3. Comparative Analysis

1. **DenseNet121 Feature Reuse**: DenseNet121 demonstrated the strongest raw feature representation for fine-grained few-shot classes, achieving **53.57% Top-1 Accuracy** and **38.41% Macro F1**. Dense feature concatenation allows gradients from rare classes to propagate efficiently to early layers.
2. **EfficientNet-B0 Efficiency**: EfficientNet-B0 delivers competitive accuracy (52.38%) at one-half the latency (17.7 ms) and one-fifth the parameter size.
3. **Overparameterization Penalty**: Larger models (ResNet50, ConvNeXt-Tiny) exhibited mild overfitting on the limited real training images ($N=382$), yielding lower validation F1 scores.
