# Image Resolution Exploration Study

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Campaign**: High-Accuracy Optimization V2 (Phase 12)  
**Date**: September 2026  

---

## 1. Experimental Setup

Fine-grained livestock morphological features (facial contours, skin wrinkles, small horns) depend heavily on spatial input dimensions. We evaluated three standardized resolutions on the EfficientNet-B0 backbone:
- $224 \times 224$ (Standard default)
- $300 \times 300$ (Intermediate high-resolution)
- $384 \times 384$ (High-resolution fine-detail)

---

## 2. Resolution vs Performance vs CPU Latency

| Input Resolution | Spatial Pixels | CPU Inference Latency | Validation Top-1 Acc | Validation Macro F1 | Validation Top-3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **224 $\times$ 224** | 50,176 px | **17.7 ms** | 52.38% | 37.23% | 67.86% |
| **300 $\times$ 300** | 90,000 px | 31.4 ms | **54.76%** | **39.15%** | **71.43%** |
| **384 $\times$ 384** | 147,456 px | 58.9 ms | 53.57% | 38.02% | 70.24% |

---

## 3. Key Findings

1. **Resolution Sweet Spot at 300x300**: Increasing input dimensions from 224 to 300 yields a **+2.38% Top-1 Accuracy** boost, capturing fine horn tips and dewlap structures.
2. **Computational Tradeoff**: While 384x384 increases latency by 3.3× (58.9 ms), it does not provide further accuracy gains over 300x300 due to interpolation blurring in lower-resolution real photographs.
