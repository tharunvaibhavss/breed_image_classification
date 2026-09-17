# PyTorch vs ONNX Runtime Test Comparison Report

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model**: EfficientNet-B0 (82 Breeds)  
**Hardware Platform**: CPU (Intel / AMD x86_64)  
**Evaluation Set**: 115 Unseen Test Images (`dataset/splits/test.csv`)  

---

## 1. Executive Equivalence & Numerical Fidelity

| Metric | Value | Verification Status |
| :--- | :--- | :--- |
| **Prediction Agreement** | **100.00%** (115 / 115) | **PASSED** (100% Identical) |
| **Maximum Numerical Probability Difference** | `1.68e-06` | **PASSED** (< $10^{-5}$ FP32 tolerance) |
| **Mean Numerical Probability Difference** | `1.67e-08` | **PASSED** (< $10^{-7}$) |
| **Numerical Equivalence Status** | **VALIDATED** | Identical decision outputs |

---

## 2. Inference Latency & Benchmarking

Benchmarking was conducted using 10 warm-up passes followed by full evaluation on all 115 test images on CPU.

| Execution Engine | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (JIT/Eager)** | 84.92 ms | 79.95 ms | 149.72 ms | 168.60 ms | ~11.8 FPS |
| **ONNX Runtime (CPU)** | 41.10 ms | 29.11 ms | 91.26 ms | 141.84 ms | ~24.3 FPS |
| **Speedup Factor** | **2.07x** (Mean) | **2.75x** (Median) | - | - | - |

---

## 3. Conclusions
1. **Mathematical Invariance**: Exporting the PyTorch EfficientNet-B0 graph to ONNX via opset 14 and constant folding introduces zero semantic degradation.
2. **Production Viability**: The 2.07x mean latency reduction (down to ~41 ms) makes ONNX Runtime the ideal execution engine for the FastAPI backend.
