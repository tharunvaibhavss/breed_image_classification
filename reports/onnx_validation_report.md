# ONNX Export and Benchmark Validation Report (82 Breeds)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model**: EfficientNet-B0 (82 Classes)  
**Status**: **PASSED**  

---

## 1. Parity and Numerical Verification

| Metric | PyTorch Inference | ONNX Runtime | Parity / Difference |
|---|---|---|---|
| **Top-1 Class Agreement** | - | - | **100.00%** (115/115 samples) |
| **Max Absolute Diff (Softmax)** | - | - | **1.683831e-06** |
| **Mean Absolute Diff (Softmax)**| - | - | **1.669692e-08** |
| **Model Disk Size** | 15.98 MB | 15.68 MB | -1.9% |

---

## 2. Latency Benchmarks (CPU)

| Metric | PyTorch (CPU) | ONNX Runtime (CPU) | Speedup |
|---|---|---|---|
| **Mean Latency** | 84.92 ms | 41.10 ms | **2.07x** |
| **Median Latency** | 79.95 ms | 29.11 ms | 2.75x |
| **95th Percentile (P95)** | 149.72 ms | 91.26 ms | 1.64x |
| **99th Percentile (P99)** | 168.60 ms | 141.84 ms | 1.19x |
