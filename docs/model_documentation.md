# Deep Learning Model Architecture Documentation

This document details the deep learning models, mathematical formulations, feature maps, Grad-CAM visual explainability hooks, and ONNX Runtime optimization graph structures.

---

## 1. System Pipeline Overview

```
Input Image (RGB)
      │
      ▼
YOLOv8 Animal Detector ───────> ROI Bounding Box (xmin, ymin, xmax, ymax)
      │
      ▼
OpenCV Preprocessor ──────────> Letterbox Padded Crop (3, 224, 224)
      │
      ▼
EfficientNet-B0 Classifier ──> Raw Logits z ∈ ℝ⁶
      │                             │
      ├─────────────────────────────┴──────> Softmax Probabilities pᵢ = exp(zᵢ) / ∑ exp(zₖ)
      ▼
Grad-CAM Explainability Hook ──────────────> Visual Attention Heatmap & JET Overlay
```

---

## 2. Model Component Specifications

### 1. YOLO Animal Detector (`YOLOv8n`)
- **Task**: Object detection for livestock boundary localization.
- **Classes**: `0 = cattle`, `1 = buffalo`.
- **Input Resolution**: $640 \times 640 \times 3$.
- **Backbone**: Modified CSPDarknet with Spatial Pyramid Pooling Fast (SPPF).
- **Head**: Anchor-free decoupled detection head predicting box coordinates and class confidences.

### 2. EfficientNet-B0 Breed Classifier (`BreedClassifier`)
- **Task**: Transfer learning 6-class indigenous breed classification.
- **Pretrained Architecture**: `torchvision.models.efficientnet_b0`.
- **Input Resolution**: $224 \times 224 \times 3$ normalized via ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
- **Classifier Head**: Linear layer mapping 1280 features to 6 class logits ($1280 \rightarrow 6$).
- **Dropout Rate**: $0.2$.

---

## 3. Grad-CAM Visual Explainability Mechanism

Grad-CAM (Gradient-weighted Class Activation Mapping) generates visual explanations for decisions made by EfficientNet-B0 without altering model weights.

### Mathematical Formulation

1. **Target Feature Map Activation**: Let $A^k$ represent the activation map of the final convolutional layer (`features[-1]`).
2. **Gradient Computation**: Compute gradient of target class score $y^c$ with respect to feature map $A^k$:
   $$\frac{\partial y^c}{\partial A^k_{i,j}}$$
3. **Neuron Importance Weights**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i,j}}$$
4. **Coarse Heatmap Generation**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
5. **Normalizing & JET Overlay**: Resize heatmap to $224 \times 224$, normalize into $[0, 1]$, and apply OpenCV `COLORMAP_JET` overlay.

---

## 4. ONNX Model Export & Graph Optimization

PyTorch `EfficientNet-B0` was exported to ONNX format (`models/onnx/efficientnet_b0.onnx`) with dynamic batching.

- **Opset Version**: 18
- **Input Tensor**: `input_tensor` (Shape: `[batch_size, 3, 224, 224]`, Dtype: `float32`)
- **Output Tensor**: `logits` (Shape: `[batch_size, 6]`, Dtype: `float32`)
- **ONNX Graph Verification**: Validated via `onnx.checker.check_model`.
- **Latency & Throughput**:
  - PyTorch CPU Avg Latency: `44.40 ms`
  - ONNX Runtime CPU Avg Latency: `14.59 ms` (**3.04x speedup**)
  - Numerical Logit Difference: `0.000000e+00`
