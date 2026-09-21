# Model Card: EfficientNet-B0 for Indian Cattle & Buffalo Breed Recognition

**Model Name**: EfficientNet-B0 (82-Breeds Expanded Model)  
**Model Version**: `efficientnet_b0_82_breeds_v1`  
**Architecture**: Convolutional Neural Network (EfficientNet-B0 Backbone + Transfer Learning Classifier Head)  
**Number of Classes**: 82 Classes (59 Indian Cattle Breeds + 23 Indian Buffalo Breeds)  
**Input Resolution**: 224 x 224 x 3 (RGB)  
**Weights File**: `models/efficientnet_b0_82_breeds_best.pth`  
**ONNX Optimized Model**: `models/efficientnet_b0_82_breeds.onnx`  
**Training Date**: September 2026  
**Evaluation Date**: September 2026  

---

## 1. Intended Use & Domain
- **Intended Use**: Assistive automated breed recognition for Indian indigenous cattle (*Bos indicus*) and buffalo (*Bubalus bubalis*) breeds registered by ICAR-NBAGR. Designed for veterinary clinics, livestock field extension workers, agricultural research institutes, and livestock censuses.
- **Not Intended Use**: Autonomous biometric certification without veterinary oversight, or classification of non-Indian or non-registered exotic/crossbreed animals outside the 82 defined ICAR-NBAGR breeds.

---

## 2. Training Methodology
- **Pretrained Initialization**: ImageNet-1K pretrained weights.
- **Stage 1 (Feature Extraction)**: Backbone frozen, custom linear head trained with AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$) for 10 epochs.
- **Stage 2 (Fine-Tuning)**: Deeper backbone layers unfrozen, trained with AdamW (learning rate $10^{-4}$, cosine annealing scheduler) for 5 epochs.
- **Loss Function**: Cross-Entropy Loss with softmax normalization.
- **Data Augmentation (Train Only)**: Random horizontal flip, small rotation ($\pm 10^\circ$), color jitter (brightness 0.15, contrast 0.15), and affine translation.

---

## 3. Performance Metrics on Unseen Test Dataset

### 3.1 82-Breed Baseline Evaluation (N = 115)
- **Top-1 Accuracy**: **13.04%** (Random baseline: 1.22%)
- **Top-3 Accuracy**: **32.17%**
- **Macro Precision**: **3.63%**
- **Weighted Precision**: **5.89%**
- **Macro Recall**: **6.61%**
- **Macro F1-Score**: **4.43%**
- **Cattle Partition (59 classes)**: Top-1 Acc 10.98%, Top-3 Acc 30.49%, Macro Prec 3.51%
- **Buffalo Partition (23 classes)**: Top-1 Acc 18.18%, Top-3 Acc 36.36%, Macro Prec 7.84%

### 3.2 Controlled Experiments & Architectural Validation
- **Controlled Experiment ($\ge 20$ images / class, N = 23)**:
  - Classes: 6 (Manda, Gir, Ongole, Sahiwal, Surti, Siri)
  - **Top-1 Accuracy**: **65.22%**
  - **Top-3 Accuracy**: **91.30%**
  - **Macro Precision**: **66.90%** | **Macro F1**: **60.16%**
- **6-Class Prototype Control (N = 18)**:
  - Classes: 6 (Gir, Ongole, Sahiwal, Jaffarabadi, Murrah, Surti)
  - **Top-1 Accuracy**: **72.22%**
  - **Top-3 Accuracy**: **88.89%**
  - **Macro Precision**: **63.29%** | **Macro F1**: **59.37%**

### 3.3 Diagnostic Context
The 82-breed model operates in an extreme few-shot regime (averaging 3.68 training images/breed; 79.3% of breeds have $<10$ images). Controlled experiments confirm that the EfficientNet-B0 architecture and OpenCV preprocessing pipeline achieve high accuracy (65.22% - 72.22% Top-1, ~90% Top-3) when classes have adequate sample depth. Full diagnostic report: `reports/82_breed_diagnostic_report.md`.


---

## 4. Inference Latency & Hardware Benchmarks (CPU)
- **ONNX Runtime Latency**: **19.03 ms** (Median: 17.31 ms)
- **PyTorch CPU Latency**: **79.21 ms**
- **ONNX Speedup Factor**: **4.16x**
- **End-to-End Latency**: **790.66 ms**

---

## 5. Explainability & Trust
Grad-CAM heatmaps highlight relevant cranial structures (horns, ears, forehead crest) and hump contours. Grad-CAM visualizer is registered non-destructively on `features[8]` without altering inference weights.
