# YOLO ROI & Context-Preserving Padding Optimization Study

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Target Campaign**: High-Accuracy Optimization V2 (Phase 9)  
**Date**: September 2026  

---

## 1. Objective and Experimental Design

In fine-grained livestock breed classification, extraneous background clutter (farmland sheds, grazing foliage, human handlers, fences) can introduce non-biological confounding shortcuts. Conversely, naive tight cropping can sever diagnostic biological landmarks:
- Horn curvature and span (crucial for Jaffarabadi, Toda, Murrah, Kankrej, Amritmahal)
- Thoracic hump height and orientation (distinguishing Zebu Bos indicus from Taurine or Buffalo)
- Dewlap size and neck folds (diagnostic for Gir, Ongole, Sahiwal)
- Tail switch length and placement

We compared three bounding box extraction protocols:
1. **Experiment R1: Full Uncropped Image**
2. **Experiment R2: Raw YOLO Bounding Box Crop (0% Padding)**
3. **Experiment R3: Context-Preserving Bounding Box Crop (15% Contextual Padding)**

---

## 2. Quantitative Comparison on Development Validation Set

| Protocol | Mean Crop Resolution | Horn Retention Rate | Hump Retention Rate | Validation Top-1 Acc | Validation Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **R1: Full Image** | 2503 $\times$ 1946 px | 100.0% | 100.0% | 34.52% | 18.22% |
| **R2: Raw YOLO Crop (0% Pad)** | 1420 $\times$ 1110 px | 82.4% | 88.6% | 35.71% | 20.14% |
| **R3: Context-Preserving (15% Pad)** | **1740 $\times$ 1360 px** | **98.2%** | **99.1%** | **38.10%** | **23.45%** |

---

## 3. Scientific Findings & Qualitative Inspection

1. **Raw Crop Truncation Defect (R2)**:
   - In tight crops (R2), horns exceeding the head bounding region were clipped in 17.6% of samples. For long-horned breeds (e.g. Kankrej, Amritmahal, Toda), horn truncation directly degraded top-1 classification by eliminating key discriminative features.
2. **Context-Preserving Padding Advantage (R3)**:
   - Applying a **15% uniform contextual expansion** retained 98.2% of peripheral horn tips and complete thoracic humps while stripping $pprox 65\%$ of irrelevant background clutter.
   - Validation Macro F1 improved from 18.22% (R1) to **23.45%** (R3).
3. **Representative Saved Crops**:
   - Sample visual crops are preserved in `experiments/high_accuracy_v2/sample_crops/` (`sample_0_R1_full.jpg`, `sample_0_R2_raw_crop.jpg`, `sample_0_R3_padded_crop.jpg`) confirming landmark preservation.

---

## 4. Conclusion

Protocol **R3 (15% Padded Contextual ROI)** is officially adopted as the primary feature extraction strategy for the high-accuracy deployment pipeline.
