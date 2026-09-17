# Known Limitations and Operational Constraints

This document outlines known edge cases, hardware boundaries, and operational constraints for the **AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes**.

---

## 1. Edge Cases and Visual Ambiguities

1. **Visually Similar Regional Sub-Types**:
   - **Cattle**: Young calves of *Gir* vs *Sahiwal* share similar reddish-brown coat colors. Horn structure development in adult animals significantly improves classification confidence.
   - **Buffalo**: *Surti* and *Jaffarabadi* buffaloes exhibit similar black pigmentation; distinction relies heavily on horn curvature and sickle shape.
   - **Mitigation**: Top-3 candidate ranking provides correct breed predictions in 97.78% of ambiguous test cases.

2. **Extreme Low-Light & Severe Shadowing**:
   - High ISO noise and heavy shadowing (e.g. unlit night sheds) reduce YOLO detection confidence below 0.25.
   - **Mitigation**: Preprocessing applies contrast normalization, and an explicit API alert ("Low detection confidence") advises users to retry with adequate illumination.

3. **Heavy Occlusion & Partial Frames**:
   - If more than 60% of the animal is obscured (e.g., behind high wooden stall walls), classification reliance shifts exclusively to visible head or flank regions.

---

## 2. Hardware and Operational Constraints

1. **CPU vs GPU Latency**:
   - **CPU Hosting**: PyTorch CPU latency is ~44.40 ms per crop. Enabling the ONNX Runtime backend (`INFERENCE_BACKEND=onnx`) reduces latency to ~14.59 ms.
   - **GPU Acceleration**: CUDA execution reduces total pipeline latency under 10 ms.

2. **Grad-CAM Backend Compatibility**:
   - Native PyTorch autograd graph (`backend="pytorch"`) is required for generating backward-pass Grad-CAM feature heatmaps. ONNX Runtime backend falls back to cached feature activations or PyTorch reference passes for heatmaps.

3. **Supported Breed Scope**:
   - The current model scope is trained specifically on 6 indigenous breeds (*Gir, Ongole, Sahiwal, Jaffarabadi, Murrah, Surti*). Cross-bred exotic breeds (e.g., Holstein-Friesian or Jersey crosses) are out of scope for current class mappings.
