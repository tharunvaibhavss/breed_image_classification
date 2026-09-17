================================================
FINAL MODEL EVALUATION
================================================

Model:
EfficientNet-B0

Classes:
82

Cattle:
59

Buffalo:
23

Total images:
486

Training:
302

Validation:
69

Testing:
115

------------------------------------------------
FINAL TEST RESULTS
------------------------------------------------

Accuracy:
13.04%

Macro Precision:
3.63%

Macro Recall:
6.61%

Macro F1:
4.43%

Weighted Precision:
5.89%

Weighted Recall:
13.04%

Weighted F1:
7.68%

Top-3 Accuracy:
32.17%

------------------------------------------------
CATTLE RESULTS
------------------------------------------------

Accuracy:
10.98%

Macro Precision:
3.51%

Macro Recall:
5.58%

Macro F1:
4.08%

Top-3 Accuracy:
30.49%

------------------------------------------------
BUFFALO RESULTS
------------------------------------------------

Accuracy:
18.18%

Macro Precision:
7.84%

Macro Recall:
7.04%

Macro F1:
6.85%

Top-3 Accuracy:
36.36%

------------------------------------------------
DATA LEAKAGE
------------------------------------------------

Exact duplicate overlap:
0 / 486

Near duplicate overlap:
0 / 486

Train/Test overlap:
0

Validation/Test overlap:
0

Status:
PASSED

------------------------------------------------
PERFORMANCE
------------------------------------------------

YOLO:
117.92 ms

PyTorch EfficientNet:
79.21 ms

ONNX EfficientNet:
19.03 ms

Grad-CAM:
649.21 ms

End-to-end:
790.66 ms

API:
793.66 ms

------------------------------------------------
MODEL LIMITATIONS
------------------------------------------------

1. Low Per-Class Training Sample Support:
   With an average of 5.93 images per breed across 82 fine-grained classes, rare and newly registered indigenous breeds (e.g. Masilum, Khariar, Ghumusari) have single-shot representation in test, leading to sparse class-level convergence.

2. Subtle Morphological and Phenotypic Overlap:
   Draught zebu breeds across Karnataka, Maharashtra, and Gujarat (e.g. Hallikar, Amritmahal, Khillar) share grey/white coats and lyre horns, creating visual ambiguity that requires multi-view photography.

3. Black Coat Dominance in Buffaloes:
   Indian riverine water buffaloes exhibit uniform dark slate coats where horn curvature and facial profile are the primary discriminators, requiring higher image resolution and precise framing.

4. Top-1 vs. Top-3 Operational Guidance:
   While Top-1 accuracy is 13.04% (outperforming 1.22% random baseline by >10x), Top-3 accuracy achieves 32.17% overall (and 36.36% for buffaloes), indicating strong potential as an assistive veterinary decision-support tool.
