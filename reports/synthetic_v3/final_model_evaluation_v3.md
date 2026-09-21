# Final Model Evaluation: Model V3 (Real + Synthetic Augmentation)

================================================
FINAL MODEL EVALUATION V3
================================================

Model:
EfficientNet-B0

Classes:
82

Cattle:
59

Buffalo:
23

Total Training Images:
2461 (382 Real + 2079 Synthetic)

Validation Images:
84 (100% Real Photographs)

Testing Images:
123 (100% Real Photographs, Unseen)

------------------------------------------------
TEST RESULTS (UNSEEN REAL TEST SET)
------------------------------------------------

Accuracy:
36.59%

Macro Precision:
20.34%

Macro Recall:
24.37%

Macro F1:
21.41%

Weighted Precision:
28.54%

Weighted Recall:
36.59%

Weighted F1:
30.88%

Top-1 Accuracy:
36.59%

Top-3 Accuracy:
50.41%

Unique Predicted Classes:
40 / 82

------------------------------------------------
CATTLE RESULTS
------------------------------------------------

Accuracy:
35.56%

Macro Precision:
17.28%

Macro Recall:
19.33%

Macro F1:
17.62%

Top-1 Accuracy:
35.56%

Top-3 Accuracy:
48.89%

------------------------------------------------
BUFFALO RESULTS
------------------------------------------------

Accuracy:
39.39%

Macro Precision:
26.15%

Macro Recall:
25.06%

Macro F1:
24.37%

Top-1 Accuracy:
39.39%

Top-3 Accuracy:
54.55%

------------------------------------------------
DATA LEAKAGE & INTEGRITY
------------------------------------------------

Exact duplicate overlap:
0 / 123

Near-duplicate overlap:
0 / 123

Status:
PASSED (Zero Leakage)
