# Final Model Evaluation: Expanded 82-Breed Model v2

================================================
FINAL MODEL EVALUATION v2
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
589

Training:
382

Validation:
84

Testing:
123

------------------------------------------------
TEST RESULTS (UNSEEN TEST SET)
------------------------------------------------

Accuracy:
33.33%

Macro Precision:
17.71%

Macro Recall:
19.30%

Macro F1:
16.94%

Weighted Precision:
28.22%

Weighted Recall:
33.33%

Weighted F1:
27.59%

Top-3 Accuracy:
50.41%

Unique Predicted Classes:
33 / 82

------------------------------------------------
CATTLE RESULTS
------------------------------------------------

Accuracy:
34.44%

Macro Precision:
15.43%

Macro Recall:
17.19%

Macro F1:
15.31%

Top-3 Accuracy:
50.00%

------------------------------------------------
BUFFALO RESULTS
------------------------------------------------

Accuracy:
30.30%

Macro Precision:
18.74%

Macro Recall:
16.44%

Macro F1:
16.09%

Top-3 Accuracy:
51.52%

------------------------------------------------
EXPERIMENT C (CLASSES >= 20 IMAGES)
------------------------------------------------

Classes Meeting Threshold:
9

Accuracy:
59.46%

Top-3 Accuracy:
75.68%

Macro F1:
42.26%

------------------------------------------------
DATA LEAKAGE
------------------------------------------------

Exact duplicate overlap:
0 / 589

Near duplicate overlap:
0 / 589

Train/Test overlap:
0

Validation/Test overlap:
0

Status:
PASSED
