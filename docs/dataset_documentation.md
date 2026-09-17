# Dataset and Data Splitting Documentation

This document describes the indigenous Indian cattle and buffalo dataset structure, perceptual hashing duplicate detection, group-aware 70/15/15 dataset splitting, and manifest format.

---

## 1. Supported Indigenous Breeds Catalog (6 Classes)

The dataset covers 6 major indigenous livestock breeds originating across India:

```
data/raw/dataset/
├── cattle/
│   ├── gir/           (Class 0: Cattle, Gujarat)
│   ├── ongole/        (Class 1: Cattle, Andhra Pradesh)
│   └── sahiwal/       (Class 2: Cattle, Punjab)
└── buffalo/
    ├── jaffarabadi/   (Class 3: Buffalo, Gujarat)
    ├── murrah/        (Class 4: Buffalo, Haryana)
    └── surti/         (Class 5: Buffalo, Gujarat)
```

---

## 2. Perceptual dHash Duplicate Detection

To prevent data contamination and duplicate image inclusion:

1. **dHash Computation**: Compute 64-bit difference hash (dHash) for each image.
2. **Hamming Distance Comparison**: Images with Hamming distance $d \le 2$ or identical filename augmented patterns are flagged as duplicate candidates.
3. **Removal**: Identical and duplicate images are pruned prior to dataset splitting.

---

## 3. Group-Aware 70/15/15 Dataset Splitting

Data leakage occurs if images of the same individual animal or source group appear across training and test sets.

To enforce zero data leakage:
- **Group Key Extraction**: Extract individual animal / farm group identifiers from source metadata or subfolder prefix patterns.
- **Group-Aware Stratified Split**:
  - **Training Set**: 70% of groups per breed class.
  - **Validation Set**: 15% of groups per breed class.
  - **Test Set**: 15% of groups per breed class (strictly unseen).

### Verified Leakage Metric
- **Group Leakage Rate**: **0.00%** (Zero group overlap between train, val, and test splits).

---

## 4. Manifest Schema (`dataset_manifest.json`)

The processed dataset manifest records image metadata and split assignments:

```json
{
  "dataset_version": "dataset_v001",
  "created_at": "2026-08-27T18:00:00Z",
  "summary": {
    "total_images": 1200,
    "train_images": 840,
    "val_images": 180,
    "test_images": 180,
    "group_leakage_count": 0
  },
  "classes": {
    "0": "Gir",
    "1": "Ongole",
    "2": "Sahiwal",
    "3": "Jaffarabadi",
    "4": "Murrah",
    "5": "Surti"
  }
}
```
