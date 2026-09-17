# Dataset Audit Report

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Official Taxonomy Source**: ICAR-National Bureau of Animal Genetic Resources (NBAGR) & ICAR-CIRB  
**Audit Date**: 2026-09-17  
**Audit Status**: VERIFIED COMPLETE  

---

## 1. Executive Summary

| Category | Official ICAR-NBAGR Target | Verified in Dataset | Completeness |
| :--- | :--- | :--- | :--- |
| **Cattle Breeds** | 59 | 59 | 100.0% |
| **Buffalo Breeds** | 23 | 23 | 100.0% |
| **Total Breed Classes** | 82 | 82 | 100.0% |
| **Total Verified Clean Images** | - | 486 | - |
| **Cattle Images** | - | 349 | 71.8% |
| **Buffalo Images** | - | 137 | 28.2% |

---

## 2. Dataset Hierarchy & Directory Structure

```
dataset/
├── cleaned/
│   ├── cattle/             # 59 breed subdirectories
│   └── buffalo/            # 23 breed subdirectories
├── expanded_82_breeds/     # Flat standardized repository of 82 classes
├── metadata/
│   ├── breed_details.csv   # ICAR-NBAGR accession, home tract, utility
│   ├── image_metadata.csv  # Dimensions, channels, format, source
│   └── source_manifest.csv # Source attribution and URLs
├── rejected/
│   ├── corrupted/          # Corrupted header / non-image files
│   ├── duplicates/         # Exact and near-duplicate images quarantined
│   └── low_resolution/     # Images below 224x224 px threshold
└── splits/
    ├── train.csv           # 302 images (62.1%)
    ├── validation.csv      # 69 images (14.2%)
    ├── test.csv            # 115 images (23.7%, 100% of 82 classes represented)
    └── split_statistics.csv# Per-breed distribution across splits
```

---

## 3. Class Distribution & Sample Statistics

- **Total Classes**: 82 (59 Cattle, 23 Buffalo)
- **Total Images**: 486
- **Minimum Images per Class**: 1
- **Maximum Images per Class**: 31
- **Average Images per Class**: 5.93
- **Median Images per Class**: 2.00

### Note on Taxonomic Discrepancy / Homonyms
The breed name **Bargur** exists under both species:
1. **Bargur (Cattle)**: Native to Bargur hills, Erode district, Tamil Nadu (Accession: INDIA_CATTLE_1800_BARGUR_03003). Brown/red with white specks.
2. **Bargur (Buffalo)**: Native to western ghats of Erode, Tamil Nadu (Accession: INDIA_BUFFALO_1800_BARGUR_01014).
Both are distinct official breeds cataloged separately in `models/class_names.json` as `cow_bargur` (index 26) and `buffalo_bargur` (index 1).

---

## 4. Image Quality Verification

- **Decodability**: 100% of the 486 active images are fully decodable via PIL and OpenCV (`cv2.imread`).
- **Color Channels**: All images normalized to 3-channel RGB.
- **Minimum Dimensions**: All images meet or exceed 224 x 224 pixels.
- **Quarantined Artifacts**:
  - 1 corrupted header image quarantined to `dataset/rejected/corrupted/`
  - 20 duplicates quarantined to `dataset/rejected/duplicates/`
  - 9 low-resolution images quarantined to `dataset/rejected/low_resolution/`
