# Dataset Quality & Ingestion Report: Dataset Expansion v2

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Dataset Root**: `dataset/expanded_82_breeds_v2/`  
**Evaluation Target**: 82 ICAR-NBAGR Registered Indigenous Breeds (59 Cattle, 23 Buffalo)  
**Date**: September 2026  

---

## 1. Quality Control Summary Statistics

| Metric | Measured Value |
| :--- | :--- |
| **Total Valid Images Ingested** | **589** |
| **Total Target Classes** | **82** (59 Cattle, 23 Buffalo) |
| **Exact Hash Duplicates Removed** | **5** |
| **Perceptual Near-Duplicates Removed (pHash $\le 3$)** | **0** |
| **Corrupted / Invalid Files Quarantined** | **0** |
| **Minimum Class Count** | **1** |
| **Maximum Class Count** | **43** |
| **Mean Images / Class** | **7.18** |
| **Median Images / Class** | **2.00** |
| **Standard Deviation** | **8.96** |

---

## 2. Species-Specific Representation

| Species | Breeds Count | Total Valid Images | Average Images / Breed |
| :--- | :---: | :---: | :---: |
| **Cattle (*Bos indicus*)** | 59 | 449 | 7.61 |
| **Buffalo (*Bubalus bubalis*)** | 23 | 140 | 6.09 |
| **Total** | **82** | **589** | **7.18** |

---

## 3. Data Integrity & Verification Verdict
- All images verified to be 3-channel RGB without truncation or header corruption.
- Exact and near-duplicate filtering strictly executed prior to partition splitting to guarantee zero leakage.
