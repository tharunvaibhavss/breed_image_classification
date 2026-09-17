# Data Leakage Verification Report

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Audit Date**: 2026-09-17 14:04:31 UTC  
**Random Seed**: 42 (Deterministic Stratified Partitioning)  

---

## 1. Summary of Audit

| Metric | Result |
|---|---|
| **Total images checked** | 486 |
| **Exact duplicates detected & quarantined** | 13 |
| **Near duplicates detected & quarantined** | 7 |
| **Train/Test exact overlap** | 0 |
| **Train/Validation exact overlap** | 0 |
| **Validation/Test exact overlap** | 0 |
| **Train/Test perceptual overlap (pHash distance ≤ 4)** | 0 |
| **Train/Validation perceptual overlap (pHash distance ≤ 4)** | 0 |
| **Validation/Test perceptual overlap (pHash distance ≤ 4)** | 0 |
| **Potential leakage cases** | 0 |
| **Final leakage status** | **PASSED** |

---

## 2. Partition Breakdown

- **Total Classes**: 82 (59 Cattle, 23 Buffalo)
- **Train Set**: 302 images (62.1%)
- **Validation Set**: 69 images (14.2%)
- **Test Set**: 115 images (23.7%)
- **Classes Represented in Test Set**: 82 / 82 (100% of classes represented)

---

## 3. Leakage Prevention Protocol

1. **Cryptographic Deduplication**: Both MD5 and SHA-256 hashes were computed for every image in the dataset. Any identical files or multi-download duplicates were quarantined to `dataset/rejected/duplicates/`.
2. **Perceptual Deduplication**: Perceptual hash (`pHash`) and difference hash (`dHash`) with a strict Hamming distance threshold of $\le 4$ were evaluated across all image pairs to identify and eliminate resized, recompressed, or near-identical multi-shot photos.
3. **Group-Aware Splitting**: Photos originating from the same source or photographer were grouped and partitioned into single splits to prevent intra-sequence leakage.
4. **Zero Overlap Guarantee**: Post-split cross-set verification confirms 0 exact matches and 0 perceptual overlaps between the training, validation, and unseen test partitions.

**Audit Status**: **PASSED** — Clean, leakage-free dataset ready for deep learning model training.
