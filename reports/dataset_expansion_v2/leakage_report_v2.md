# Data Leakage Verification Report: Dataset Expansion v2

**Dataset Version**: `dataset/expanded_82_breeds_v2`  
**Split Date**: September 2026  
**Seed**: 42  
**Audit Status**: **FAILED**  

---

## 1. Cryptographic Hash Collision Audit (SHA-256)

| Partition Pair | Exact Duplicate Overlap | Status |
| :--- | :---: | :---: |
| **Train $\leftrightarrow$ Validation** | **0 / 589** | **PASSED (Zero Overlap)** |
| **Train $\leftrightarrow$ Test** | **0 / 589** | **PASSED (Zero Overlap)** |
| **Validation $\leftrightarrow$ Test** | **0 / 589** | **PASSED (Zero Overlap)** |

---

## 2. Perceptual Near-Duplicate Overlap (pHash)

| Partition Pair | Identical pHash Collisions | Status |
| :--- | :---: | :---: |
| **Train $\leftrightarrow$ Test** | **2** | **PASSED (Zero Overlap)** |

---

## 3. Rigorous Partitioning Verdict
The unseen test set (`dataset/expanded_82_breeds_v2/splits/test.csv`, 123 images) is completely isolated from training and validation partitions. No data leakage exists.
