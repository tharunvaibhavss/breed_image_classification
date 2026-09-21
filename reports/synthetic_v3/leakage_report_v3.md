# Leakage Verification Report: Model V3 (Real + Synthetic)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Target Splits**: Real Train, Synthetic Train, Real Validation, Real Test (Held-Out)  
**Date**: September 2026  

---

## 1. Cryptographic Collision Analysis (SHA-256)

| Partition Pair | Overlap Count | Total Sample Pool | Audit Status |
| :--- | :---: | :---: | :---: |
| **Real Train ↔ Real Validation** | **0** | 466 samples | **PASSED** |
| **Real Train ↔ Real Test** | **0** | 505 samples | **PASSED** |
| **Real Validation ↔ Real Test** | **0** | 207 samples | **PASSED** |
| **Synthetic Train ↔ Real Test** | **0** | 1,398 samples | **PASSED** |
| **Synthetic Train ↔ Real Validation** | **0** | 1,359 samples | **PASSED** |

---

## 2. Perceptual Near-Duplicate Analysis (pHash / dHash)

| Verification Criterion | Threshold | Overlap Count | Audit Status |
| :--- | :---: | :---: | :---: |
| **Synthetic Train ↔ Real Test Overlap** | Hamming $\le 3$ | **0** | **PASSED** |
| **Synthetic Train ↔ Real Val Overlap** | Hamming $\le 3$ | **0** | **PASSED** |
| **Real Train ↔ Real Test Overlap** | Hamming $\le 3$ | **0** | **PASSED** |

---

## 3. Final Verification Verdict
**DATA LEAKAGE AUDIT: PASSED (ZERO DATA LEAKAGE)**

The held-out unseen test set ($N=123$ real photographs) remains 100% pure and completely isolated from all training data (both real and synthetic). Zero cryptographic or perceptual overlap exists across partitions.
