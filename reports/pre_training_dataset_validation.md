# Pre-Training Dataset Validation Report

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Authority**: ICAR-NBAGR & ICAR-CIRB  
**Validation Date**: 2026-09-17 14:01:47 UTC  

---

## 1. Official Registry Integrity
- **Registered Cattle Breeds**: 59 / 59
- **Registered Buffalo Breeds**: 23 / 23
- **Total Official Breeds**: 82 / 82
- **Official Authority URL (Cattle)**: https://nbagr.res.in/cattle-breed
- **Official Authority URL (Buffalo)**: https://nbagr.res.in/node/114

---

## 2. Image Decodability & Quality Checks
- **Total Verified Clean Images**: 509
- **Zero-Byte / Broken Header Images Rejected**: 1
- **Sub-Resolution Images (< 224x224 px) Rejected**: 1
- **Decodability Guarantee**: 100% of verified images successfully parsed and verified by both **PIL** and **OpenCV (cv2.imread)**.
- **Color Format**: Standardized RGB JPEG format with preserved original metadata.

---

## 3. Representation Across Classes
- **Total Breeds in Dataset**: 82
- **Breeds with Verified Images**: 82
- **Underrepresented / Rare Breeds**: 0

---

## 4. Quality Assurance Summary
All valid images are housed in `dataset/cleaned/` and `dataset/expanded_82_breeds/` with strict per-image metadata recorded in `dataset/metadata/image_metadata.csv`.
