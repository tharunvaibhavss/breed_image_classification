# Dataset Card: ICAR-NBAGR 82 Indian Cattle & Buffalo Breeds Dataset

**Dataset Title**: Comprehensive 82-Class Indigenous Indian Livestock Breed Image Dataset  
**Authority Reference**: ICAR-National Bureau of Animal Genetic Resources (NBAGR) & ICAR-CIRB  
**Collection Date**: September 2026  
**License**: Government Open Access / Academic Research Use Only  

---

## 1. Dataset Overview & Scope
- **Total Breeds Cataloged**: 82 Breeds (59 Cattle, 23 Buffalo)
- **Total Verified Images**: 486 RGB Images
- **Cattle Images**: 349 Images across 59 Breeds
- **Buffalo Images**: 137 Images across 23 Breeds
- **Format**: High-resolution RGB JPEG images standardized with 224x224 minimum dimensions.

---

## 2. Data Collection & Verification Pipeline
1. **Authoritative Taxonomy**: Official breed registries, accession numbers, and home tract descriptions obtained directly from ICAR-NBAGR and ICAR-CIRB portals.
2. **Quality Filtering**:
   - PIL and OpenCV decodability validation (100% pass rate).
   - Zero-byte and corrupted header isolation (`dataset/rejected/corrupted/`).
   - Low-resolution rejection (< 224x224 px) into `dataset/rejected/low_resolution/`.
3. **Deduplication**:
   - Cryptographic SHA-256 and MD5 bitwise deduplication.
   - Perceptual hashing (`pHash`, 64-bit DCT, Hamming distance threshold $\le 4$) to eliminate crops, resizes, and visually identical downloads.
   - All 20 duplicate downloads segregated into `dataset/rejected/duplicates/`.

---

## 3. Partitioning & Data Leakage Prevention
- **Partition Scheme**: 70% Training (302 images) / 15% Validation (69 images) / 15% Testing (115 images).
- **Class Representation**: 100% of the 82 breeds have verified samples in the unseen test dataset (`dataset/splits/test.csv`).
- **Data Leakage Status**: **PASSED** (0 exact hash matches and 0 perceptual near-duplicate overlaps across splits).

---

## 4. Known Imbalance & Research Limitations
- **Long-tail distribution**: Samples per breed range from 1 to 31 (Mean: 5.93, Median: 2.00).
- Highly popular breeds (Gir: 31, Ongole: 29, Kangayam: 17, Murrah: 16) have solid representation, whereas newly cataloged / geographically remote breeds (e.g. Masilum, Khariar, Ghumusari) have limited public imagery.
- Research report explicitly documents support counts for all classes without synthetic oversampling of the test set.
