# Real Dataset Inventory Report

## 1. Overview and Scanning Findings

A full filesystem scan was conducted on `data/raw/dataset/` within the project repository to inventory all physical images.

### Scanning Parameters
- **Target Path**: `data/raw/dataset/`
- **Expected Subdirectories**:
  - `cattle/Gir/`, `cattle/Ongole/`, `cattle/Sahiwal/`
  - `buffalo/Jaffarabadi/`, `buffalo/Murrah/`, `buffalo/Surti/`

---

## 2. Empirical Inventory Metrics

| Category | Metric Count | Details / Notes |
| :--- | :--- | :--- |
| **Total Images Found** | **0** | No physical image files (.jpg, .jpeg, .png, .webp, .bmp) exist in `data/raw/dataset/`. |
| **Cattle Images** | **0** | `data/raw/dataset/cattle/` contains 0 files. |
| **Buffalo Images** | **0** | `data/raw/dataset/buffalo/` contains 0 files. |
| **Valid Images** | **0** | N/A |
| **Corrupted / Unreadable Images** | **0** | N/A |
| **Zero-Byte Files** | **0** | N/A |
| **Exact Duplicate Files** | **0** | N/A |
| **Augmented Files (`aug_*`)** | **0** | N/A |

---

## 3. Class Inventory Breakdown

| Animal Type | Breed Name | Expected Path | Actual Images Found | Status |
| :--- | :--- | :--- | :--- | :--- |
| Cattle | **Gir** | `data/raw/dataset/cattle/Gir` | 0 | **EMPTY** |
| Cattle | **Ongole** | `data/raw/dataset/cattle/Ongole` | 0 | **EMPTY** |
| Cattle | **Sahiwal** | `data/raw/dataset/cattle/Sahiwal` | 0 | **EMPTY** |
| Buffalo | **Jaffarabadi** | `data/raw/dataset/buffalo/Jaffarabadi` | 0 | **EMPTY** |
| Buffalo | **Murrah** | `data/raw/dataset/buffalo/Murrah` | 0 | **EMPTY** |
| Buffalo | **Surti** | `data/raw/dataset/buffalo/Surti` | 0 | **EMPTY** |

---

## 4. Summary & Findings

The raw dataset directory `data/raw/dataset/` currently contains zero physical image files. Unit and integration test suites in the repository rely on synthetic image arrays and mock data fixtures.

Per the audit rules, metrics for real dataset images cannot be computed without physical images on disk, and results will not be fabricated.
