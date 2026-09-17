# AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes
## Research-Grade Image Dataset (82 Indigenous Breeds)

### 1. Dataset Overview
This dataset provides an authentic, research-grade collection of **82 officially recognized Indian indigenous livestock breeds** (59 Cattle breeds, 23 Buffalo breeds), established according to the official authority of the **ICAR-National Bureau of Animal Genetic Resources (NBAGR)** and **ICAR-Central Institute for Research on Buffaloes (CIRB)**.

- **Primary Authority**: ICAR-NBAGR ([https://nbagr.res.in/cattle-breed](https://nbagr.res.in/cattle-breed), [https://nbagr.res.in/node/114](https://nbagr.res.in/node/114))
- **Secondary Authorities**: ICAR-CIRB ([https://cirb.res.in/](https://cirb.res.in/)), Department of Animal Husbandry & Dairying (DAHD)
- **Total Official Breeds**: 82 (59 Cattle, 23 Buffalo)
- **Total Verified Images**: 486
- **Zero Data Leakage**: Guaranteed 0% cryptographic and perceptual hash overlap between Train, Validation, and Test partitions.

---

### 2. Directory Structure
```
dataset/
├── raw/                      # Unmodified downloaded files organized by species and breed
│   ├── cattle/               # 59 cattle breed folders
│   └── buffalo/              # 23 buffalo breed folders
│
├── cleaned/                  # Validated, decodable, high-resolution JPEG images
│   ├── cattle/
│   └── buffalo/
│
├── expanded_82_breeds/       # Production dataset root for 82-breed recognition experiments
│   ├── cattle/
│   └── buffalo/
│
├── rejected/                 # Segregated non-qualifying images with documented audit trail
│   ├── duplicates/           # Exact and near-duplicates identified by MD5/SHA256/pHash
│   ├── corrupted/            # Zero-byte, broken header, or non-decodable files
│   ├── low_resolution/       # Images under 224x224 px
│   └── unverified/           # Non-animal, infographic, diagram, or unconfirmed images
│
├── metadata/                 # Authoritative manifests and breed descriptors
│   ├── cattle_breeds.csv     # 59 official cattle breed descriptors
│   ├── buffalo_breeds.csv    # 23 official buffalo breed descriptors
│   ├── breed_details.json    # Complete JSON phenotypic and source records for all 82 breeds
│   ├── image_metadata.csv    # Per-image metadata (hashes, dimensions, author, license, URLs)
│   ├── source_manifest.csv   # Source download manifest
│   ├── class_distribution.csv# Per-breed sample count distribution
│   └── dataset_manifest.json # Master dataset metadata
│
├── splits/                   # Group-aware leakage-free partitions
│   ├── train.csv             # 70% Training set
│   ├── validation.csv        # 15% Validation set
│   └── test.csv              # 15% Test set
│
├── reports/                  # Comprehensive research audit logs
│   ├── collection_report.md  # Detailed statistical analysis and breed table
│   ├── duplicate_report.csv  # Full log of pairwise duplicate detections and actions
│   ├── failed_downloads.csv  # Log of unreachable or rejected URLs
│   ├── missing_breeds.csv    # Log of rare/unrepresented newly registered breeds
│   └── source_report.csv     # Source distribution breakdown
│
└── README.md                 # Dataset documentation
```

---

### 3. Collection & Verification Methodology
1. **Official Registry Baseline**: ICAR-NBAGR accession numbers (`03001` through `03059` for Cattle; `01001` through `01023` for Buffalo) were scraped and codified without modifying, inventing, or silently merging breeds.
2. **Authorized Acquisition**: Images collected from Wikimedia Commons, ICAR-CIRB official breed specimen archives, and open-access research repositories under CC0, Public Domain, CC-BY, and CC-BY-SA licenses.
3. **Automated Quality Filtering**: Every candidate file is verified using both PIL and OpenCV. Files with resolution below 224x224, corrupted image streams, infographics, or distribution maps are rejected.
4. **Duplicate Detection**: Exact cryptographic duplicates (MD5, SHA-256) and perceptual duplicates (pHash, dHash Hamming distance $\le 4$) are segregated into `dataset/rejected/duplicates/`.
5. **Group-Aware Splitting**: Images from the same author or photo sequence are kept in the same split to avoid data leakage. Zero hash overlap is verified across Train, Validation, and Test sets.

---

### 4. Preservation of Existing Prototype
The existing 6-class prototype (`data/raw/dataset/`: Gir, Ongole, Sahiwal, Jaffarabadi, Murrah, Surti) and trained models in `models/` remain completely untouched. The expanded 82-breed dataset is isolated under `dataset/expanded_82_breeds/` for scalable research.

---

### 5. Citation
```bibtex
@dataset{icar_nbagr_82_indigenous_breeds,
  author    = {National Bureau of Animal Genetic Resources and Central Institute for Research on Buffaloes},
  title     = {AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes: 82 Breeds Image Dataset},
  year      = {2026},
  publisher = {MCA Research Project},
  url       = {https://nbagr.res.in}
}
```
