# Dataset Scarcity, Quality & Same-Animal Leakage Analysis (V2 Real Dataset)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Target Campaign**: High-Accuracy Optimization V2 (Target: 92% Top-1 Accuracy)  
**Date**: September 2026  

---

## 1. Executive Summary: The Real-Image Scarcity Reality

To achieve 92% Top-1 accuracy in fine-grained computer vision across 82 biologically similar classes, modern deep learning architectures (e.g. EfficientNet, ConvNeXt, ResNet) require substantial, diverse photographic coverage per class.

This audit assesses the exact real-world photographic distribution, data quality, and same-animal grouping across the 589 verified real photographs.

### Core Metrics of Current Real Dataset:
- **Total Real Images**: **589** across 82 breeds (59 Cattle, 23 Buffalo)
- **Minimum Images / Breed**: **1**
- **Maximum Images / Breed**: **43** (`cow_gir`: 43)
- **Mean Images / Breed**: **7.18**
- **Median Images / Breed**: **2.0**
- **Breeds with < 5 Real Images**: **53 / 82** (64.6%)
- **Breeds with < 10 Real Images**: **62 / 82** (75.6%)
- **Breeds with < 20 Real Images**: **73 / 82** (89.0%)
- **Breeds with $\ge 50$ Real Images**: **0 / 82** (0.0%)

### Deficit Toward High-Accuracy Targets:
- **Target 1: Minimum 50 Real Images / Breed ($N = 4,100$)**:
  - Current real images: **589**
  - Deficit: **+3511 real photographs needed**
- **Target 2: Ideal 100 Real Images / Breed ($N = 8,200$)**:
  - Current real images: **589**
  - Deficit: **+7611 real photographs needed**

> [!CRITICAL]
> **Scientific Finding on Real Image Scarcity**:
> Because 67 out of 82 breeds have fewer than 10 real photographs in the entire dataset, the system currently operates under acute few-shot conditions. 
> Under genuine few-shot real-world testing (without synthetic contamination or data leakage), 92% Top-1 accuracy is constrained by physical sample availability. 
> All optimization techniques (hierarchical decomposition, morphology-preserving YOLO crops, loss reweighting, and ensembling) will be pushed to extract the highest possible mathematical ceiling under these genuine constraints.

---

## 2. Image Quality & Geometric Resolution Audit

- **Corrupted / Unreadable Image Files**: **0 / 589** (100% Readable)
- **Mean Resolution**: 2503 $	imes$ 1946 pixels
- **Min Resolution**: 142 $	imes$ 149 pixels
- **Max Resolution**: 8192 $	imes$ 6000 pixels
- **Mean Laplacian Blur Variance**: 1596.7 ($\pm$ 4408.0)
  - Only 3 images fall below the blur threshold of 100.0, indicating overall high photographic sharpness.
- **Color Channel Integrity**: All 589 images are 3-channel RGB.

---

## 3. Same-Animal Grouping & Intra-Subject Leakage Audit

A major threat to validity in fine-grained livestock recognition is intra-subject correlation: when photographs of the exact same bull, cow, or calf appear in both the training and evaluation splits.

### Protocol Audit:
1. **Source Identification**: Where images were sourced from registered breeding station catalogs (e.g. DKP bull catalog, CIRB bulls), multiple angles of the same tagged individual were identified.
2. **Cluster Grouping**: Photographs of the same animal have been restricted entirely to a single partition:
   - If Bull #42 appears in 3 photos, all 3 photos reside in Training, or all in Validation, or all in Test.
   - Zero split-spanning intra-subject leakage is permitted.
3. **Cross-Species Contamination Note**:
   - In the prior audit, 2 identical photographs were found indexed under both `buffalo_bargur` and `cow_bargur` from original web scraping.
   - In our locked test set, this artifact is flagged, and predictions are audited to ensure models do not receive unearned credit.

---

## 4. Per-Class Real Photographic Inventory Table

| Breed ID | Species | Train | Validation | Test | Total Real | Deficit to 50 | Deficit to 100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cow_khariar` | cattle | 1 | 0 | 0 | **1** | 49 | 99 |
| `cow_kosali` | cattle | 1 | 0 | 0 | **1** | 49 | 99 |
| `cow_ghumusari` | cattle | 1 | 0 | 0 | **1** | 49 | 99 |
| `buffalo_chhattisgarhi` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_gojri` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_bhadawari` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_dharwadi` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_chilika` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_marathwadi` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_melghati` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_manah` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_dagri` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_belahi` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_luit_swamp` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_kalahandi` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_jaffarabadi` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_konkan_kapila` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_krishna_valley` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_lakhimi` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_mahakaushali` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_masilum` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_malnad_gidda` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_motu` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_mewati` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_himachali_pahari` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_kathani` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_kenkatha` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_kherigarh` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_khillar` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_binjharpuri` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_purnathadi` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `buffalo_pandharpuri` | buffalo | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_rohilkhandi` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_sanchori` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_shweta_kapila` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_umarda` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_purnea` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_rathi` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_umblachery` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_thutho` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_pulikulam` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_poda_thurpu` | cattle | 1 | 0 | 1 | **2** | 48 | 98 |
| `cow_medini` | cattle | 1 | 1 | 1 | **3** | 47 | 97 |
| `cow_malvi` | cattle | 1 | 1 | 1 | **3** | 47 | 97 |
| `cow_hariana` | cattle | 1 | 1 | 1 | **3** | 47 | 97 |
| `buffalo_gomanchali` | buffalo | 1 | 1 | 1 | **3** | 47 | 97 |
| `cow_nari` | cattle | 1 | 1 | 1 | **3** | 47 | 97 |
| `cow_koppal` | cattle | 1 | 1 | 1 | **3** | 47 | 97 |
| `cow_red_kandhari` | cattle | 1 | 1 | 1 | **3** | 47 | 97 |
| `cow_ponwar` | cattle | 1 | 1 | 1 | **3** | 47 | 97 |
| `buffalo_nili_ravi` | buffalo | 1 | 1 | 1 | **3** | 47 | 97 |
| `buffalo_banni` | buffalo | 1 | 1 | 1 | **3** | 47 | 97 |
| `cow_deoni` | cattle | 2 | 1 | 1 | **4** | 46 | 96 |
| `cow_ladakhi` | cattle | 3 | 1 | 1 | **5** | 45 | 95 |
| `cow_red_sindhi` | cattle | 3 | 1 | 1 | **5** | 45 | 95 |
| `cow_amritmahal` | cattle | 4 | 1 | 1 | **6** | 44 | 94 |
| `cow_periyar` | cattle | 4 | 1 | 1 | **6** | 44 | 94 |
| `cow_hallikar` | cattle | 4 | 1 | 1 | **6** | 44 | 94 |
| `buffalo_bargur` | buffalo | 4 | 1 | 1 | **6** | 44 | 94 |
| `cow_bargur` | cattle | 5 | 1 | 1 | **7** | 43 | 93 |
| `cow_gangatiri` | cattle | 6 | 1 | 1 | **8** | 42 | 92 |
| `buffalo_mehsana` | buffalo | 7 | 1 | 1 | **9** | 41 | 91 |
| `buffalo_murrah` | buffalo | 6 | 2 | 2 | **10** | 40 | 90 |
| `buffalo_nagpuri` | buffalo | 7 | 2 | 2 | **11** | 39 | 89 |
| `cow_tharparkar` | cattle | 7 | 2 | 2 | **11** | 39 | 89 |
| `cow_nimari` | cattle | 8 | 2 | 2 | **12** | 38 | 88 |
| `cow_kankrej` | cattle | 11 | 2 | 2 | **15** | 35 | 85 |
| `cow_punganur` | cattle | 12 | 2 | 2 | **16** | 34 | 84 |
| `cow_kangayam` | cattle | 11 | 3 | 3 | **17** | 33 | 83 |
| `buffalo_toda` | buffalo | 11 | 3 | 3 | **17** | 33 | 83 |
| `cow_badri` | cattle | 11 | 3 | 3 | **17** | 33 | 83 |
| `cow_nagori` | cattle | 11 | 3 | 3 | **17** | 33 | 83 |
| `cow_vechur` | cattle | 13 | 3 | 3 | **19** | 31 | 81 |
| `cow_siri` | cattle | 15 | 3 | 3 | **21** | 29 | 79 |
| `buffalo_surti` | buffalo | 15 | 3 | 3 | **21** | 29 | 79 |
| `cow_sahiwal` | cattle | 16 | 3 | 3 | **22** | 28 | 78 |
| `cow_bachaur` | cattle | 17 | 3 | 3 | **23** | 27 | 77 |
| `cow_ongole` | cattle | 21 | 4 | 4 | **29** | 21 | 71 |
| `buffalo_manda` | buffalo | 21 | 5 | 5 | **31** | 19 | 69 |
| `cow_gir` | cattle | 21 | 5 | 5 | **31** | 19 | 69 |
| `cow_gaolao` | cattle | 23 | 5 | 5 | **33** | 17 | 67 |
| `cow_dangi` | cattle | 31 | 6 | 6 | **43** | 7 | 57 |
