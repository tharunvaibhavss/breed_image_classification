# Synthetic Dataset Augmentation Plan (v3)

**Project Title**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Authoritative Reference**: ICAR-NBAGR & ICAR-CIRB  
**Target Milestone**: Supplementary Training Set Augmentation (V3 - Minimum 20 Images/Class Target)  
**Strict Scientific Methodology**:
1. Synthetic images serve strictly as **Training Augmentation**.
2. **Zero Synthetic Leakage**: Validation and Test sets must strictly remain **100% Real Unseen Photographs**.
3. **Minimum 20 Target Enforcement**: Every single class is supplemented with synthetic training images so that the training set achieves at least 20 images per class (Synthetic Req = max(0, 20 - R_train)). Classes with >= 20 training images receive 0 synthetic samples.

---

## 1. Class Distribution & Scarcity Breakdown

| Priority Category | Real Image Threshold | Number of Breeds | Description / Handling Strategy |
| :--- | :---: | :---: | :--- |
| **HIGH Priority** | **1–9 real images** | **62** | Critically under-represented rare breeds requiring substantial augmentation up to 20 training images. |
| **MEDIUM Priority** | **10–19 real images** | **11** | Moderately represented breeds supplemented to reach 20 training images. |
| **SUFFICIENT Priority**| **$\ge 20$ real images** | **9** | Breeds already meeting or exceeding the 20-image threshold. No synthetic images required. |

### Global Dataset Metrics
- **Total Registered Breeds**: 82 (59 Cattle, 23 Buffalo)
- **Total Real Images Available**: **589**
  - Real Training Images: **382** (64.9%)
  - Real Validation Images: **84** (14.3%)
  - Real Unseen Test Images: **123** (20.9%)
- **Total Synthetic Images Planned**: **1275**
- **Projected Total Augmented Training Set**: **1657** (382 Real + 1275 Synthetic)
- **Minimum Training Count across all 82 Breeds**: **20**
- **Real Training Proportion**: **23.05%**
- **Synthetic Training Proportion**: **76.95%**

---

## 2. Breed-by-Breed Synthetic Requirements Table

| Breed ID | Breed Name | Species | Real Total ($R$) | Real Train | Real Val | Real Test | Synthetic Req. | Target Train Count | Scarcity Priority |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cow_ghumusari` | **Ghumusari** | Cattle | 1 | 1 | 0 | 0 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_khariar` | **Khariar** | Cattle | 1 | 1 | 0 | 0 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_kosali` | **Kosali** | Cattle | 1 | 1 | 0 | 0 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_belahi` | **Belahi** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_bhadawari` | **Bhadawari** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_binjharpuri` | **Binjharpuri** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_chhattisgarhi` | **Chhattisgarhi** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_chilika` | **Chilika** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_dagri` | **Dagri** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_dharwadi` | **Dharwadi** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_gojri` | **Gojri** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_himachali_pahari` | **Himachali Pahari** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_jaffarabadi` | **Jaffarabadi** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_kalahandi` | **Kalahandi** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_kathani` | **Kathani** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_kenkatha` | **Kenkatha** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_kherigarh` | **Kherigarh** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_khillar` | **Khillar** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_konkan_kapila` | **Konkan Kapila** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_krishna_valley` | **Krishna Valley** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_lakhimi` | **Lakhimi** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_luit_swamp` | **Luit (Swamp)** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_mahakaushali` | **Mahakaushali** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_malnad_gidda` | **Malnad Gidda** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_manah` | **Manah** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_marathwadi` | **Marathwadi** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_masilum` | **Masilum** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_melghati` | **Melghati** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_mewati` | **Mewati** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_motu` | **Motu** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_pandharpuri` | **Pandharpuri** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_poda_thurpu` | **Poda Thurpu** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_pulikulam` | **Pulikulam** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_purnathadi` | **Purnathadi** | Buffalo | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_purnea` | **Purnea** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_rathi` | **Rathi** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_rohilkhandi` | **Rohilkhandi** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_sanchori` | **Sanchori** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_shweta_kapila` | **Shweta Kapila** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_thutho` | **Thutho** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_umarda` | **Umarda** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_umblachery` | **Umblachery** | Cattle | 2 | 1 | 0 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_banni` | **Banni** | Buffalo | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_gomanchali` | **Gomanchali** | Buffalo | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_hariana` | **Hariana** | Cattle | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_koppal` | **Koppal** | Cattle | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_malvi` | **Malvi** | Cattle | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_medini` | **Medini** | Cattle | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_nari` | **Nari** | Cattle | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_nili_ravi` | **Nili Ravi** | Buffalo | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_ponwar` | **Ponwar** | Cattle | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_red_kandhari` | **Red Kandhari** | Cattle | 3 | 1 | 1 | 1 | **19** | **20** | `HIGH (1-9 REAL)` |
| `cow_deoni` | **Deoni** | Cattle | 4 | 2 | 1 | 1 | **18** | **20** | `HIGH (1-9 REAL)` |
| `cow_ladakhi` | **Ladakhi** | Cattle | 5 | 3 | 1 | 1 | **17** | **20** | `HIGH (1-9 REAL)` |
| `cow_red_sindhi` | **Red Sindhi** | Cattle | 5 | 3 | 1 | 1 | **17** | **20** | `HIGH (1-9 REAL)` |
| `cow_amritmahal` | **Amritmahal** | Cattle | 6 | 4 | 1 | 1 | **16** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_bargur` | **Bargur** | Buffalo | 6 | 4 | 1 | 1 | **16** | **20** | `HIGH (1-9 REAL)` |
| `cow_hallikar` | **Hallikar** | Cattle | 6 | 4 | 1 | 1 | **16** | **20** | `HIGH (1-9 REAL)` |
| `cow_periyar` | **Periyar** | Cattle | 6 | 4 | 1 | 1 | **16** | **20** | `HIGH (1-9 REAL)` |
| `cow_bargur` | **Bargur** | Cattle | 7 | 5 | 1 | 1 | **15** | **20** | `HIGH (1-9 REAL)` |
| `cow_gangatiri` | **Gangatiri** | Cattle | 8 | 6 | 1 | 1 | **14** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_mehsana` | **Mehsana** | Buffalo | 9 | 7 | 1 | 1 | **13** | **20** | `HIGH (1-9 REAL)` |
| `buffalo_murrah` | **Murrah** | Buffalo | 10 | 6 | 2 | 2 | **14** | **20** | `MEDIUM (10-19 REAL)` |
| `buffalo_nagpuri` | **Nagpuri** | Buffalo | 11 | 7 | 2 | 2 | **13** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_tharparkar` | **Tharparkar** | Cattle | 11 | 7 | 2 | 2 | **13** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_nimari` | **Nimari** | Cattle | 12 | 8 | 2 | 2 | **12** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_kankrej` | **Kankrej** | Cattle | 15 | 11 | 2 | 2 | **9** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_punganur` | **Punganur** | Cattle | 16 | 12 | 2 | 2 | **8** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_badri` | **Badri** | Cattle | 17 | 11 | 3 | 3 | **9** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_kangayam` | **Kangayam** | Cattle | 17 | 11 | 3 | 3 | **9** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_nagori` | **Nagori** | Cattle | 17 | 11 | 3 | 3 | **9** | **20** | `MEDIUM (10-19 REAL)` |
| `buffalo_toda` | **Toda** | Buffalo | 17 | 11 | 3 | 3 | **9** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_vechur` | **Vechur** | Cattle | 19 | 13 | 3 | 3 | **7** | **20** | `MEDIUM (10-19 REAL)` |
| `cow_siri` | **Siri** | Cattle | 21 | 15 | 3 | 3 | **5** | **20** | `SUFFICIENT (>=20 REAL)` |
| `buffalo_surti` | **Surti** | Buffalo | 21 | 15 | 3 | 3 | **5** | **20** | `SUFFICIENT (>=20 REAL)` |
| `cow_sahiwal` | **Sahiwal** | Cattle | 22 | 16 | 3 | 3 | **4** | **20** | `SUFFICIENT (>=20 REAL)` |
| `cow_bachaur` | **Bachaur** | Cattle | 23 | 17 | 3 | 3 | **3** | **20** | `SUFFICIENT (>=20 REAL)` |
| `cow_ongole` | **Ongole** | Cattle | 29 | 21 | 4 | 4 | **0** | **21** | `SUFFICIENT (>=20 REAL)` |
| `cow_gir` | **Gir** | Cattle | 31 | 21 | 5 | 5 | **0** | **21** | `SUFFICIENT (>=20 REAL)` |
| `buffalo_manda` | **Manda** | Buffalo | 31 | 21 | 5 | 5 | **0** | **21** | `SUFFICIENT (>=20 REAL)` |
| `cow_gaolao` | **Gaolao** | Cattle | 33 | 23 | 5 | 5 | **0** | **23** | `SUFFICIENT (>=20 REAL)` |
| `cow_dangi` | **Dangi** | Cattle | 43 | 31 | 6 | 6 | **0** | **31** | `SUFFICIENT (>=20 REAL)` |

---

## 3. Strict Confirmation of Baseline & V2 Preservation

1. **Existing Datasets Untouched**:
   - `dataset/expanded_82_breeds/` (V1 Baseline: 486 images) remains preserved.
   - `dataset/expanded_82_breeds_v2/` (V2 Real: 589 images) remains preserved.
2. **Existing Models & Reports Untouched**:
   - `models/efficientnet_b0_82_breeds_best.pth` (V1) is preserved.
   - `models/expanded_82_breeds_v2/best_model_v2.pth` (V2) is preserved.
   - `reports/dataset_expansion_v2/` (V2 evaluation suite) is preserved.
3. **Isolated New Version Directory**:
   - Synthetic images will be strictly placed into: `dataset/synthetic_82_breeds_v3/synthetic/`
   - Model weights will be strictly placed into: `models/synthetic_82_breeds_v3/`
   - Evaluation reports will be strictly placed into: `reports/synthetic_v3/`
