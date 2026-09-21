import os
import pandas as pd
from pathlib import Path

# Load split manifest v2
manifest_path = Path('reports/dataset_expansion_v2/split_manifest_v2.csv')
if not manifest_path.exists():
    manifest_path = Path('dataset/metadata_v2/master_manifest_v2.csv')

manifest = pd.read_csv(manifest_path)
print(f"Loaded manifest with {len(manifest)} total image entries.")

# Group by breed
breed_stats = manifest.groupby(['breed_id', 'breed_name', 'species']).agg(
    total_real=('split', 'count'),
    train_real=('split', lambda s: (s == 'train').sum()),
    val_real=('split', lambda s: (s == 'validation').sum()),
    test_real=('split', lambda s: (s == 'test').sum())
).reset_index()

def get_priority(r):
    if r >= 30:
        return 'SUFFICIENT (>=30 REAL)'
    elif r >= 15:
        return 'MEDIUM (15-29 REAL)'
    else:
        return 'HIGH (1-14 REAL)'

def get_synthetic_required(row):
    # User requirement: Minimum 30 images per class
    # Ensure each class reaches at least 30 training images
    train_r = row['train_real']
    return max(0, 30 - train_r)

breed_stats['priority'] = breed_stats['total_real'].apply(get_priority)
breed_stats['synthetic_images_required'] = breed_stats.apply(get_synthetic_required, axis=1)
breed_stats['target_training_count'] = breed_stats['train_real'] + breed_stats['synthetic_images_required']
breed_stats['combined_total'] = breed_stats['total_real'] + breed_stats['synthetic_images_required']

# Sort by real_image_count ascending (scarcity), then breed name
breed_stats = breed_stats.sort_values(by=['total_real', 'breed_name']).reset_index(drop=True)

# Save as CSV
Path('reports').mkdir(exist_ok=True)
breed_stats.to_csv('reports/synthetic_dataset_plan_v3.csv', index=False)

# Markdown output
md = f"""# Synthetic Dataset Augmentation Plan (v3)

**Project Title**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Authoritative Reference**: ICAR-NBAGR & ICAR-CIRB  
**Target Milestone**: Supplementary Training Set Augmentation (V3 - Minimum 30 Images/Class Target)  
**Strict Scientific Methodology**:
1. Synthetic images serve strictly as **Training Augmentation**.
2. **Zero Synthetic Leakage**: Validation and Test sets must strictly remain **100% Real Unseen Photographs**.
3. **Minimum 30 Target Enforcement**: Every single class is supplemented with synthetic training images so that the training set achieves at least 30 images per class (Synthetic Req = max(0, 30 - R_train)). Classes with >= 30 training images receive 0 synthetic samples.

---

## 1. Class Distribution & Scarcity Breakdown

| Priority Category | Real Image Threshold | Number of Breeds | Description / Handling Strategy |
| :--- | :---: | :---: | :--- |
| **HIGH Priority** | **1–14 real images** | **{(breed_stats['total_real'] < 15).sum()}** | Critically under-represented rare breeds requiring substantial augmentation up to 30 training images. |
| **MEDIUM Priority** | **15–29 real images** | **{((breed_stats['total_real'] >= 15) & (breed_stats['total_real'] < 30)).sum()}** | Moderately represented breeds supplemented to reach 30 training images. |
| **SUFFICIENT Priority**| **$\ge 30$ real images** | **{(breed_stats['total_real'] >= 30).sum()}** | Breeds already meeting or exceeding the 30-image threshold. Minimal or no synthetic images required. |

### Global Dataset Metrics
- **Total Registered Breeds**: 82 (59 Cattle, 23 Buffalo)
- **Total Real Images Available**: **{breed_stats['total_real'].sum()}**
  - Real Training Images: **{breed_stats['train_real'].sum()}** (64.9%)
  - Real Validation Images: **{breed_stats['val_real'].sum()}** (14.3%)
  - Real Unseen Test Images: **{breed_stats['test_real'].sum()}** (20.9%)
- **Total Synthetic Images Planned**: **{breed_stats['synthetic_images_required'].sum()}**
- **Projected Total Augmented Training Set**: **{breed_stats['target_training_count'].sum()}** ({breed_stats['train_real'].sum()} Real + {breed_stats['synthetic_images_required'].sum()} Synthetic)
- **Minimum Training Count across all 82 Breeds**: **{breed_stats['target_training_count'].min()}**
- **Real Training Proportion**: **{breed_stats['train_real'].sum() / breed_stats['target_training_count'].sum() * 100:.2f}%**
- **Synthetic Training Proportion**: **{breed_stats['synthetic_images_required'].sum() / breed_stats['target_training_count'].sum() * 100:.2f}%**

---

## 2. Breed-by-Breed Synthetic Requirements Table

| Breed ID | Breed Name | Species | Real Total ($R$) | Real Train | Real Val | Real Test | Synthetic Req. | Target Train Count | Scarcity Priority |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

for _, r in breed_stats.iterrows():
    md += f"| `{r['breed_id']}` | **{r['breed_name']}** | {r['species'].capitalize()} | {r['total_real']} | {r['train_real']} | {r['val_real']} | {r['test_real']} | **{r['synthetic_images_required']}** | **{r['target_training_count']}** | `{r['priority']}` |\n"

md += """
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
"""

with open('reports/synthetic_dataset_plan_v3.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("Saved reports/synthetic_dataset_plan_v3.md and reports/synthetic_dataset_plan_v3.csv")
