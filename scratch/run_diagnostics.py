import os
import json
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
from collections import Counter

print("="*60)
print("RUNNING 82-BREED COMPREHENSIVE DIAGNOSTICS")
print("="*60)

# -------------------------------------------------------------
# 1. DATASET DISTRIBUTION ANALYSIS
# -------------------------------------------------------------
train_df = pd.read_csv('dataset/splits/train.csv')
val_df = pd.read_csv('dataset/splits/validation.csv')
test_df = pd.read_csv('dataset/splits/test.csv')

with open('models/class_names.json', 'r') as f:
    class_mapping = json.load(f)

idx_to_class = class_mapping['idx_to_class']
idx_to_breed = class_mapping['idx_to_breed_name']
idx_to_species = class_mapping['idx_to_species']

distribution_rows = []

for idx_str, class_id in idx_to_class.items():
    idx = int(idx_str)
    bname = idx_to_breed[idx_str]
    species = idx_to_species[idx_str]
    
    n_train = len(train_df[(train_df['breed_name'] == bname) & (train_df['species'].str.lower() == species.lower())])
    n_val = len(val_df[(val_df['breed_name'] == bname) & (val_df['species'].str.lower() == species.lower())])
    n_test = len(test_df[(test_df['breed_name'] == bname) & (test_df['species'].str.lower() == species.lower())])
    n_total = n_train + n_val + n_test
    
    distribution_rows.append({
        'breed_id': class_id,
        'breed_name': bname,
        'species': species,
        'total_images': n_total,
        'train_images': n_train,
        'validation_images': n_val,
        'test_images': n_test
    })

dist_df = pd.DataFrame(distribution_rows)
dist_df.to_csv('reports/dataset_class_distribution.csv', index=False)
print(f"[STEP 1] Saved reports/dataset_class_distribution.csv ({len(dist_df)} classes)")

total_series = dist_df['total_images']
print(f"Total Images: {total_series.sum()}")
print(f"Minimum: {total_series.min()}")
print(f"Maximum: {total_series.max()}")
print(f"Mean: {total_series.mean():.2f}")
print(f"Median: {total_series.median():.2f}")
print(f"Standard Deviation: {total_series.std():.2f}")

lt_10 = dist_df[dist_df['total_images'] < 10]
lt_20 = dist_df[dist_df['total_images'] < 20]
lt_50 = dist_df[dist_df['total_images'] < 50]
lt_100 = dist_df[dist_df['total_images'] < 100]

print(f"Classes with < 10 images: {len(lt_10)} / 82 ({len(lt_10)/82*100:.1f}%)")
print(f"Classes with < 20 images: {len(lt_20)} / 82 ({len(lt_20)/82*100:.1f}%)")
print(f"Classes with < 50 images: {len(lt_50)} / 82 ({len(lt_50)/82*100:.1f}%)")
print(f"Classes with < 100 images: {len(lt_100)} / 82 ({len(lt_100)/82*100:.1f}%)")

# -------------------------------------------------------------
# 2. VERIFY CLASS MAPPING
# -------------------------------------------------------------
print("\n" + "-"*60)
print("[STEP 2] VERIFYING CLASS MAPPING")
print("-"*60)
print(f"num_classes in json: {class_mapping.get('num_classes')}")
print(f"Length of idx_to_class: {len(idx_to_class)}")
print(f"Length of idx_to_breed_name: {len(idx_to_breed)}")
print(f"Length of idx_to_species: {len(idx_to_species)}")

# Check for duplicates or mismatches
class_id_counts = Counter(idx_to_class.values())
dup_classes = [k for k, v in class_id_counts.items() if v > 1]
print(f"Duplicate class_ids in mapping: {dup_classes}")

# Check cattle vs buffalo partition in indices
# Indices 0..22 are buffalo, 23..81 are cattle
buffalo_indices = [int(k) for k, v in idx_to_species.items() if v == 'buffalo']
cattle_indices = [int(k) for k, v in idx_to_species.items() if v == 'cattle']
print(f"Buffalo indices range: {min(buffalo_indices)} to {max(buffalo_indices)} (count: {len(buffalo_indices)})")
print(f"Cattle indices range: {min(cattle_indices)} to {max(cattle_indices)} (count: {len(cattle_indices)})")

# Print first 5 and last 5 class mappings
print("Sample class mappings:")
for i in [0, 1, 22, 23, 26, 81]:
    si = str(i)
    print(f"  Index {i:2d} -> breed_id: {idx_to_class[si]:<22} breed_name: {idx_to_breed[si]:<16} species: {idx_to_species[si]}")

# -------------------------------------------------------------
# 5. VERIFY MODEL CHECKPOINT
# -------------------------------------------------------------
print("\n" + "-"*60)
print("[STEP 5] VERIFYING MODEL CHECKPOINT")
print("-"*60)
ckpt_path = 'models/efficientnet_b0_82_breeds_best.pth'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 82)

# Load state dict
# If keys start with features. or backbone.features.
if any(k.startswith('backbone.') for k in state_dict.keys()):
    state_dict_clean = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
else:
    state_dict_clean = state_dict

model.load_state_dict(state_dict_clean)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
classifier_dim = (model.classifier[1].in_features, model.classifier[1].out_features)

print(f"Checkpoint loaded successfully: {ckpt_path}")
print(f"Architecture: EfficientNet-B0")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Classifier head dimensions: in_features={classifier_dim[0]}, out_features={classifier_dim[1]}")

# -------------------------------------------------------------
# 6. VERIFY TRAINING LEARNING FROM HISTORY
# -------------------------------------------------------------
print("\n" + "-"*60)
print("[STEP 6] VERIFYING TRAINING LEARNING DYNAMICS")
print("-"*60)
hist_df = pd.read_csv('reports/training_history.csv')
print("Training History Overview:")
print(hist_df.to_string(index=False))

best_val_acc = hist_df['val_acc'].max()
best_val_epoch = hist_df.loc[hist_df['val_acc'].idxmax(), 'epoch']
final_train_acc = hist_df.iloc[-1]['train_acc']
final_val_acc = hist_df.iloc[-1]['val_acc']
final_train_loss = hist_df.iloc[-1]['train_loss']
final_val_loss = hist_df.iloc[-1]['val_loss']

print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% at Epoch {best_val_epoch}")
print(f"Final Train Accuracy: {final_train_acc*100:.2f}% | Final Train Loss: {final_train_loss:.4f}")
print(f"Final Val Accuracy: {final_val_acc*100:.2f}% | Final Val Loss: {final_val_loss:.4f}")

# -------------------------------------------------------------
# 7. VERIFY TEST PREDICTIONS AND PREDICTION DISTRIBUTION
# -------------------------------------------------------------
print("\n" + "-"*60)
print("[STEP 7] TEST PREDICTIONS DISTRIBUTION & BIAS")
print("-"*60)
test_preds = pd.read_csv('reports/test_predictions.csv')
n_correct = int(test_preds['correct'].sum())
n_incorrect = len(test_preds) - n_correct
unique_preds = test_preds['predicted_label'].nunique()
unique_pred_names = test_preds['predicted_breed_name'].nunique()

print(f"Total Test Instances: {len(test_preds)}")
print(f"Correct Predictions: {n_correct} ({n_correct/len(test_preds)*100:.2f}%)")
print(f"Incorrect Predictions: {n_incorrect} ({n_incorrect/len(test_preds)*100:.2f}%)")
print(f"Unique Predicted Classes out of 82: {unique_preds} ({unique_preds/82*100:.1f}%)")

pred_counts = test_preds['predicted_breed_name'].value_counts()
print("\nTop 10 Most Frequently Predicted Breeds:")
for bname, count in pred_counts.head(10).items():
    pct = count / len(test_preds) * 100
    # Also find training sample count for this breed
    train_count = dist_df[dist_df['breed_name'] == bname]['train_images'].sum()
    print(f"  {bname:<18}: {count:2d} predictions ({pct:5.1f}%) | Training images available: {train_count}")

# -------------------------------------------------------------
# 8. TOP CONFUSION PAIRS
# -------------------------------------------------------------
print("\n" + "-"*60)
print("[STEP 8] TOP 20 CONFUSION PAIRS")
print("-"*60)
errors_df = test_preds[~test_preds['correct']].copy()
confusion_pairs = errors_df.groupby(['true_breed_name', 'predicted_breed_name', 'species', 'predicted_species']).size().reset_index(name='count')
confusion_pairs = confusion_pairs.sort_values(by='count', ascending=False)

print("Top 20 Confusion Pairs:")
for i, row in confusion_pairs.head(20).reset_index(drop=True).iterrows():
    same_sp = "Same Species" if row['species'].lower() == row['predicted_species'].lower() else "CROSS-SPECIES"
    print(f"  {i+1:2d}. {row['true_breed_name']} ({row['species']}) -> {row['predicted_breed_name']} ({row['predicted_species']}) [Count: {row['count']}] ({same_sp})")

# -------------------------------------------------------------
# 9. CONFIDENCE CALIBRATION & HIGH-CONFIDENCE ERRORS
# -------------------------------------------------------------
print("\n" + "-"*60)
print("[STEP 9] CONFIDENCE ANALYSIS")
print("-"*60)
correct_conf = test_preds[test_preds['correct']]['confidence']
incorrect_conf = test_preds[~test_preds['correct']]['confidence']

print(f"Mean Confidence (Overall): {test_preds['confidence'].mean():.4f}")
print(f"Median Confidence (Overall): {test_preds['confidence'].median():.4f}")
print(f"Mean Confidence (Correct): {correct_conf.mean():.4f}")
print(f"Median Confidence (Correct): {correct_conf.median():.4f}")
print(f"Mean Confidence (Incorrect): {incorrect_conf.mean():.4f}")
print(f"Median Confidence (Incorrect): {incorrect_conf.median():.4f}")

# Random guess in 82 classes = 1/82 = 0.0122 (1.22%)
# High confidence errors (> 0.15, which is >12x random baseline)
high_conf_errors = errors_df[errors_df['confidence'] >= 0.15].sort_values(by='confidence', ascending=False)
print(f"\nNumber of High-Confidence Incorrect Predictions (>= 15.0%): {len(high_conf_errors)}")
for _, row in high_conf_errors.head(5).iterrows():
    print(f"  Image: {row['image_id']} | True: {row['true_breed_name']} ({row['species']}) -> Pred: {row['predicted_breed_name']} ({row['predicted_species']}) | Conf: {row['confidence']*100:.2f}% | In Top-3: {row['in_top_3']}")

# -------------------------------------------------------------
# 10. QUANTITATIVE DATASET SIZE CONTEXT
# -------------------------------------------------------------
print("\n" + "-"*60)
print("[STEP 10] QUANTITATIVE DATASET SIZE CONTEXT")
print("-"*60)
n_train_total = len(train_df)
n_val_total = len(val_df)
n_test_total = len(test_df)
k_classes = 82

avg_train_per_class = n_train_total / k_classes
avg_val_per_class = n_val_total / k_classes
avg_test_per_class = n_test_total / k_classes

print(f"Average training images per class: {avg_train_per_class:.2f}")
print(f"Average validation images per class: {avg_val_per_class:.2f}")
print(f"Average test images per class: {avg_test_per_class:.2f}")
print(f"Theoretical degrees of freedom per class: {n_train_total} samples / (1280 weights + 1 bias) = {n_train_total/1281:.4f} samples per weight!")
