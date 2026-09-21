import os
import glob
import pandas as pd
import numpy as np

train_df = pd.read_csv('dataset/splits/train.csv')
val_df = pd.read_csv('dataset/splits/validation.csv')
test_df = pd.read_csv('dataset/splits/test.csv')

print(f"Train images: {len(train_df)}")
print(f"Val images: {len(val_df)}")
print(f"Test images: {len(test_df)}")
total = len(train_df) + len(val_df) + len(test_df)
print(f"Total split images: {total}")

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Total unique breeds in splits: {all_df['breed_name'].nunique()}")
cattle_breeds = all_df[all_df['species'].str.upper() == 'CATTLE']['breed_name'].nunique()
buffalo_breeds = all_df[all_df['species'].str.upper() == 'BUFFALO']['breed_name'].nunique()
print(f"Cattle breeds: {cattle_breeds}")
print(f"Buffalo breeds: {buffalo_breeds}")

counts = all_df['breed_name'].value_counts()
print(f"Min images/class: {counts.min()}")
print(f"Max images/class: {counts.max()}")
print(f"Average images/class: {counts.mean():.2f}")
print(f"Median images/class: {counts.median():.2f}")

# Check test representation
test_counts = test_df['breed_name'].value_counts()
print(f"Breeds represented in test: {len(test_counts)}")
zero_test = set(all_df['breed_name'].unique()) - set(test_df['breed_name'].unique())
print(f"Breeds with zero test images: {len(zero_test)}")

# Also check raw and cleaned dataset folder paths
cleaned_cattle = len([d for d in os.listdir('dataset/cleaned/cattle') if os.path.isdir(os.path.join('dataset/cleaned/cattle', d))])
cleaned_buffalo = len([d for d in os.listdir('dataset/cleaned/buffalo') if os.path.isdir(os.path.join('dataset/cleaned/buffalo', d))])
print(f"Cleaned cattle folders: {cleaned_cattle}")
print(f"Cleaned buffalo folders: {cleaned_buffalo}")
