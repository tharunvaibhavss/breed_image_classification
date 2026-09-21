import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

print("="*60)
print("RUNNING CONTROLLED EXPERIMENT (CLASSES WITH >= 20 IMAGES)")
print("="*60)

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Load existing splits
train_df = pd.read_csv('dataset/splits/train.csv')
val_df = pd.read_csv('dataset/splits/validation.csv')
test_df = pd.read_csv('dataset/splits/test.csv')

# Identify classes with >= 20 images in total
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
class_counts = all_df.groupby(['species', 'breed_name', 'breed_id']).size().reset_index(name='count')
controlled_classes_df = class_counts[class_counts['count'] >= 20].sort_values(by='count', ascending=False).reset_index(drop=True)

print(f"Identified {len(controlled_classes_df)} classes with >= 20 images:")
for _, row in controlled_classes_df.iterrows():
    print(f"  {row['breed_name']} ({row['species']}): {row['count']} total images [ID: {row['breed_id']}]")

controlled_breed_ids = controlled_classes_df['breed_id'].tolist()
class_to_idx = {b_id: i for i, b_id in enumerate(controlled_breed_ids)}
idx_to_class = {i: b_id for b_id, i in class_to_idx.items()}
num_classes = len(class_to_idx)

# Filter splits
ctrl_train = train_df[train_df['breed_id'].isin(controlled_breed_ids)].copy().reset_index(drop=True)
ctrl_val = val_df[val_df['breed_id'].isin(controlled_breed_ids)].copy().reset_index(drop=True)
ctrl_test = test_df[test_df['breed_id'].isin(controlled_breed_ids)].copy().reset_index(drop=True)

print(f"\nControlled Dataset Split:")
print(f"  Training samples: {len(ctrl_train)}")
print(f"  Validation samples: {len(ctrl_val)}")
print(f"  Test samples: {len(ctrl_test)}")
print(f"  Total samples: {len(ctrl_train) + len(ctrl_val) + len(ctrl_test)}")

# Dataset class
class FastImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.samples = []
        self.transform = transform
        for _, row in df.iterrows():
            img_path = Path('dataset') / row['relative_path']
            if img_path.exists():
                with Image.open(img_path) as im:
                    rgb_im = im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
                self.samples.append({
                    'image': rgb_im,
                    'label': class_to_idx[row['breed_id']],
                    'image_id': row['image_id'],
                    'breed_name': row['breed_name']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = item['image']
        if self.transform:
            img = self.transform(img)
        return img, item['label'], item['breed_name']

train_tf = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_tf = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = FastImageDataset(ctrl_train, transform=train_tf)
val_ds = FastImageDataset(ctrl_val, transform=eval_tf)
test_ds = FastImageDataset(ctrl_test, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# Model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Training: Stage 1 frozen backbone (10 epochs), Stage 2 fine-tune (5 epochs)
# Stage 1: Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_acc = 0.0
best_model_state = None
history = []

print("\n--- STAGE 1: CLASSIFIER HEAD TRAINING (10 EPOCHS) ---")
for epoch in range(1, 11):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    train_loss = running_loss / total
    train_acc = correct / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        
    history.append({'epoch': epoch, 'stage': 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
    print(f"Epoch {epoch:2d}/10 | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}%")

# Stage 2: Unfreeze deep layers (features[6], features[7], features[8])
print("\n--- STAGE 2: DEEP LAYER FINE-TUNING (5 EPOCHS) ---")
for param in model.features[6:].parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {'params': model.features[6:].parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 5e-4}
], weight_decay=1e-4)

for epoch in range(11, 16):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    train_loss = running_loss / total
    train_acc = correct / total
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        
    history.append({'epoch': epoch, 'stage': 2, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
    print(f"Epoch {epoch:2d}/15 | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}%")

# Save controlled checkpoint
os.makedirs('models', exist_ok=True)
controlled_model_path = 'models/controlled_efficientnet_b0.pth'
torch.save({
    'model_state_dict': best_model_state,
    'class_to_idx': class_to_idx,
    'num_classes': num_classes,
    'controlled_classes': controlled_breed_ids,
    'best_val_acc': best_val_acc
}, controlled_model_path)
print(f"\nSaved controlled model to {controlled_model_path} (Best Val Acc: {best_val_acc*100:.2f}%)")

# Evaluate strictly on unseen test set
model.load_state_dict(best_model_state)
model.eval()

y_true = []
y_pred = []
top3_correct = 0
total_test = 0

with torch.no_grad():
    for imgs, labels, _ in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())
        
        # Top-3
        top3_indices = torch.topk(probs, k=min(3, num_classes), dim=1)[1]
        for l, top3 in zip(labels, top3_indices):
            if l.item() in top3.tolist():
                top3_correct += 1
        total_test += labels.size(0)

acc = accuracy_score(y_true, y_pred)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
top3_acc = top3_correct / total_test

print("\n" + "="*60)
print("CONTROLLED EXPERIMENT RESULTS (UNSEEN TEST SET)")
print("="*60)
print(f"Number of classes: {num_classes}")
print(f"Test samples: {total_test}")
print(f"Top-1 Accuracy: {acc*100:.2f}%")
print(f"Macro Precision: {macro_p*100:.2f}%")
print(f"Macro Recall: {macro_r*100:.2f}%")
print(f"Macro F1-Score: {macro_f1*100:.2f}%")
print(f"Weighted Precision: {weighted_p*100:.2f}%")
print(f"Weighted F1-Score: {weighted_f1*100:.2f}%")
print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")

ctrl_results = {
    'num_classes': num_classes,
    'classes': controlled_breed_ids,
    'train_images': len(ctrl_train),
    'val_images': len(ctrl_val),
    'test_images': len(ctrl_test),
    'accuracy': float(acc),
    'macro_precision': float(macro_p),
    'macro_recall': float(macro_r),
    'macro_f1': float(macro_f1),
    'weighted_precision': float(weighted_p),
    'weighted_f1': float(weighted_f1),
    'top3_accuracy': float(top3_acc),
}

with open('reports/controlled_experiment_results.json', 'w') as f:
    json.dump(ctrl_results, f, indent=2)
print("Saved reports/controlled_experiment_results.json")
