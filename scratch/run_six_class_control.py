import os
import json
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
print("RUNNING 6-CLASS PROTOTYPE CONTROL EXPERIMENT")
print("Gir, Ongole, Sahiwal, Jaffarabadi, Murrah, Surti")
print("="*60)

torch.manual_seed(42)
np.random.seed(42)

train_df = pd.read_csv('dataset/splits/train.csv')
val_df = pd.read_csv('dataset/splits/validation.csv')
test_df = pd.read_csv('dataset/splits/test.csv')

six_classes_names = ['Gir', 'Ongole', 'Sahiwal', 'Jaffarabadi', 'Murrah', 'Surti']
six_train = train_df[train_df['breed_name'].isin(six_classes_names)].copy().reset_index(drop=True)
six_val = val_df[val_df['breed_name'].isin(six_classes_names)].copy().reset_index(drop=True)
six_test = test_df[test_df['breed_name'].isin(six_classes_names)].copy().reset_index(drop=True)

class_to_idx = {name: i for i, name in enumerate(sorted(six_classes_names))}
num_classes = len(class_to_idx)

print(f"Six-class split counts:")
print(f"  Training: {len(six_train)}")
print(f"  Validation: {len(six_val)}")
print(f"  Testing: {len(six_test)}")

class FastDataset(Dataset):
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
                    'label': class_to_idx[row['breed_name']],
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

train_ds = FastDataset(six_train, transform=train_tf)
val_ds = FastDataset(six_val, transform=eval_tf)
test_ds = FastDataset(six_test, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

device = torch.device("cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()

for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_acc = 0.0
best_model_state = None

for epoch in range(1, 11):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
        
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    val_acc = val_correct / val_total if val_total > 0 else 0
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()

# Fine-tune stage 2
for param in model.features[6:].parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {'params': model.features[6:].parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 5e-4}
], weight_decay=1e-4)

for epoch in range(11, 16):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
        
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    val_acc = val_correct / val_total if val_total > 0 else 0
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()

torch.save({
    'model_state_dict': best_model_state,
    'class_to_idx': class_to_idx,
    'best_val_acc': best_val_acc
}, 'models/six_class_control_efficientnet_b0.pth')
print("Saved models/six_class_control_efficientnet_b0.pth")

# Test evaluation
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
        top3_indices = torch.topk(probs, k=min(3, num_classes), dim=1)[1]
        for l, top3 in zip(labels, top3_indices):
            if l.item() in top3.tolist():
                top3_correct += 1
        total_test += labels.size(0)

acc = accuracy_score(y_true, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
top3_acc = top3_correct / total_test

print("\n" + "="*60)
print("SIX-CLASS PROTOTYPE CONTROL EVALUATION (UNSEEN TEST SET)")
print("="*60)
print(f"Accuracy: {acc*100:.2f}%")
print(f"Macro Precision: {p*100:.2f}%")
print(f"Macro Recall: {r*100:.2f}%")
print(f"Macro F1: {f1*100:.2f}%")
print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")

six_res = {
    'classes': six_classes_names,
    'train_images': len(six_train),
    'val_images': len(six_val),
    'test_images': len(six_test),
    'accuracy': float(acc),
    'macro_precision': float(p),
    'macro_recall': float(r),
    'macro_f1': float(f1),
    'top3_accuracy': float(top3_acc)
}
with open('reports/six_class_experiment_results.json', 'w') as f:
    json.dump(six_res, f, indent=2)
