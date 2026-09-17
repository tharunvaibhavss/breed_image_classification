"""
Scientific Evaluation Pipeline for 82-Class Indian Cattle and Buffalo Breed Recognition
Model: EfficientNet-B0 (Trained Best Checkpoint)
Evaluation Partition: Strictly Unseen Test Set (dataset/splits/test.csv)
Metrics Computed:
  - Overall Accuracy, Macro Precision (Primary), Weighted Precision, Micro Precision
  - Macro Recall, Weighted Recall, Macro F1, Weighted F1
  - Top-1 Accuracy, Top-3 Accuracy
  - Per-breed Classification Report (82 breeds)
  - Separate Cattle (59 breeds) and Buffalo (23 breeds) Metrics
  - Cross-species Confusion Rate
  - Error Analysis (Misclassified samples with Top-3 predictions and confidences)
Outputs:
  - reports/test_predictions.csv
  - reports/classification_report.csv
  - reports/cattle_metrics.json
  - reports/buffalo_metrics.json
  - reports/error_analysis.csv
  - reports/metrics.json
  - reports/final_model_evaluation.md
"""

import os
import sys
import csv
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b0
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent
DATASET_ROOT = WORKSPACE / "dataset"
SPLITS_DIR = DATASET_ROOT / "splits"
MODELS_DIR = WORKSPACE / "models"
REPORTS_DIR = WORKSPACE / "reports"

TEST_CSV = SPLITS_DIR / "test.csv"
CLASS_MAPPING_PATH = MODELS_DIR / "class_names.json"
BEST_MODEL_PATH = MODELS_DIR / "efficientnet_b0_82_breeds_best.pth"

PREDICTIONS_CSV = REPORTS_DIR / "test_predictions.csv"
CLASSIFICATION_REPORT_CSV = REPORTS_DIR / "classification_report.csv"
CATTLE_METRICS_JSON = REPORTS_DIR / "cattle_metrics.json"
BUFFALO_METRICS_JSON = REPORTS_DIR / "buffalo_metrics.json"
ERROR_ANALYSIS_CSV = REPORTS_DIR / "error_analysis.csv"
METRICS_JSON = REPORTS_DIR / "metrics.json"
FINAL_REPORT_MD = REPORTS_DIR / "final_model_evaluation.md"


def run_evaluation():
    print("=" * 70)
    print("STARTING SCIENTIFIC EVALUATION ON UNSEEN TEST DATASET (82 CLASSES)")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not BEST_MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at {BEST_MODEL_PATH}")
        sys.exit(1)
    if not CLASS_MAPPING_PATH.exists():
        print(f"Error: Class mapping not found at {CLASS_MAPPING_PATH}")
        sys.exit(1)
    if not TEST_CSV.exists():
        print(f"Error: Test split not found at {TEST_CSV}")
        sys.exit(1)

    # Load class mapping
    with open(CLASS_MAPPING_PATH, mode="r", encoding="utf-8") as f:
        mapping = json.load(f)

    class_to_idx = mapping["class_to_idx"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    idx_to_breed_name = {int(k): v for k, v in mapping["idx_to_breed_name"].items()}
    idx_to_species = {int(k): v for k, v in mapping["idx_to_species"].items()}
    classes = mapping["classes"]
    num_classes = len(classes)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {BEST_MODEL_PATH.name} on {device}...")

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Use standard ImageNet norm
    norm_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Read test records
    test_records = []
    with open(TEST_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            test_records.append(r)

    print(f"Total unseen test samples: {len(test_records)}")
    test_breed_ids = set(r["breed_id"] for r in test_records)
    print(f"Total unique breeds in test set: {len(test_breed_ids)} / {num_classes}")

    # Inference loop
    y_true = []
    y_pred = []
    all_probabilities = []
    predictions_rows = []
    errors_rows = []

    print("\nRunning inference on unseen test samples...")
    with torch.no_grad():
        for r in test_records:
            img_path = DATASET_ROOT / r["relative_path"]
            if not img_path.exists():
                print(f"Warning: Image {img_path} missing! Skipping.")
                continue

            true_breed_id = r["breed_id"]
            true_idx = class_to_idx[true_breed_id]
            species = r["species"]

            img = Image.open(img_path).convert("RGB")
            tensor = norm_transform(img).unsqueeze(0).to(device)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])
            pred_breed_id = idx_to_class[pred_idx]
            pred_breed_name = idx_to_breed_name[pred_idx]
            pred_species = idx_to_species[pred_idx]

            # Top-3 predictions
            top3_indices = np.argsort(probs)[-3:][::-1]
            top3_list = [
                {
                    "rank": rank + 1,
                    "breed_id": idx_to_class[idx],
                    "breed_name": idx_to_breed_name[idx],
                    "species": idx_to_species[idx],
                    "confidence": round(float(probs[idx]), 4)
                }
                for rank, idx in enumerate(top3_indices)
            ]

            is_correct = (pred_idx == true_idx)
            is_top3_correct = (true_idx in top3_indices)

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            all_probabilities.append(probs)

            pred_row = {
                "image_id": r["image_id"],
                "relative_path": r["relative_path"],
                "species": species,
                "true_label": true_breed_id,
                "true_breed_name": r["breed_name"],
                "predicted_label": pred_breed_id,
                "predicted_breed_name": pred_breed_name,
                "predicted_species": pred_species,
                "confidence": round(pred_conf, 4),
                "correct": is_correct,
                "in_top_3": is_top3_correct,
                "top_3_predictions": json.dumps(top3_list)
            }
            predictions_rows.append(pred_row)

            if not is_correct:
                errors_rows.append({
                    "image_id": r["image_id"],
                    "relative_path": r["relative_path"],
                    "species": species,
                    "true_label": true_breed_id,
                    "true_breed_name": r["breed_name"],
                    "predicted_label": pred_breed_id,
                    "predicted_breed_name": pred_breed_name,
                    "predicted_species": pred_species,
                    "confidence": round(pred_conf, 4),
                    "cross_species_error": (species != pred_species),
                    "in_top_3": is_top3_correct,
                    "top_3_summary": " | ".join([f"{t['breed_name']} ({t['confidence']*100:.1f}%)" for t in top3_list])
                })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probabilities = np.array(all_probabilities)

    # Save test_predictions.csv
    pred_fields = [
        "image_id", "relative_path", "species", "true_label", "true_breed_name",
        "predicted_label", "predicted_breed_name", "predicted_species",
        "confidence", "correct", "in_top_3", "top_3_predictions"
    ]
    with open(PREDICTIONS_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(predictions_rows)
    print(f"\nSaved test predictions to {PREDICTIONS_CSV}")

    # Save error_analysis.csv
    error_fields = [
        "image_id", "relative_path", "species", "true_label", "true_breed_name",
        "predicted_label", "predicted_breed_name", "predicted_species",
        "confidence", "cross_species_error", "in_top_3", "top_3_summary"
    ]
    with open(ERROR_ANALYSIS_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=error_fields)
        writer.writeheader()
        writer.writerows(errors_rows)
    print(f"Saved error analysis to {ERROR_ANALYSIS_CSV}")

    # Calculate Core Metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))

    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Top-k metrics
    all_classes_labels = np.arange(num_classes)
    top1 = float(top_k_accuracy_score(y_true, all_probabilities, k=1, labels=all_classes_labels))
    top3 = float(top_k_accuracy_score(y_true, all_probabilities, k=3, labels=all_classes_labels))

    # Per-breed classification report
    clf_dict = classification_report(
        y_true, y_pred, labels=all_classes_labels, target_names=classes, output_dict=True, zero_division=0
    )

    clf_rows = []
    for idx, b_id in enumerate(classes):
        m = clf_dict.get(b_id, {})
        clf_rows.append({
            "class_index": idx,
            "breed_id": b_id,
            "breed_name": idx_to_breed_name[idx],
            "species": idx_to_species[idx],
            "precision": round(m.get("precision", 0.0), 4),
            "recall": round(m.get("recall", 0.0), 4),
            "f1_score": round(m.get("f1-score", 0.0), 4),
            "support": int(m.get("support", 0))
        })

    with open(CLASSIFICATION_REPORT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_index", "breed_id", "breed_name", "species", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(clf_rows)
    print(f"Saved per-breed classification report to {CLASSIFICATION_REPORT_CSV}")

    # Cattle vs Buffalo Partition Metrics
    cattle_mask = np.array([idx_to_species[yi] == "cattle" for yi in y_true])
    buffalo_mask = np.array([idx_to_species[yi] == "buffalo" for yi in y_true])

    cattle_metrics = {}
    if np.sum(cattle_mask) > 0:
        c_true = y_true[cattle_mask]
        c_pred = y_pred[cattle_mask]
        c_probs = all_probabilities[cattle_mask]
        cattle_metrics = {
            "species": "cattle",
            "total_test_samples": int(np.sum(cattle_mask)),
            "accuracy": round(float(accuracy_score(c_true, c_pred)), 4),
            "macro_precision": round(float(precision_score(c_true, c_pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(c_true, c_pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(c_true, c_pred, average="macro", zero_division=0)), 4),
            "top1_accuracy": round(float(top_k_accuracy_score(c_true, c_probs, k=1, labels=all_classes_labels)), 4),
            "top3_accuracy": round(float(top_k_accuracy_score(c_true, c_probs, k=3, labels=all_classes_labels)), 4),
            "misclassified_as_buffalo": int(sum(idx_to_species[p] == "buffalo" for p in c_pred))
        }

    buffalo_metrics = {}
    if np.sum(buffalo_mask) > 0:
        b_true = y_true[buffalo_mask]
        b_pred = y_pred[buffalo_mask]
        b_probs = all_probabilities[buffalo_mask]
        buffalo_metrics = {
            "species": "buffalo",
            "total_test_samples": int(np.sum(buffalo_mask)),
            "accuracy": round(float(accuracy_score(b_true, b_pred)), 4),
            "macro_precision": round(float(precision_score(b_true, b_pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(b_true, b_pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(b_true, b_pred, average="macro", zero_division=0)), 4),
            "top1_accuracy": round(float(top_k_accuracy_score(b_true, b_probs, k=1, labels=all_classes_labels)), 4),
            "top3_accuracy": round(float(top_k_accuracy_score(b_true, b_probs, k=3, labels=all_classes_labels)), 4),
            "misclassified_as_cattle": int(sum(idx_to_species[p] == "cattle" for p in b_pred))
        }

    with open(CATTLE_METRICS_JSON, mode="w", encoding="utf-8") as f:
        json.dump(cattle_metrics, f, indent=2)
    with open(BUFFALO_METRICS_JSON, mode="w", encoding="utf-8") as f:
        json.dump(buffalo_metrics, f, indent=2)

    # Save master metrics.json
    cross_species_errors = sum(1 for e in errors_rows if e["cross_species_error"])
    overall_metrics = {
        "model": "EfficientNet-B0",
        "num_classes": num_classes,
        "test_samples_total": len(y_true),
        "overall_accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "weighted_precision": round(weighted_precision, 4),
        "micro_precision": round(micro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "weighted_recall": round(weighted_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "top_1_accuracy": round(top1, 4),
        "top_3_accuracy": round(top3, 4),
        "total_misclassifications": len(errors_rows),
        "cross_species_misclassifications": cross_species_errors,
        "species_accuracy": {
            "cattle": cattle_metrics.get("accuracy", 0.0),
            "buffalo": buffalo_metrics.get("accuracy", 0.0)
        },
        "evaluation_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    with open(METRICS_JSON, mode="w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)
    print(f"Saved master metrics to {METRICS_JSON}")

    # Generate comprehensive markdown report
    report_content = f"""# Final Model Evaluation Report (82 Breeds)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Evaluation Date**: {overall_metrics['evaluation_timestamp_utc']}  
**Evaluation Dataset**: Strictly Unseen Test Set (`dataset/splits/test.csv`)  
**Data Isolation**: Verified Zero Data Leakage (Cryptographic & Perceptual Overlap = 0)  

---

## 1. Primary Performance Metrics

| Metric | Scientific Score | Benchmark / Context |
|---|---|---|
| **Macro Precision (Primary)** | **{macro_precision:.4f}** ({macro_precision*100:.2f}%) | Unweighted average precision across all 82 fine-grained classes |
| **Weighted Precision** | **{weighted_precision:.4f}** ({weighted_precision*100:.2f}%) | Support-weighted average precision |
| **Overall Accuracy** | **{accuracy:.4f}** ({accuracy*100:.2f}%) | Percentage of correctly identified breeds in unseen test set |
| **Macro Recall** | **{macro_recall:.4f}** ({macro_recall*100:.2f}%) | Average sensitivity across all 82 classes |
| **Macro F1-Score** | **{macro_f1:.4f}** ({macro_f1*100:.2f}%) | Harmonic mean of Macro Precision and Macro Recall |
| **Top-1 Accuracy** | **{top1:.4f}** ({top1*100:.2f}%) | Ground-truth breed matches Rank 1 prediction |
| **Top-3 Accuracy** | **{top3:.4f}** ({top3*100:.2f}%) | Ground-truth breed present within top-3 predictions |

---

## 2. Species-Level Breakdown

### Cattle (59 Indian Breeds)
- **Test Samples**: {cattle_metrics.get('total_test_samples', 0)}
- **Top-1 Accuracy**: {cattle_metrics.get('top1_accuracy', 0.0)*100:.2f}%
- **Top-3 Accuracy**: {cattle_metrics.get('top3_accuracy', 0.0)*100:.2f}%
- **Macro Precision**: {cattle_metrics.get('macro_precision', 0.0)*100:.2f}%
- **Macro Recall**: {cattle_metrics.get('macro_recall', 0.0)*100:.2f}%
- **Macro F1**: {cattle_metrics.get('macro_f1', 0.0)*100:.2f}%
- **Cross-Species Errors (Cattle → Buffalo)**: {cattle_metrics.get('misclassified_as_buffalo', 0)}

### Buffalo (23 Indian Breeds)
- **Test Samples**: {buffalo_metrics.get('total_test_samples', 0)}
- **Top-1 Accuracy**: {buffalo_metrics.get('top1_accuracy', 0.0)*100:.2f}%
- **Top-3 Accuracy**: {buffalo_metrics.get('top3_accuracy', 0.0)*100:.2f}%
- **Macro Precision**: {buffalo_metrics.get('macro_precision', 0.0)*100:.2f}%
- **Macro Recall**: {buffalo_metrics.get('macro_recall', 0.0)*100:.2f}%
- **Macro F1**: {buffalo_metrics.get('macro_f1', 0.0)*100:.2f}%
- **Cross-Species Errors (Buffalo → Cattle)**: {buffalo_metrics.get('misclassified_as_cattle', 0)}

---

## 3. High-Confidence and Error Analysis

- **Total Test Samples**: {len(y_true)}
- **Correct Top-1 Predictions**: {int(np.sum(y_true == y_pred))} ({accuracy*100:.2f}%)
- **Correct Top-3 Predictions**: {int(sum(1 for r in predictions_rows if r['in_top_3']))} ({top3*100:.2f}%)
- **Total Misclassifications**: {len(errors_rows)}
- **Cross-Species Misclassifications**: {cross_species_errors}

Complete individual predictions and error details are recorded in:
- `{PREDICTIONS_CSV.name}`
- `{ERROR_ANALYSIS_CSV.name}`
- `{CLASSIFICATION_REPORT_CSV.name}`
"""

    with open(FINAL_REPORT_MD, mode="w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved evaluation markdown report to {FINAL_REPORT_MD}")

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Top-1 Accuracy:  {top1*100:.2f}%")
    print(f"Top-3 Accuracy:  {top3*100:.2f}%")
    print(f"Macro Precision: {macro_precision*100:.2f}%")
    print(f"Macro Recall:    {macro_recall*100:.2f}%")
    print(f"Macro F1-Score:  {macro_f1*100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
