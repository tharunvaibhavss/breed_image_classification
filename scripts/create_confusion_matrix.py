"""
Generates the 82x82 Confusion Matrix, Precision/Recall visualizations,
and Training/Validation Performance curves.
Outputs:
  - plots/confusion_matrix.png
  - reports/confusion_matrix.csv
  - plots/training_loss.png
  - plots/validation_loss.png
  - plots/training_accuracy.png
  - plots/validation_accuracy.png
  - plots/per_class_precision.png
"""

import os
import sys
import csv
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

WORKSPACE = Path(__file__).resolve().parent.parent
PLOTS_DIR = WORKSPACE / "plots"
REPORTS_DIR = WORKSPACE / "reports"
MODELS_DIR = WORKSPACE / "models"

CLASS_MAPPING_PATH = MODELS_DIR / "class_names.json"
PREDICTIONS_CSV = REPORTS_DIR / "test_predictions.csv"
HISTORY_CSV = REPORTS_DIR / "training_history.csv"
CONFUSION_CSV = REPORTS_DIR / "confusion_matrix.csv"
CLF_REPORT_CSV = REPORTS_DIR / "classification_report.csv"


def run_visualization():
    print("=" * 70)
    print("GENERATING EVALUATION PLOTS AND CONFUSION MATRIX")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTIONS_CSV.exists():
        print(f"Error: {PREDICTIONS_CSV} not found. Run evaluation first.")
        sys.exit(1)

    # 1. Load classes
    with open(CLASS_MAPPING_PATH, mode="r", encoding="utf-8") as f:
        mapping = json.load(f)
    classes = mapping["classes"]
    class_to_idx = mapping["class_to_idx"]
    num_classes = len(classes)

    # 2. Read test predictions
    y_true = []
    y_pred = []
    with open(PREDICTIONS_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            y_true.append(class_to_idx[r["true_label"]])
            y_pred.append(class_to_idx[r["predicted_label"]])

    # 3. Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # Save confusion matrix CSV
    with open(CONFUSION_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class"] + classes)
        for idx, row in enumerate(cm):
            writer.writerow([classes[idx]] + list(row))
    print(f"Saved confusion matrix CSV to {CONFUSION_CSV}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(24, 22))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted Breed", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Breed", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Confusion Matrix ({num_classes} Breeds on Unseen Test Partition)", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    cm_plot_path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")

    # 4. Plot Training History (if available)
    if HISTORY_CSV.exists():
        epochs = []
        tr_loss = []
        val_loss = []
        tr_acc = []
        val_acc = []

        with open(HISTORY_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                epochs.append(int(r["epoch"]))
                tr_loss.append(float(r["train_loss"]))
                val_loss.append(float(r["val_loss"]))
                tr_acc.append(float(r["train_acc"]) * 100.0)
                val_acc.append(float(r["val_acc"]) * 100.0)

        # Plot Loss Curves
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, tr_loss, "b-o", linewidth=2, label="Training Loss")
        plt.plot(epochs, val_loss, "r--s", linewidth=2, label="Validation Loss")
        plt.axvline(x=10.5, color="gray", linestyle=":", label="Stage 1 / Stage 2 Boundary")
        plt.title("Training and Validation Loss Progression (EfficientNet-B0)", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Cross-Entropy Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_loss.png", dpi=200)
        plt.close()

        # Validation Loss standalone plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, val_loss, "r-s", linewidth=2, label="Validation Loss")
        plt.title("Validation Loss Progression Across Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Validation Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "validation_loss.png", dpi=200)
        plt.close()

        # Plot Accuracy Curves
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, tr_acc, "g-o", linewidth=2, label="Training Accuracy (%)")
        plt.plot(epochs, val_acc, "m--^", linewidth=2, label="Validation Accuracy (%)")
        plt.axvline(x=10.5, color="gray", linestyle=":", label="Stage 1 / Stage 2 Boundary")
        plt.title("Training and Validation Accuracy Progression (EfficientNet-B0)", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_accuracy.png", dpi=200)
        plt.close()

        # Validation Accuracy standalone plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, val_acc, "m-^", linewidth=2, label="Validation Accuracy (%)")
        plt.title("Validation Accuracy Progression Across Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Validation Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "validation_accuracy.png", dpi=200)
        plt.close()
        print("Saved training & validation loss and accuracy curves to plots/")

    # 5. Per-Class Precision Bar Chart
    if CLF_REPORT_CSV.exists():
        breeds = []
        precisions = []
        with open(CLF_REPORT_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                breeds.append(r["breed_name"])
                precisions.append(float(r["precision"]) * 100.0)

        plt.figure(figsize=(22, 10))
        bars = plt.bar(range(len(breeds)), precisions, color="#2b5c8f", edgecolor="#1a365d")
        plt.xticks(range(len(breeds)), breeds, rotation=90, fontsize=6)
        plt.xlabel("Breed Name", fontsize=12, fontweight="bold")
        plt.ylabel("Precision (%)", fontsize=12, fontweight="bold")
        plt.title("Per-Class Precision Across All 82 Breeds (Unseen Test Partition)", fontsize=14, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "per_class_precision.png", dpi=200)
        plt.close()
        print("Saved per-class precision bar chart to plots/per_class_precision.png")


if __name__ == "__main__":
    run_visualization()
