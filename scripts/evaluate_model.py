#!/usr/bin/env python3
"""CLI script to run test set evaluation and error analysis for EfficientNet-B0 breed classifier."""

import sys
import os
import json
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.classification.dataset import create_dataloaders
from ml.evaluation.evaluator import ModelEvaluator
from ml.evaluation.error_analyzer import ErrorAnalyzer
from ml.evaluation.report_generator import (
    generate_confusion_matrix_plot,
    generate_evaluation_reports,
)


def main():
    """Execute Phase 7 Model Evaluation and Error Analysis pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "processed" / "dataset_manifest.json"
    dataset_dir = project_root / "data" / "raw" / "dataset"
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"
    docs_dir = project_root / "docs"

    best_model_path = models_dir / "efficientnet_best.pth"

    print("=" * 70)
    print(" PHASE 7 - MODEL EVALUATION AND ERROR ANALYSIS")
    print("=" * 70)
    print(f" Manifest Path  : {manifest_path}")
    print(f" Model Weights  : {best_model_path}")
    print(f" Output Reports : {processed_dir}")
    print(f" Docs Output    : {docs_dir}")
    print("-" * 70)

    # 1. Load manifest records
    manifest_records = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            manifest_records = data.get("manifest_records", [])

    test_samples = [r for r in manifest_records if r.get("split") == "test"]
    print(f"[1/4] Loaded {len(test_samples)} unseen test set records...")

    # 2. Create PyTorch DataLoaders
    print("[2/4] Constructing test set PyTorch DataLoader...")
    dataloaders = create_dataloaders(
        manifest_records=manifest_records,
        base_dir=dataset_dir,
        batch_size=16,
        num_workers=0,
        target_size=(224, 224),
    )
    test_loader = dataloaders["test"]

    # 3. Evaluate EfficientNet-B0 Model
    print("[3/4] Running model evaluation on test set...")
    evaluator = ModelEvaluator(num_classes=6)
    eval_stats = evaluator.evaluate(
        test_loader=test_loader, model_path=best_model_path
    )

    # 4. Perform Error Analysis
    print("[4/4] Performing error analysis and generating reports...")
    analyzer = ErrorAnalyzer()
    error_stats = analyzer.analyze_errors(
        y_true=eval_stats.get("y_true", []),
        y_pred_probs=eval_stats.get("y_pred_probs", []),
        manifest_samples=test_samples,
    )

    # Render confusion matrix plot
    plot_path = generate_confusion_matrix_plot(
        confusion_matrix=eval_stats["confusion_matrix"],
        output_path=docs_dir / "confusion_matrix.png",
    )

    # Generate JSON and Markdown reports
    eval_json, error_json, md_doc = generate_evaluation_reports(
        eval_stats=eval_stats,
        error_stats=error_stats,
        output_dir=processed_dir,
        docs_dir=docs_dir,
    )

    print("=" * 70)
    print(" EVALUATION SUMMARY")
    print("=" * 70)
    print(f" Total Test Samples   : {eval_stats['total_test_samples']}")
    print(f" Top-1 Accuracy       : {eval_stats['accuracy']}%")
    print(f" Top-3 Accuracy       : {eval_stats['top_3_accuracy']}%")
    print(f" Precision (Macro)    : {eval_stats['precision_macro']}%")
    print(f" Recall (Macro)       : {eval_stats['recall_macro']}%")
    print(f" F1-Score (Macro)     : {eval_stats['f1_macro']}%")
    print(f" F1-Score (Weighted)  : {eval_stats['f1_weighted']}%")
    print(f" Confusion Matrix Plot: {plot_path}")
    print(f" Markdown Report      : {md_doc}")
    print("=" * 70)


if __name__ == "__main__":
    main()
