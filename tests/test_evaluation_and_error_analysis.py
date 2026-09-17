"""Unit tests for Phase 7 model evaluation, error analysis, confusion matrix, and report generation."""

import json
from pathlib import Path
import numpy as np
import pytest

from ml.evaluation.evaluator import ModelEvaluator, CLASS_NAMES
from ml.evaluation.error_analyzer import ErrorAnalyzer
from ml.evaluation.report_generator import (
    generate_confusion_matrix_plot,
    generate_evaluation_reports,
)


def test_confusion_matrix_computation():
    """Verify ModelEvaluator produces a valid 6x6 confusion matrix."""
    evaluator = ModelEvaluator(num_classes=6)

    y_true = np.array([0, 0, 1, 2, 3, 4, 5, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 3, 4, 5, 0, 1, 2])  # 1 error: true 0 pred 1

    cm = evaluator.compute_confusion_matrix(y_true, y_pred)

    assert cm.shape == (6, 6)
    assert np.sum(cm) == 10
    assert cm[0, 0] == 2
    assert cm[0, 1] == 1  # 1 misclassified sample


def test_per_class_metrics_computation():
    """Verify per-class precision, recall, F1, and support calculation."""
    evaluator = ModelEvaluator(num_classes=6)
    cm = np.zeros((6, 6), dtype=int)
    cm[0, 0] = 5
    cm[1, 1] = 5
    cm[2, 2] = 5

    per_class = evaluator.compute_per_class_metrics(cm)

    assert len(per_class) == 6
    assert per_class["Gir"]["precision"] == 100.0
    assert per_class["Gir"]["recall"] == 100.0
    assert per_class["Gir"]["support"] == 5


def test_error_analyzer_grouping():
    """Verify ErrorAnalyzer correctly identifies confusion pairs and low confidence samples."""
    analyzer = ErrorAnalyzer()

    y_true = [0, 0, 1, 2, 3, 4, 5]
    # Class 0 predicted as Class 2 (Gir misclassified as Sahiwal)
    y_pred_probs = [
        [0.9, 0.02, 0.05, 0.01, 0.01, 0.01],  # Correct (conf 0.90)
        [0.1, 0.05, 0.80, 0.02, 0.02, 0.01],  # Incorrect: True Gir(0) -> Pred Sahiwal(2)
        [0.05, 0.40, 0.35, 0.10, 0.05, 0.05],  # Low confidence: Conf 0.40 < 0.50
        [0.01, 0.01, 0.95, 0.01, 0.01, 0.01],  # Correct Sahiwal(2)
        [0.01, 0.01, 0.01, 0.95, 0.01, 0.01],  # Correct Jaffarabadi(3)
        [0.01, 0.01, 0.01, 0.01, 0.95, 0.01],  # Correct Murrah(4)
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.95],  # Correct Surti(5)
    ]

    report = analyzer.analyze_errors(y_true, y_pred_probs, confidence_threshold=0.50)

    assert report["total_eval_samples"] == 7
    assert report["total_errors"] == 1
    assert report["total_low_confidence_samples"] == 1
    assert len(report["top_breed_confusion_pairs"]) == 1
    assert report["top_breed_confusion_pairs"][0]["true_breed"] == "Gir"
    assert report["top_breed_confusion_pairs"][0]["predicted_breed"] == "Sahiwal"


def test_report_generator_outputs(tmp_path: Path):
    """Verify confusion matrix plot and evaluation report generation."""
    cm = [[5, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0], [0, 0, 5, 0, 0, 0], [0, 0, 0, 5, 0, 0], [0, 0, 0, 0, 5, 0], [0, 0, 0, 0, 0, 5]]
    plot_path = tmp_path / "confusion_matrix.png"

    res_plot = generate_confusion_matrix_plot(cm, output_path=plot_path)
    assert res_plot.exists()

    eval_stats = {
        "total_test_samples": 30,
        "test_loss": 0.05,
        "accuracy": 100.0,
        "top_3_accuracy": 100.0,
        "precision_macro": 100.0,
        "recall_macro": 100.0,
        "f1_macro": 100.0,
        "f1_weighted": 100.0,
        "per_class_metrics": {
            "Gir": {"class_id": 0, "precision": 100.0, "recall": 100.0, "f1_score": 100.0, "support": 5}
        },
    }

    error_stats = {
        "total_errors": 0,
        "error_rate_percent": 0.0,
        "total_low_confidence_samples": 0,
        "top_breed_confusion_pairs": [],
    }

    eval_json, error_json, md_doc = generate_evaluation_reports(
        eval_stats=eval_stats,
        error_stats=error_stats,
        output_dir=tmp_path / "processed",
        docs_dir=tmp_path / "docs",
    )

    assert eval_json.exists()
    assert error_json.exists()
    assert md_doc.exists()

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["evaluation_metrics"]["accuracy"] == 100.0
