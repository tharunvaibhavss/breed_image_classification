"""Report & Heatmap Generator module for evaluation metrics and error analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.config import settings
from ml.evaluation.evaluator import CLASS_NAMES


def generate_confusion_matrix_plot(
    confusion_matrix: List[List[int]],
    class_names: List[str] = CLASS_NAMES,
    output_path: Optional[Path] = None,
) -> Path:
    """Render 6x6 annotated confusion matrix heatmap plot.

    Args:
        confusion_matrix: 2D list or numpy array of shape (6, 6).
        class_names: List of breed class names.
        output_path: Target PNG output file path.

    Returns:
        Path to output plot PNG.
    """
    if output_path is None:
        output_path = Path("docs/confusion_matrix.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_np = np.array(confusion_matrix, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_np,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )

    ax.set_title("Breed Classification Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Breed Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Breed Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def generate_evaluation_reports(
    eval_stats: Dict[str, Any],
    error_stats: Dict[str, Any],
    output_dir: Path,
    docs_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Generate evaluation_report.json, error_analysis_report.json, and docs/evaluation_report.md.

    Args:
        eval_stats: Evaluation metrics dictionary from ModelEvaluator.
        error_stats: Error analysis dictionary from ErrorAnalyzer.
        output_dir: Output directory (e.g. data/processed).
        docs_dir: Documentation output directory (e.g. docs).

    Returns:
        Tuple of (eval_json_path, error_json_path, markdown_doc_path).
    """
    output_dir = Path(output_dir)
    docs_dir = Path(docs_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    eval_json_path = output_dir / "evaluation_report.json"
    error_json_path = output_dir / "error_analysis_report.json"
    markdown_path = docs_dir / "evaluation_report.md"

    # 1. Save evaluation_report.json
    eval_payload = {
        "report_title": "Model Evaluation Report",
        "phase": "Phase 7 - Model Evaluation and Error Analysis",
        "architecture": "EfficientNet-B0",
        "evaluation_metrics": eval_stats,
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    # 2. Save error_analysis_report.json
    with open(error_json_path, "w", encoding="utf-8") as f:
        json.dump(error_stats, f, indent=2)

    # 3. Generate docs/evaluation_report.md
    lines = [
        "# EfficientNet-B0 Model Evaluation & Error Analysis Report",
        "",
        "**Phase**: Phase 7 - Model Evaluation and Error Analysis  ",
        "**Architecture**: EfficientNet-B0 Transfer Learning Backbone  ",
        f"**Test Set Samples Inspected**: `{eval_stats.get('total_test_samples', 0)}`  ",
        "",
        "## 📊 Overall Performance Metrics",
        "",
        f"- **Top-1 Accuracy**: `{eval_stats.get('accuracy', 0.0)}%`",
        f"- **Top-3 Accuracy**: `{eval_stats.get('top_3_accuracy', 0.0)}%`",
        f"- **Precision (Macro)**: `{eval_stats.get('precision_macro', 0.0)}%`",
        f"- **Recall (Macro)**: `{eval_stats.get('recall_macro', 0.0)}%`",
        f"- **F1-Score (Macro)**: `{eval_stats.get('f1_macro', 0.0)}%`",
        f"- **F1-Score (Weighted)**: `{eval_stats.get('f1_weighted', 0.0)}%`",
        f"- **Test Loss**: `{eval_stats.get('test_loss', 0.0)}`",
        "",
        "## 🐂 Per-Class Breed Performance Breakdown",
        "",
        "| Class ID | Breed Name | Precision (%) | Recall (%) | F1-Score (%) | Support |",
        "|---|---|---|---|---|---|",
    ]

    per_class = eval_stats.get("per_class_metrics", {})
    for breed, m in per_class.items():
        lines.append(
            f"| {m.get('class_id', 0)} | {breed} | {m.get('precision', 0.0)} | {m.get('recall', 0.0)} | {m.get('f1_score', 0.0)} | {m.get('support', 0)} |"
        )

    lines.extend(
        [
            "",
            "## 🔍 Error Analysis & Breed Confusion",
            "",
            f"- **Total Misclassifications**: `{error_stats.get('total_errors', 0)}`",
            f"- **Error Rate**: `{error_stats.get('error_rate_percent', 0.0)}%`",
            f"- **Low Confidence Predictions (<0.50)**: `{error_stats.get('total_low_confidence_samples', 0)}`",
            "",
            "### Top Breed Confusion Pairs",
            "",
        ]
    )

    conf_pairs = error_stats.get("top_breed_confusion_pairs", [])
    if conf_pairs:
        for pair in conf_pairs[:5]:
            lines.append(
                f"- **{pair['true_breed']}** misclassified as **{pair['predicted_breed']}**: `{pair['misclassified_count']}` instances"
            )
    else:
        lines.append("- No breed confusion pairs detected.")

    lines.append("")

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 4. Log metrics to MLflow if available
    try:
        import mlflow

        uri = settings.MLFLOW_TRACKING_URI
        if not uri or uri == "http://localhost:5000":
            uri = "file:./mlruns"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("breed-classification-evaluation")
        with mlflow.start_run(run_name="test-set-evaluation-run"):
            mlflow.log_metrics(
                {
                    "test_loss": eval_stats.get("test_loss", 0.0),
                    "test_accuracy": eval_stats.get("accuracy", 0.0),
                    "test_top3_acc": eval_stats.get("top_3_accuracy", 0.0),
                    "test_f1_macro": eval_stats.get("f1_macro", 0.0),
                    "test_f1_weighted": eval_stats.get("f1_weighted", 0.0),
                    "test_errors_count": error_stats.get("total_errors", 0),
                }
            )
            plot_path = docs_dir / "confusion_matrix.png"
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path))
    except Exception:
        pass

    return eval_json_path, error_json_path, markdown_path
