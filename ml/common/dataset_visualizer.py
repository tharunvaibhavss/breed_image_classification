"""Dataset Visualization module for generating distribution charts and visual summary reports.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from ml.common.dataset_inspector import DatasetInspector


def generate_visualization_plots(
    inspector: DatasetInspector, output_dir: Path
) -> Optional[Path]:
    """Generate dataset distribution and dimension plots saved as PNG image.

    Args:
        inspector: DatasetInspector instance with generated statistics.
        output_dir: Output directory path.

    Returns:
        Path to generated plot image or None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "dataset_visualization.png"

    summary = inspector.generate_summary_statistics()
    breed_counts: Dict[str, int] = summary.get("images_per_breed", {})

    # Set aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Bar Chart: Samples per Breed
    breeds = list(breed_counts.keys()) if breed_counts else ["No Images"]
    counts = list(breed_counts.values()) if breed_counts else [0]

    bars = axes[0].bar(breeds, counts, color="#2b5c8f", edgecolor="#1b365d")
    axes[0].set_title("Sample Count per Breed", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Breed Name")
    axes[0].set_ylabel("Number of Images")
    axes[0].tick_params(axis="x", rotation=30)

    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # 2. Scatter Plot: Image Dimensions (Width vs Height)
    widths = [m.width for m in inspector.images_metadata if m.width > 0]
    heights = [m.height for m in inspector.images_metadata if m.height > 0]

    if widths and heights:
        axes[1].scatter(widths, heights, alpha=0.7, color="#d95f02", edgecolors="w")
        axes[1].set_title("Image Dimension Distribution", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Width (pixels)")
        axes[1].set_ylabel("Height (pixels)")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No Image Dimensions Available\n(Dataset Empty or Unprocessed)",
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        axes[1].set_title("Image Dimension Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def generate_markdown_visual_report(
    inspector: DatasetInspector, output_dir: Path
) -> Path:
    """Generate Markdown visual summary report dataset_report.md.

    Args:
        inspector: Inspected DatasetInspector instance.
        output_dir: Directory path to save report.

    Returns:
        Path to generated dataset_report.md file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_report.md"

    summary = inspector.generate_summary_statistics()
    dim_stats = summary.get("dimension_statistics", {})
    dup_stats = summary.get("duplicate_analysis", {})

    lines = [
        "# Dataset Inspection & Analysis Visual Report",
        "",
        "**Phase**: Phase 1 - Dataset Inspection and Organization  ",
        f"**Dataset Location**: `{inspector.dataset_dir}`  ",
        "",
        "## 📊 Executive Summary Statistics",
        "",
        f"- **Total Images**: `{summary.get('total_images', 0)}`",
        f"- **Valid Readable Images**: `{summary.get('valid_images', 0)}`",
        f"- **Corrupted Images**: `{summary.get('corrupted_images', 0)}`",
        f"- **Unreadable Images**: `{summary.get('unreadable_images', 0)}`",
        f"- **Zero-Byte Files**: `{summary.get('zero_byte_files', 0)}`",
        f"- **Missing Label Files**: `{summary.get('missing_label_files_count', 0)}`",
        f"- **Unexpected Files**: `{summary.get('unexpected_files_count', 0)}`",
        "",
        "## 🐂 Class Distribution (Images per Breed)",
        "",
        "| Animal Type | Breed Name | Image Count |",
        "|---|---|---|",
    ]

    breed_counts = summary.get("images_per_breed", {})
    if breed_counts:
        for breed, count in breed_counts.items():
            info = inspector.registry.get_by_name(breed)
            animal = info.animal_type if info else "unknown"
            lines.append(f"| {animal.capitalize()} | {breed} | {count} |")
    else:
        lines.append("| N/A | No breeds found | 0 |")

    lines.extend(
        [
            "",
            "## 📐 Image Dimension & Size Statistics",
            "",
            f"- **Width**: Min `{dim_stats.get('min_width', 0)}px`, Max `{dim_stats.get('max_width', 0)}px`, Mean `{dim_stats.get('mean_width', 0)}px`, Median `{dim_stats.get('median_width', 0)}px`",
            f"- **Height**: Min `{dim_stats.get('min_height', 0)}px`, Max `{dim_stats.get('max_height', 0)}px`, Mean `{dim_stats.get('mean_height', 0)}px`, Median `{dim_stats.get('median_height', 0)}px`",
            f"- **Aspect Ratio (W/H)**: Min `{dim_stats.get('min_aspect_ratio', 0)}`, Max `{dim_stats.get('max_aspect_ratio', 0)}`, Mean `{dim_stats.get('mean_aspect_ratio', 0)}`",
            f"- **File Size**: Min `{dim_stats.get('min_file_size_bytes', 0)} bytes`, Max `{dim_stats.get('max_file_size_bytes', 0)} bytes`, Mean `{dim_stats.get('mean_file_size_bytes', 0)} bytes`",
            "",
            "## 🔍 Duplicate & Augmented Image Analysis",
            "",
            f"- **Exact Hash Duplicate Groups**: `{dup_stats.get('total_exact_duplicate_groups', 0)}`",
            f"- **Exact Duplicate Files Count**: `{dup_stats.get('total_exact_duplicate_files', 0)}`",
            f"- **Files with 'aug_' Filename Pattern**: `{dup_stats.get('total_augmented_filename_matches', 0)}`",
            f"- **Perceptual Similar Image Groups (dHash)**: `{dup_stats.get('total_perceptual_similar_groups', 0)}`",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path
