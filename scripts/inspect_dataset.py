#!/usr/bin/env python3
"""CLI script to perform dataset inspection, duplicate analysis, report generation, and visualization."""

import sys
import os
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.common.breed_registry import BreedRegistry
from ml.common.dataset_inspector import DatasetInspector
from ml.common.dataset_report import generate_dataset_reports
from ml.common.dataset_visualizer import (
    generate_visualization_plots,
    generate_markdown_visual_report,
)


def main():
    """Execute complete Phase 1 dataset inspection and report generation workflow."""
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "data" / "raw" / "dataset"
    processed_dir = project_root / "data" / "processed"
    docs_dir = project_root / "docs"

    print("=" * 70)
    print(" PHASE 1 - DATASET INSPECTION & ORGANIZATION")
    print("=" * 70)
    print(f" Target Dataset Directory : {dataset_dir}")
    print(f" Output Processed Dir    : {processed_dir}")
    print(f" Output Documentation Dir: {docs_dir}")
    print("-" * 70)

    # 1. Initialize Registry & Inspector
    registry = BreedRegistry()
    inspector = DatasetInspector(dataset_dir=dataset_dir, registry=registry)

    # 2. Perform non-destructive scan
    print("[1/4] Scanning raw dataset files (non-destructive)...")
    inspector.scan()

    # 3. Generate summary statistics
    print("[2/4] Analyzing image metadata, hashes, and duplicates...")
    summary = inspector.generate_summary_statistics()

    # 4. Generate JSON and CSV Reports
    print("[3/4] Exporting dataset_report.json and dataset_report.csv...")
    json_path, csv_path = generate_dataset_reports(inspector, processed_dir)
    # Also copy to docs for documentation access
    generate_dataset_reports(inspector, docs_dir)

    # 5. Generate Visualizations & Reports
    print("[4/4] Rendering visualization plots and markdown report...")
    plot_path = generate_visualization_plots(inspector, docs_dir)
    markdown_path = generate_markdown_visual_report(inspector, docs_dir)

    print("=" * 70)
    print(" DATASET SUMMARY STATISTICS REPORT")
    print("=" * 70)
    print(f" Total Images Found             : {summary['total_images']}")
    print(f" Valid Readable Images          : {summary['valid_images']}")
    print(f" Corrupted Images               : {summary['corrupted_images']}")
    print(f" Unreadable Files               : {summary['unreadable_images']}")
    print(f" Zero-Byte Files                : {summary['zero_byte_files']}")
    print(f" Unexpected Files Count         : {summary['unexpected_files_count']}")
    print(f" Missing Label Files Count      : {summary['missing_label_files_count']}")
    print("-" * 70)
    print(" IMAGES PER ANIMAL TYPE:")
    for animal, count in summary["images_per_animal_type"].items():
        print(f"   - {animal.capitalize():<15}: {count}")
    print("-" * 70)
    print(" IMAGES PER BREED:")
    for breed, count in summary["images_per_breed"].items():
        print(f"   - {breed:<18}: {count}")
    print("-" * 70)
    print(" DUPLICATE & AUGMENTATION FINDINGS:")
    dup = summary["duplicate_analysis"]
    print(f"   - Exact Duplicate Groups (SHA-256): {dup['total_exact_duplicate_groups']}")
    print(f"   - Exact Duplicate Files Count    : {dup['total_exact_duplicate_files']}")
    print(f"   - 'aug_' Filename Matches        : {dup['total_augmented_filename_matches']}")
    print(f"   - Perceptual Similar Groups (dHash): {dup['total_perceptual_similar_groups']}")
    print("=" * 70)
    print(f" JSON Report Exported : {json_path}")
    print(f" CSV Report Exported  : {csv_path}")
    print(f" Markdown Visual Doc : {markdown_path}")
    if plot_path:
        print(f" Plot Chart Rendered : {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
