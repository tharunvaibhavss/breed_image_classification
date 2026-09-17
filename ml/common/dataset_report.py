"""Dataset Report Generator module.

Generates dataset_report.json and dataset_report.csv summary and per-image audit reports.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Tuple
from ml.common.dataset_inspector import DatasetInspector


def generate_dataset_reports(
    inspector: DatasetInspector, output_dir: Path
) -> Tuple[Path, Path]:
    """Generate dataset_report.json and dataset_report.csv in output_dir.

    Args:
        inspector: Inspected DatasetInspector instance containing images_metadata.
        output_dir: Output directory path (e.g. data/processed or docs).

    Returns:
        Tuple of (json_file_path, csv_file_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_stats = inspector.generate_summary_statistics()
    breed_registry_dict = inspector.registry.to_dict()

    json_path = output_dir / "dataset_report.json"
    csv_path = output_dir / "dataset_report.csv"

    # 1. Write dataset_report.json
    full_json_payload: Dict[str, Any] = {
        "report_title": "Dataset Inspection & Analysis Report",
        "phase": "Phase 1 - Dataset Inspection and Organization",
        "dataset_directory": str(inspector.dataset_dir),
        "breed_registry": breed_registry_dict,
        "summary_statistics": summary_stats,
        "images_metadata": [meta.to_dict() for meta in inspector.images_metadata],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_payload, f, indent=2)

    # 2. Write dataset_report.csv
    csv_headers = [
        "relative_path",
        "filename",
        "animal_type",
        "breed_name",
        "width",
        "height",
        "channels",
        "aspect_ratio",
        "format",
        "file_size_bytes",
        "is_corrupted",
        "is_unreadable",
        "is_zero_byte",
        "is_augmented_filename",
        "md5_hash",
        "sha256_hash",
        "dhash",
        "error_message",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for meta in inspector.images_metadata:
            writer.writerow(meta.to_dict())

    return json_path, csv_path
