#!/usr/bin/env python3
"""CLI script to run complete Phase 2 dataset validation, leakage grouping, splitting, and manifest creation."""

import sys
import os
import json
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.common.breed_registry import BreedRegistry
from ml.common.dataset_inspector import DatasetInspector
from ml.preprocessing.dataset_validator import DatasetValidator
from ml.preprocessing.leakage_analyzer import LeakageAnalyzer
from ml.preprocessing.group_splitter import GroupAwareSplitter
from ml.preprocessing.yolo_annotator import YOLOAnnotatorSetup
from ml.preprocessing.manifest_manager import DatasetManifestManager


def main():
    """Execute complete Phase 2 dataset validation, leakage analysis, and splitting workflow."""
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "data" / "raw" / "dataset"
    processed_dir = project_root / "data" / "processed"
    annotations_dir = project_root / "data" / "annotations"
    docs_dir = project_root / "docs"

    print("=" * 70)
    print(" PHASE 2 - DATASET VALIDATION, LEAKAGE ANALYSIS & SPLITTING")
    print("=" * 70)
    print(f" Target Dataset Directory : {dataset_dir}")
    print(f" Output Processed Dir    : {processed_dir}")
    print(f" Output Annotations Dir  : {annotations_dir}")
    print("-" * 70)

    # 1. Dataset Inspection
    print("[1/6] Scanning raw dataset...")
    registry = BreedRegistry()
    inspector = DatasetInspector(dataset_dir=dataset_dir, registry=registry)
    inspector.scan()

    # 2. Data Validation
    print("[2/6] Validating image quality, resolution bounds, and formats...")
    validator = DatasetValidator(min_width=64, min_height=64)
    val_results = validator.validate_dataset(inspector)
    val_json, val_csv = validator.generate_validation_reports(val_results, processed_dir)

    # 3. Leakage Analysis & Grouping
    print("[3/6] Analyzing file hashes, dHash, and aug_ patterns for source grouping...")
    analyzer = LeakageAnalyzer()
    grouped_records = analyzer.analyze_and_group(inspector.images_metadata)

    # 4. Group-Aware Train/Val/Test Splitting
    print("[4/6] Performing group-aware 70/15/15 train/val/test split...")
    splitter = GroupAwareSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    split_records = splitter.split_dataset(grouped_records)
    balance_info = splitter.calculate_class_balance_and_weights(split_records)

    # 5. YOLO Detection Structure Setup
    print("[5/6] Initializing YOLO annotation directory structure & yolo_config.yaml...")
    yolo_setup = YOLOAnnotatorSetup(base_annotation_dir=annotations_dir)
    yolo_setup.setup_directories()
    yolo_yaml_path = yolo_setup.create_dataset_yaml(project_root)

    # 6. Unified Manifest Generation
    print("[6/6] Generating unified dataset manifest...")
    manifest_mgr = DatasetManifestManager()
    manifest_records = manifest_mgr.build_manifest(split_records, val_results)
    manifest_json, manifest_csv = manifest_mgr.export_manifest(
        manifest_records, version_id="dataset_v001", output_dir=processed_dir
    )
    # Also export to docs for accessibility
    manifest_mgr.export_manifest(
        manifest_records, version_id="dataset_v001", output_dir=docs_dir
    )

    print("=" * 70)
    print(" PHASE 2 WORKFLOW SUMMARY REPORT")
    print("=" * 70)
    print(f" Dataset Version Identified     : dataset_v001")
    print(f" Total Images Processed          : {len(manifest_records)}")
    print(f" Total Unique Source Groups      : {len(set(r.source_group for r in manifest_records))}")
    print("-" * 70)
    print(" SPLIT DISTRIBUTION (Group-Aware):")
    split_counts: dict = {}
    for r in manifest_records:
        split_counts[r.split] = split_counts.get(r.split, 0) + 1
    for s_name, count in split_counts.items():
        print(f"   - {s_name.capitalize():<10}: {count} images")
    print("-" * 70)
    print(" TRAINING INVERSE CLASS FREQUENCY WEIGHTS:")
    for breed, weight in balance_info["train_class_weights_inverse_freq"].items():
        print(f"   - {breed:<18}: {weight}")
    print("=" * 70)
    print(f" Validation JSON Exported : {val_json}")
    print(f" Manifest JSON Exported   : {manifest_json}")
    print(f" Manifest CSV Exported    : {manifest_csv}")
    print(f" YOLO Config Generated    : {yolo_yaml_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
