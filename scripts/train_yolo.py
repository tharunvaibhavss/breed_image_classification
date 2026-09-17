#!/usr/bin/env python3
"""CLI script to verify bounding box annotations and run YOLO animal detection training."""

import sys
import os
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.detection.annotation_verifier import AnnotationVerifier
from ml.detection.yolo_trainer import YOLOModelTrainer


def main():
    """Verify annotations and run YOLO animal detection training if annotations exist."""
    project_root = Path(__file__).resolve().parent.parent
    annotation_dir = project_root / "data" / "annotations"
    config_path = annotation_dir / "yolo" / "yolo_config.yaml"
    models_dir = project_root / "models"

    print("=" * 70)
    print(" PHASE 5 - YOLO ANIMAL DETECTION VERIFICATION & TRAINING")
    print("=" * 70)
    print(f" Annotation Directory : {annotation_dir}")
    print(f" YOLO Dataset Config  : {config_path}")
    print("-" * 70)

    # 1. Verify Bounding Box Annotations
    print("[1/2] Verifying bounding box annotations...")
    verifier = AnnotationVerifier(annotation_dir=annotation_dir)
    verification_report = verifier.verify_annotations()

    print("=" * 70)
    print(" ANNOTATION AUDIT REPORT")
    print("=" * 70)
    print(f" Total Images Found         : {verification_report['total_images']}")
    print(f" Total Label Files Found    : {verification_report['total_label_files']}")
    print(f" Total Valid Bounding Boxes : {verification_report['total_bounding_boxes']}")
    print(f" Missing Annotations Count  : {verification_report['missing_annotations_count']}")
    print(f" Invalid Label Lines Count  : {verification_report['invalid_label_lines_count']}")
    print("-" * 70)
    print(" SPLIT BREAKDOWN:")
    for split_name, stats in verification_report["split_breakdown"].items():
        print(
            f"   - {split_name.capitalize():<6}: {stats['images_count']} images, "
            f"{stats['label_files_count']} label files, "
            f"{stats['bounding_boxes_count']} boxes, "
            f"{stats['missing_labels_count']} missing labels"
        )
    print("=" * 70)

    # Check readiness for training
    if not verification_report["is_ready_for_training"]:
        print("\n[STOP] TRAINING STOPPED: Bounding-box annotations are incomplete or missing.")
        print("-" * 70)
        print("EXACT MISSING/INVALID REASONS:")

        if verification_report["total_images"] == 0:
            print(" - Zero image files found in data/annotations/yolo/images/")

        if verification_report["total_bounding_boxes"] == 0:
            print(" - Zero valid bounding box label files found in data/annotations/yolo/labels/")

        if verification_report["missing_annotations_count"] > 0:
            print(f" - {verification_report['missing_annotations_count']} image files are missing corresponding .txt label files.")

        if verification_report["invalid_label_lines_count"] > 0:
            print(f" - {verification_report['invalid_label_lines_count']} label lines have invalid syntax or coordinate values.")
            for err in verification_report["invalid_label_lines"][:5]:
                print(f"     * {err}")

        print("-" * 70)
        print("INSTRUCTIONS:")
        print(" 1. Please refer to docs/yolo_annotation_guide.md for annotation rules.")
        print(" 2. Add valid bounding box .txt label files to data/annotations/yolo/labels/ (train/val/test).")
        print(" 3. Re-run python scripts/train_yolo.py once annotations exist.")
        print("=" * 70)
        sys.exit(0)

    # 2. If annotations are valid, proceed with YOLO training
    print("[2/2] Annotations verified! Launching YOLOv8 animal detection training...")
    trainer = YOLOModelTrainer(config_path=config_path, output_dir=models_dir)
    res = trainer.train(epochs=50, batch_size=16, imgsz=640)

    print("=" * 70)
    print(" TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f" Best Model Checkpoint : {res['best_model_path']}")
    print(f" Last Model Checkpoint : {res['last_model_path']}")
    print(f" MLflow Run ID         : {res['mlflow_run_id']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
