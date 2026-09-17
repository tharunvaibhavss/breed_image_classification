"""Dataset Validation module for detecting image quality issues and resolution thresholds.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ml.common.dataset_inspector import DatasetInspector, ImageMetadata

MIN_WIDTH = 64
MIN_HEIGHT = 64


class ValidationResult:
    """Class representing image validation status."""

    def __init__(
        self,
        metadata: ImageMetadata,
        quality_status: str,  # 'valid', 'corrupted', 'unreadable', 'zero_byte', 'too_small', 'invalid_label'
        validation_notes: List[str],
    ):
        self.metadata = metadata
        self.quality_status = quality_status
        self.validation_notes = validation_notes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize validation result."""
        d = self.metadata.to_dict()
        d["quality_status"] = self.quality_status
        d["validation_notes"] = "; ".join(self.validation_notes) if self.validation_notes else "Pass"
        return d


class DatasetValidator:
    """Validator engine inspecting images for defects, resolution bounds, and metadata integrity."""

    def __init__(
        self,
        min_width: int = MIN_WIDTH,
        min_height: int = MIN_HEIGHT,
    ):
        self.min_width = min_width
        self.min_height = min_height

    def validate_image(self, meta: ImageMetadata) -> ValidationResult:
        """Validate an individual image metadata object against quality rules.

        Args:
            meta: ImageMetadata instance.

        Returns:
            ValidationResult instance.
        """
        notes: List[str] = []

        if meta.is_zero_byte:
            return ValidationResult(
                metadata=meta,
                quality_status="zero_byte",
                validation_notes=["Zero-byte file size"],
            )

        if meta.is_unreadable:
            return ValidationResult(
                metadata=meta,
                quality_status="unreadable",
                validation_notes=[meta.error_message or "Unreadable file"],
            )

        if meta.is_corrupted:
            return ValidationResult(
                metadata=meta,
                quality_status="corrupted",
                validation_notes=[meta.error_message or "Corrupted image payload"],
            )

        # Check resolution thresholds
        if meta.width < self.min_width or meta.height < self.min_height:
            notes.append(
                f"Image resolution ({meta.width}x{meta.height}) below minimum threshold ({self.min_width}x{self.min_height})"
            )
            return ValidationResult(
                metadata=meta,
                quality_status="too_small",
                validation_notes=notes,
            )

        # Check breed label validity
        if meta.breed_name == "unknown" or meta.animal_type == "unknown":
            notes.append("Invalid or unassigned breed/animal label")
            return ValidationResult(
                metadata=meta,
                quality_status="invalid_label",
                validation_notes=notes,
            )

        return ValidationResult(
            metadata=meta,
            quality_status="valid",
            validation_notes=[],
        )

    def validate_dataset(
        self, inspector: DatasetInspector
    ) -> List[ValidationResult]:
        """Validate all images discovered by DatasetInspector."""
        if not inspector.images_metadata:
            inspector.scan()

        results: List[ValidationResult] = []
        for meta in inspector.images_metadata:
            results.append(self.validate_image(meta))
        return results

    def generate_validation_reports(
        self, results: List[ValidationResult], output_dir: Path
    ) -> Tuple[Path, Path]:
        """Export validation_report.json and validation_report.csv."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "validation_report.json"
        csv_path = output_dir / "validation_report.csv"

        # Tally validation status counts
        status_counts: Dict[str, int] = {}
        for r in results:
            status_counts[r.quality_status] = status_counts.get(r.quality_status, 0) + 1

        json_data = {
            "validation_title": "Dataset Quality & Validation Report",
            "phase": "Phase 2 - Dataset Validation",
            "total_files_inspected": len(results),
            "quality_status_counts": status_counts,
            "min_resolution_threshold": {
                "min_width": self.min_width,
                "min_height": self.min_height,
            },
            "validation_details": [r.to_dict() for r in results],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        if results:
            csv_headers = list(results[0].to_dict().keys())
        else:
            csv_headers = ["relative_path", "quality_status", "validation_notes"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())

        return json_path, csv_path
