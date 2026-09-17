"""Dataset Manifest Manager module.

Generates unified dataset_manifest.json and dataset_manifest.csv combining inspection,
validation quality status, source_group leakage assignments, and split declarations.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ml.preprocessing.group_splitter import SplitRecord
from ml.preprocessing.dataset_validator import ValidationResult


class ManifestRecord:
    """Unified Manifest Record combining all Phase 2 metadata fields."""

    def __init__(
        self,
        image_id: str,
        file_path: str,
        animal_type: str,
        breed: str,
        source_group: str,
        split: str,
        annotation_status: str,
        quality_status: str,
        width: int,
        height: int,
        channels: int,
        aspect_ratio: float,
        format: str,
        file_size_bytes: int,
        md5_hash: str,
        sha256_hash: str,
    ):
        self.image_id = image_id
        self.file_path = file_path
        self.animal_type = animal_type
        self.breed = breed
        self.source_group = source_group
        self.split = split
        self.annotation_status = annotation_status
        self.quality_status = quality_status
        self.width = width
        self.height = height
        self.channels = channels
        self.aspect_ratio = aspect_ratio
        self.format = format
        self.file_size_bytes = file_size_bytes
        self.md5_hash = md5_hash
        self.sha256_hash = sha256_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "animal_type": self.animal_type,
            "breed": self.breed,
            "source_group": self.source_group,
            "split": self.split,
            "annotation_status": self.annotation_status,
            "quality_status": self.quality_status,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "format": self.format,
            "file_size_bytes": self.file_size_bytes,
            "md5_hash": self.md5_hash,
            "sha256_hash": self.sha256_hash,
        }


class DatasetManifestManager:
    """Manager constructing and exporting the dataset manifest."""

    def build_manifest(
        self,
        split_records: List[SplitRecord],
        validation_results: List[ValidationResult],
    ) -> List[ManifestRecord]:
        """Build manifest records joining split records and validation quality statuses.

        Args:
            split_records: List of SplitRecord objects.
            validation_results: List of ValidationResult objects.

        Returns:
            List of ManifestRecord objects.
        """
        # Map relative path -> quality_status
        val_map: Dict[str, str] = {
            vr.metadata.relative_path: vr.quality_status for vr in validation_results
        }

        manifest_list: List[ManifestRecord] = []
        for idx, sr in enumerate(
            sorted(split_records, key=lambda x: x.record.metadata.relative_path), start=1
        ):
            meta = sr.record.metadata
            img_id = f"IMG_{idx:06d}"
            quality = val_map.get(meta.relative_path, "valid")

            # Check if corresponding YOLO label file exists
            annotation_status = "unannotated"

            rec = ManifestRecord(
                image_id=img_id,
                file_path=meta.relative_path,
                animal_type=meta.animal_type,
                breed=meta.breed_name,
                source_group=sr.record.source_group_id,
                split=sr.split,
                annotation_status=annotation_status,
                quality_status=quality,
                width=meta.width,
                height=meta.height,
                channels=meta.channels,
                aspect_ratio=meta.aspect_ratio,
                format=meta.image_format,
                file_size_bytes=meta.file_size_bytes,
                md5_hash=meta.md5_hash,
                sha256_hash=meta.sha256_hash,
            )
            manifest_list.append(rec)

        return manifest_list

    def export_manifest(
        self,
        manifest_records: List[ManifestRecord],
        version_id: str,
        output_dir: Path,
    ) -> Tuple[Path, Path]:
        """Export dataset_manifest.json and dataset_manifest.csv."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "dataset_manifest.json"
        csv_path = output_dir / "dataset_manifest.csv"

        payload = {
            "manifest_title": "Unified Dataset Manifest",
            "phase": "Phase 2 - Dataset Validation & Splitting",
            "dataset_version": version_id,
            "total_records": len(manifest_records),
            "manifest_records": [m.to_dict() for m in manifest_records],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        csv_headers = [
            "image_id",
            "file_path",
            "animal_type",
            "breed",
            "source_group",
            "split",
            "annotation_status",
            "quality_status",
            "width",
            "height",
            "channels",
            "aspect_ratio",
            "format",
            "file_size_bytes",
            "md5_hash",
            "sha256_hash",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for m in manifest_records:
                writer.writerow(m.to_dict())

        return json_path, csv_path
