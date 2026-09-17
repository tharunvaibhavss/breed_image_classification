"""Unit tests for exact duplicate detection and aug_ file pattern matching."""

import numpy as np
from PIL import Image
from pathlib import Path

from ml.common.dataset_inspector import DatasetInspector


def test_exact_hash_duplicate_detection(tmp_path: Path):
    """Verify exact duplicate files are grouped by SHA-256 hash."""
    sahiwal_dir = tmp_path / "cattle" / "Sahiwal"
    sahiwal_dir.mkdir(parents=True)

    img_data = np.ones((50, 50, 3), dtype=np.uint8) * 128
    original_path = sahiwal_dir / "sahiwal_orig.png"
    copy_path = sahiwal_dir / "sahiwal_copy.png"

    Image.fromarray(img_data).save(original_path)
    Image.fromarray(img_data).save(copy_path)

    inspector = DatasetInspector(dataset_dir=tmp_path)
    inspector.scan()
    dup_analysis = inspector.analyze_duplicates()

    assert dup_analysis["total_exact_duplicate_groups"] == 1
    assert dup_analysis["total_exact_duplicate_files"] == 1


def test_augmented_filename_pattern_detection(tmp_path: Path):
    """Verify files with aug_ pattern in filename are identified."""
    surti_dir = tmp_path / "buffalo" / "Surti"
    surti_dir.mkdir(parents=True)

    img = Image.fromarray(np.zeros((30, 30, 3), dtype=np.uint8))
    aug_file = surti_dir / "aug_surti_01.jpg"
    img.save(aug_file)

    inspector = DatasetInspector(dataset_dir=tmp_path)
    inspector.scan()
    dup_analysis = inspector.analyze_duplicates()

    assert dup_analysis["total_augmented_filename_matches"] == 1
    assert "aug_surti_01.jpg" in dup_analysis["augmented_filename_files"][0]
