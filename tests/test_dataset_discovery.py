"""Unit tests for dataset discovery and file traversal."""

import numpy as np
from PIL import Image
from pathlib import Path

from ml.common.dataset_inspector import DatasetInspector
from ml.common.breed_registry import BreedRegistry


def test_empty_dataset_directory_scan(tmp_path: Path):
    """Verify scanning an empty dataset directory yields 0 images."""
    inspector = DatasetInspector(dataset_dir=tmp_path)
    metadata = inspector.scan()
    assert len(metadata) == 0
    stats = inspector.generate_summary_statistics()
    assert stats["total_images"] == 0


def test_dataset_discovery_with_mock_structure(tmp_path: Path):
    """Verify discovery correctly assigns breed and animal_type from directory hierarchy."""
    gir_dir = tmp_path / "cattle" / "Gir"
    gir_dir.mkdir(parents=True)

    # Save dummy image
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = gir_dir / "gir_01.jpg"
    Image.fromarray(img_array).save(img_path)

    inspector = DatasetInspector(dataset_dir=tmp_path)
    metadata = inspector.scan()

    assert len(metadata) == 1
    assert metadata[0].breed_name == "Gir"
    assert metadata[0].animal_type == "cattle"
    assert metadata[0].width == 100
    assert metadata[0].height == 100
