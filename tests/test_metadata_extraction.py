"""Unit tests for image metadata extraction (dimensions, channels, format, aspect ratio)."""

import numpy as np
from PIL import Image
from pathlib import Path

from ml.common.dataset_inspector import DatasetInspector


def test_metadata_extraction_accuracy(tmp_path: Path):
    """Verify width, height, channels, aspect ratio, and format extraction."""
    breed_dir = tmp_path / "buffalo" / "Jaffarabadi"
    breed_dir.mkdir(parents=True)

    # 120 width x 60 height RGB image (aspect ratio 2.0)
    arr = np.random.randint(0, 255, (60, 120, 3), dtype=np.uint8)
    img_path = breed_dir / "sample.jpg"
    Image.fromarray(arr).save(img_path, format="JPEG")

    inspector = DatasetInspector(dataset_dir=tmp_path)
    metadata = inspector.scan()

    assert len(metadata) == 1
    m = metadata[0]
    assert m.width == 120
    assert m.height == 60
    assert m.channels == 3
    assert m.aspect_ratio == 2.0
    assert m.image_format == "JPEG"
    assert m.file_size_bytes > 0
    assert len(m.md5_hash) == 32
    assert len(m.sha256_hash) == 64
