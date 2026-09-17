"""Unit tests for image validation, zero-byte detection, and corruption handling."""

from pathlib import Path
from ml.common.dataset_inspector import DatasetInspector


def test_zero_byte_file_detection(tmp_path: Path):
    """Verify zero-byte files are flagged as corrupted and zero-byte."""
    breed_dir = tmp_path / "cattle" / "Ongole"
    breed_dir.mkdir(parents=True)

    zero_file = breed_dir / "corrupted_zero.jpg"
    zero_file.write_bytes(b"")

    inspector = DatasetInspector(dataset_dir=tmp_path)
    metadata = inspector.scan()

    assert len(metadata) == 1
    assert metadata[0].is_zero_byte is True
    assert metadata[0].is_corrupted is True


def test_corrupted_payload_detection(tmp_path: Path):
    """Verify files with random non-image binary content are caught as corrupted."""
    breed_dir = tmp_path / "buffalo" / "Murrah"
    breed_dir.mkdir(parents=True)

    corrupted_file = breed_dir / "invalid_image.png"
    corrupted_file.write_bytes(b"NOT_A_VALID_IMAGE_HEADER_1234567890")

    inspector = DatasetInspector(dataset_dir=tmp_path)
    metadata = inspector.scan()

    assert len(metadata) == 1
    assert metadata[0].is_corrupted is True
