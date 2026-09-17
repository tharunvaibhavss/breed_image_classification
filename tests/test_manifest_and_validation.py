"""Unit tests for dataset validation rules and unified manifest creation."""

import numpy as np
from PIL import Image
from pathlib import Path

from ml.common.dataset_inspector import DatasetInspector
from ml.preprocessing.dataset_validator import DatasetValidator
from ml.preprocessing.leakage_analyzer import LeakageAnalyzer
from ml.preprocessing.group_splitter import GroupAwareSplitter
from ml.preprocessing.manifest_manager import DatasetManifestManager


def test_small_image_resolution_validation(tmp_path: Path):
    """Verify that images smaller than 64x64 are flagged as 'too_small'."""
    gir_dir = tmp_path / "cattle" / "Gir"
    gir_dir.mkdir(parents=True)

    # 32x32 image (too small)
    small_img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    small_img.save(gir_dir / "small.jpg")

    # 100x100 image (valid)
    valid_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    valid_img.save(gir_dir / "normal.jpg")

    inspector = DatasetInspector(dataset_dir=tmp_path)
    inspector.scan()

    validator = DatasetValidator(min_width=64, min_height=64)
    results = validator.validate_dataset(inspector)

    small_res = [r for r in results if r.metadata.file_path.name == "small.jpg"][0]
    normal_res = [r for r in results if r.metadata.file_path.name == "normal.jpg"][0]

    assert small_res.quality_status == "too_small"
    assert normal_res.quality_status == "valid"


def test_manifest_building_and_export(tmp_path: Path):
    """Verify building and exporting unified dataset manifest."""
    gir_dir = tmp_path / "cattle" / "Gir"
    gir_dir.mkdir(parents=True)

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img.save(gir_dir / "gir_01.jpg")

    inspector = DatasetInspector(dataset_dir=tmp_path)
    inspector.scan()

    validator = DatasetValidator()
    val_results = validator.validate_dataset(inspector)

    analyzer = LeakageAnalyzer()
    grouped_records = analyzer.analyze_and_group(inspector.images_metadata)

    splitter = GroupAwareSplitter(seed=42)
    split_records = splitter.split_dataset(grouped_records)

    manifest_mgr = DatasetManifestManager()
    manifest_records = manifest_mgr.build_manifest(split_records, val_results)

    assert len(manifest_records) == 1
    m = manifest_records[0]
    assert m.image_id == "IMG_000001"
    assert m.breed == "Gir"
    assert m.animal_type == "cattle"
    assert m.split in ["train", "val", "test"]
    assert m.quality_status == "valid"

    json_p, csv_p = manifest_mgr.export_manifest(manifest_records, "dataset_v001", tmp_path)
    assert json_p.exists()
    assert csv_p.exists()
