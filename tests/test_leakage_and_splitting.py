"""Unit tests verifying zero data leakage across splits and split reproducibility."""

import numpy as np
from PIL import Image
from pathlib import Path

from ml.common.dataset_inspector import DatasetInspector
from ml.preprocessing.leakage_analyzer import LeakageAnalyzer
from ml.preprocessing.group_splitter import GroupAwareSplitter


def test_zero_data_leakage_across_splits(tmp_path: Path):
    """Verify that images in the same source_group NEVER leak across train/val/test splits."""
    gir_dir = tmp_path / "cattle" / "Gir"
    gir_dir.mkdir(parents=True)

    # Create original and augmented file
    img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
    img.save(gir_dir / "gir_10.jpg")
    img.save(gir_dir / "aug_gir_10.jpg")  # Same stem & contents

    inspector = DatasetInspector(dataset_dir=tmp_path)
    inspector.scan()

    analyzer = LeakageAnalyzer()
    grouped_records = analyzer.analyze_and_group(inspector.images_metadata)

    # Check that both images were assigned the exact same source_group_id
    group_ids = {r.source_group_id for r in grouped_records}
    assert len(group_ids) == 1

    # Perform group-aware split
    splitter = GroupAwareSplitter(seed=42)
    split_records = splitter.split_dataset(grouped_records)

    # Check that both images landed in the EXACT SAME split
    assigned_splits = {sr.split for sr in split_records}
    assert len(assigned_splits) == 1, "Data leakage detected! Same source group split across different splits."


def test_split_reproducibility(tmp_path: Path):
    """Verify that using the same random seed yields 100% identical group splits."""
    gir_dir = tmp_path / "cattle" / "Gir"
    gir_dir.mkdir(parents=True)

    # Create 10 dummy images
    for i in range(10):
        img = Image.fromarray(np.ones((40, 40, 3), dtype=np.uint8) * (i * 20))
        img.save(gir_dir / f"gir_{i:02d}.jpg")

    inspector = DatasetInspector(dataset_dir=tmp_path)
    inspector.scan()

    analyzer = LeakageAnalyzer()
    grouped_records = analyzer.analyze_and_group(inspector.images_metadata)

    # Split 1 with seed=42
    splitter_1 = GroupAwareSplitter(seed=42)
    split_1 = {sr.record.metadata.relative_path: sr.split for sr in splitter_1.split_dataset(grouped_records)}

    # Split 2 with seed=42
    splitter_2 = GroupAwareSplitter(seed=42)
    split_2 = {sr.record.metadata.relative_path: sr.split for sr in splitter_2.split_dataset(grouped_records)}

    assert split_1 == split_2, "Split non-reproducible with fixed seed!"
