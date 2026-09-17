"""Group-Aware Train/Val/Test Splitter with Class Weighting support.

Performs deterministic, leakage-free dataset splitting grouped by source_group_id.
"""

import random
from typing import Dict, List, Any, Tuple
from ml.preprocessing.leakage_analyzer import GroupedImageRecord


class SplitRecord:
    """Record linking GroupedImageRecord to its assigned dataset split."""

    def __init__(self, record: GroupedImageRecord, split: str):
        self.record = record
        self.split = split  # 'train', 'val', or 'test'

    def to_dict(self) -> Dict[str, Any]:
        d = self.record.to_dict()
        d["split"] = self.split
        return d


class GroupAwareSplitter:
    """Group-aware dataset splitter preventing data leakage by assigning entire source_groups to splits."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_dataset(
        self, records: List[GroupedImageRecord]
    ) -> List[SplitRecord]:
        """Perform deterministic group-aware train/val/test split.

        Args:
            records: List of GroupedImageRecord objects.

        Returns:
            List of SplitRecord objects with assigned split ('train', 'val', 'test').
        """
        if not records:
            return []

        # Group records by source_group_id
        group_to_records: Dict[str, List[GroupedImageRecord]] = {}
        for r in records:
            group_to_records.setdefault(r.source_group_id, []).append(r)

        # Sort group IDs for deterministic ordering before shuffling
        unique_groups = sorted(group_to_records.keys())

        # Shuffle groups using fixed seed
        rng = random.Random(self.seed)
        rng.shuffle(unique_groups)

        num_groups = len(unique_groups)
        num_train = max(1, int(round(num_groups * self.train_ratio)))
        num_val = int(round(num_groups * self.val_ratio))

        if num_groups > 1 and num_val == 0 and num_groups - num_train > 0:
            num_val = 1

        num_test = num_groups - num_train - num_val
        if num_test < 0:
            num_test = 0
            num_val = num_groups - num_train

        train_groups = set(unique_groups[:num_train])
        val_groups = set(unique_groups[num_train : num_train + num_val])
        test_groups = set(unique_groups[num_train + num_val :])

        split_records: List[SplitRecord] = []
        for grp_id, grp_recs in group_to_records.items():
            if grp_id in train_groups:
                assigned_split = "train"
            elif grp_id in val_groups:
                assigned_split = "val"
            else:
                assigned_split = "test"

            for rec in grp_recs:
                split_records.append(SplitRecord(record=rec, split=assigned_split))

        return split_records

    def calculate_class_balance_and_weights(
        self, split_records: List[SplitRecord]
    ) -> Dict[str, Any]:
        """Calculate class distribution per split and inverse class frequency weights for training."""
        split_counts: Dict[str, Dict[str, int]] = {
            "train": {},
            "val": {},
            "test": {},
        }
        train_breed_counts: Dict[str, int] = {}

        for sr in split_records:
            split_name = sr.split
            breed = sr.record.metadata.breed_name
            split_counts[split_name][breed] = (
                split_counts[split_name].get(breed, 0) + 1
            )
            if split_name == "train":
                train_breed_counts[breed] = train_breed_counts.get(breed, 0) + 1

        # Calculate inverse class frequency weights for train split:
        # W_c = N_total / (K * N_c)
        total_train_samples = sum(train_breed_counts.values())
        num_classes = len(train_breed_counts) if train_breed_counts else 1

        class_weights: Dict[str, float] = {}
        for breed, count in train_breed_counts.items():
            if count > 0:
                class_weights[breed] = round(
                    total_train_samples / (num_classes * count), 4
                )
            else:
                class_weights[breed] = 1.0

        return {
            "split_distribution": split_counts,
            "train_class_counts": train_breed_counts,
            "train_class_weights_inverse_freq": class_weights,
        }
