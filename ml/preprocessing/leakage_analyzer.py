"""Leakage Analysis & Source Image Grouping module.

Groups derived, augmented, and identical images into unique source_group_id clusters to
prevent data leakage between train, validation, and test splits.
"""

from typing import Dict, List, Set, Any
from ml.common.dataset_inspector import ImageMetadata, calculate_hamming_distance


class GroupedImageRecord:
    """Record linking ImageMetadata to its assigned source group."""

    def __init__(self, metadata: ImageMetadata, source_group_id: str):
        self.metadata = metadata
        self.source_group_id = source_group_id

    def to_dict(self) -> Dict[str, Any]:
        d = self.metadata.to_dict()
        d["source_group"] = self.source_group_id
        return d


class LeakageAnalyzer:
    """Analyzer engine identifying image relationships and assigning source group IDs."""

    def analyze_and_group(
        self, images_metadata: List[ImageMetadata]
    ) -> List[GroupedImageRecord]:
        """Group images by exact hash, filename patterns, and perceptual hash similarity.

        Args:
            images_metadata: List of ImageMetadata from DatasetInspector.

        Returns:
            List of GroupedImageRecord objects with assigned source_group_id.
        """
        grouped_records: List[GroupedImageRecord] = []
        if not images_metadata:
            return grouped_records

        # Map relative path -> ImageMetadata
        path_to_meta: Dict[str, ImageMetadata] = {
            m.relative_path: m for m in images_metadata
        }

        # Build adjacency graph for grouping related images
        adj: Dict[str, Set[str]] = {m.relative_path: set() for m in images_metadata}

        # 1. Connect exact SHA-256 hash duplicates
        hash_groups: Dict[str, List[str]] = {}
        for m in images_metadata:
            if m.sha256_hash:
                hash_groups.setdefault(m.sha256_hash, []).append(m.relative_path)

        for sha256_val, paths in hash_groups.items():
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    adj[paths[i]].add(paths[j])
                    adj[paths[j]].add(paths[i])

        # 2. Connect images with 'aug_' filename stem patterns
        # e.g., 'gir_01.jpg' and 'aug_gir_01.jpg' or 'gir_01_aug.jpg'
        stem_groups: Dict[str, List[str]] = {}
        for m in images_metadata:
            fname = m.file_path.name.lower()
            # Strip common augmentation prefixes/suffixes
            clean_stem = (
                fname.replace("aug_", "")
                .replace("_aug", "")
                .replace("aug-", "")
                .replace("-aug", "")
            )
            # Combine breed and clean_stem as group key
            key = f"{m.breed_name.lower()}_{clean_stem}"
            stem_groups.setdefault(key, []).append(m.relative_path)

        for key, paths in stem_groups.items():
            if len(paths) > 1:
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        adj[paths[i]].add(paths[j])
                        adj[paths[j]].add(paths[i])

        # 3. Connect perceptual hash (dHash) matches within same breed
        valid_dhashes = [m for m in images_metadata if m.dhash and not m.is_corrupted]
        for i in range(len(valid_dhashes)):
            m1 = valid_dhashes[i]
            for j in range(i + 1, len(valid_dhashes)):
                m2 = valid_dhashes[j]
                if m1.breed_name == m2.breed_name:
                    dist = calculate_hamming_distance(m1.dhash, m2.dhash)
                    if dist <= 4:
                        adj[m1.relative_path].add(m2.relative_path)
                        adj[m2.relative_path].add(m1.relative_path)

        # Connected component traversal to assign group IDs
        visited: Set[str] = set()
        group_counter = 1

        for m in sorted(images_metadata, key=lambda x: x.relative_path):
            if m.relative_path in visited:
                continue

            # BFS / DFS cluster search
            cluster: List[str] = []
            queue = [m.relative_path]
            visited.add(m.relative_path)

            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            group_id = f"group_{group_counter:04d}"
            group_counter += 1

            for rel_p in cluster:
                grouped_records.append(
                    GroupedImageRecord(
                        metadata=path_to_meta[rel_p], source_group_id=group_id
                    )
                )

        return grouped_records
