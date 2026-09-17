"""Dataset Inspector module for Indian Cattle and Buffalo Breed Dataset.

Performs non-destructive inventory inspection, OpenCV image analysis, corruption checks,
exact duplicate detection via hashing, and augmented file relationship identification.
"""

import os
import hashlib
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import cv2
import numpy as np
from PIL import Image

from ml.common.breed_registry import BreedRegistry

# Recognized image file extensions
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}


def compute_file_hashes(file_path: Path) -> Tuple[str, str]:
    """Compute MD5 and SHA-256 hashes of a file.

    Args:
        file_path: Path to the target file.

    Returns:
        Tuple of (md5_hash, sha256_hash) strings.
    """
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            md5.update(chunk)
            sha256.update(chunk)

    return md5.hexdigest(), sha256.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> Optional[str]:
    """Compute difference hash (dHash) for perceptual image similarity comparison.

    Args:
        image_path: Path to the image file.
        hash_size: Grid resolution (default 8 for 64-bit hash).

    Returns:
        Hexadecimal dHash string or None if unreadable.
    """
    try:
        # Load grayscale image via PIL
        with Image.open(image_path) as img:
            img_gray = img.convert("L").resize(
                (hash_size + 1, hash_size), Image.Resampling.BILINEAR
            )
            pixels = np.array(img_gray, dtype=np.int32)

            # Compare adjacent pixels horizontally
            diff = pixels[:, 1:] > pixels[:, :-1]
            # Convert boolean array to hex string
            decimal_val = 0
            for bit in diff.flatten():
                decimal_val = (decimal_val << 1) | int(bit)
            return f"{decimal_val:016x}"
    except Exception:
        return None


def calculate_hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    """Compute Hamming distance between two hex-encoded 64-bit dHash strings."""
    if len(hash1_hex) != len(hash2_hex):
        return 64
    val1 = int(hash1_hex, 16)
    val2 = int(hash2_hex, 16)
    x = val1 ^ val2
    # Count set bits
    return bin(x).count("1")


class ImageMetadata:
    """Data container for inspected image metadata."""

    def __init__(
        self,
        file_path: Path,
        relative_path: str,
        animal_type: str,
        breed_name: str,
        file_size_bytes: int,
        width: int = 0,
        height: int = 0,
        channels: int = 0,
        aspect_ratio: float = 0.0,
        image_format: str = "UNKNOWN",
        is_corrupted: bool = False,
        is_unreadable: bool = False,
        is_zero_byte: bool = False,
        error_message: Optional[str] = None,
        md5_hash: str = "",
        sha256_hash: str = "",
        dhash: Optional[str] = None,
        is_augmented_filename: bool = False,
    ):
        self.file_path = file_path
        self.relative_path = relative_path
        self.animal_type = animal_type
        self.breed_name = breed_name
        self.file_size_bytes = file_size_bytes
        self.width = width
        self.height = height
        self.channels = channels
        self.aspect_ratio = aspect_ratio
        self.image_format = image_format
        self.is_corrupted = is_corrupted
        self.is_unreadable = is_unreadable
        self.is_zero_byte = is_zero_byte
        self.error_message = error_message
        self.md5_hash = md5_hash
        self.sha256_hash = sha256_hash
        self.dhash = dhash
        self.is_augmented_filename = is_augmented_filename

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata object to dictionary representation."""
        return {
            "relative_path": self.relative_path,
            "filename": self.file_path.name,
            "animal_type": self.animal_type,
            "breed_name": self.breed_name,
            "file_size_bytes": self.file_size_bytes,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "format": self.image_format,
            "is_corrupted": self.is_corrupted,
            "is_unreadable": self.is_unreadable,
            "is_zero_byte": self.is_zero_byte,
            "error_message": self.error_message,
            "md5_hash": self.md5_hash,
            "sha256_hash": self.sha256_hash,
            "dhash": self.dhash,
            "is_augmented_filename": self.is_augmented_filename,
        }


class DatasetInspector:
    """Inspector for performing complete dataset inventory, validation, and analysis."""

    def __init__(self, dataset_dir: Path, registry: Optional[BreedRegistry] = None):
        """Initialize DatasetInspector.

        Args:
            dataset_dir: Path to raw dataset root (e.g. data/raw/dataset)
            registry: Optional BreedRegistry instance.
        """
        self.dataset_dir = Path(dataset_dir)
        self.registry = registry if registry is not None else BreedRegistry()
        self.images_metadata: List[ImageMetadata] = []
        self.unexpected_files: List[str] = []
        self.unexpected_folders: List[str] = []
        self.missing_label_files: List[str] = []

    def inspect_file(self, file_path: Path, animal_type: str, breed_name: str) -> ImageMetadata:
        """Inspect an individual file and return extracted metadata."""
        rel_path = str(file_path.relative_to(self.dataset_dir))
        file_name = file_path.name
        is_aug_fn = "aug_" in file_name.lower() or "_aug" in file_name.lower()

        # Check zero-byte file
        try:
            size_bytes = file_path.stat().st_size
        except Exception as e:
            return ImageMetadata(
                file_path=file_path,
                relative_path=rel_path,
                animal_type=animal_type,
                breed_name=breed_name,
                file_size_bytes=0,
                is_unreadable=True,
                error_message=f"Failed to access file stat: {str(e)}",
            )

        if size_bytes == 0:
            return ImageMetadata(
                file_path=file_path,
                relative_path=rel_path,
                animal_type=animal_type,
                breed_name=breed_name,
                file_size_bytes=0,
                is_zero_byte=True,
                is_corrupted=True,
                error_message="Zero-byte file",
            )

        # Calculate file hashes
        try:
            md5_h, sha256_h = compute_file_hashes(file_path)
        except Exception as e:
            return ImageMetadata(
                file_path=file_path,
                relative_path=rel_path,
                animal_type=animal_type,
                breed_name=breed_name,
                file_size_bytes=size_bytes,
                is_unreadable=True,
                error_message=f"Failed to compute file hash: {str(e)}",
            )

        # Compute dHash
        dhash_val = compute_dhash(file_path)

        # Inspect dimensions with OpenCV
        try:
            img_np = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img_np is None:
                # Fallback to PIL decode
                with Image.open(file_path) as pil_img:
                    pil_img.verify()
                with Image.open(file_path) as pil_img:
                    w, h = pil_img.size
                    fmt = pil_img.format or file_path.suffix.lstrip(".").upper()
                    channels = len(pil_img.getbands())
                    ar = w / h if h > 0 else 0.0
                    return ImageMetadata(
                        file_path=file_path,
                        relative_path=rel_path,
                        animal_type=animal_type,
                        breed_name=breed_name,
                        file_size_bytes=size_bytes,
                        width=w,
                        height=h,
                        channels=channels,
                        aspect_ratio=ar,
                        image_format=fmt,
                        md5_hash=md5_h,
                        sha256_hash=sha256_h,
                        dhash=dhash_val,
                        is_augmented_filename=is_aug_fn,
                    )

            if len(img_np.shape) == 2:
                h, w = img_np.shape
                channels = 1
            elif len(img_np.shape) == 3:
                h, w, channels = img_np.shape
            else:
                h, w = img_np.shape[0], img_np.shape[1]
                channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

            ar = w / h if h > 0 else 0.0
            fmt = file_path.suffix.lstrip(".").upper()
            if fmt == "JPG":
                fmt = "JPEG"

            return ImageMetadata(
                file_path=file_path,
                relative_path=rel_path,
                animal_type=animal_type,
                breed_name=breed_name,
                file_size_bytes=size_bytes,
                width=w,
                height=h,
                channels=channels,
                aspect_ratio=ar,
                image_format=fmt,
                md5_hash=md5_h,
                sha256_hash=sha256_h,
                dhash=dhash_val,
                is_augmented_filename=is_aug_fn,
            )

        except Exception as e:
            return ImageMetadata(
                file_path=file_path,
                relative_path=rel_path,
                animal_type=animal_type,
                breed_name=breed_name,
                file_size_bytes=size_bytes,
                is_corrupted=True,
                error_message=f"Corrupted image payload: {str(e)}",
                md5_hash=md5_h,
                sha256_hash=sha256_h,
            )

    def scan(self) -> List[ImageMetadata]:
        """Perform full non-destructive dataset directory traversal."""
        self.images_metadata = []
        self.unexpected_files = []
        self.unexpected_folders = []
        self.missing_label_files = []

        if not self.dataset_dir.exists():
            return self.images_metadata

        # Discover any new breeds from actual structure
        self.registry.discover_from_dataset_dir(self.dataset_dir)

        # Traverse dataset root
        for root, dirs, files in os.walk(self.dataset_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(self.dataset_dir)
            parts = rel_root.parts

            for file_name in files:
                if file_name.startswith("."):
                    continue

                file_path = root_path / file_name
                ext = file_path.suffix.lower()

                if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                    self.unexpected_files.append(str(file_path.relative_to(self.dataset_dir)))
                    continue

                # Determine animal_type and breed_name from directory structure
                if len(parts) == 0:
                    # File placed at dataset root
                    animal_type = "unknown"
                    breed_name = "unknown"
                    self.missing_label_files.append(str(file_path.relative_to(self.dataset_dir)))
                elif len(parts) == 1:
                    # File placed at animal root (e.g. data/raw/dataset/cattle/img.jpg)
                    animal_type = parts[0].lower()
                    breed_name = "unknown"
                    self.missing_label_files.append(str(file_path.relative_to(self.dataset_dir)))
                else:
                    animal_type = parts[0].lower()
                    breed_name = parts[1]

                meta = self.inspect_file(file_path, animal_type, breed_name)
                self.images_metadata.append(meta)

        return self.images_metadata

    def analyze_duplicates(self) -> Dict[str, Any]:
        """Perform exact duplicate detection and augmented image relationship analysis."""
        # 1. Exact duplicates by SHA-256 hash
        hash_to_files: Dict[str, List[str]] = {}
        for m in self.images_metadata:
            if m.sha256_hash:
                hash_to_files.setdefault(m.sha256_hash, []).append(m.relative_path)

        exact_duplicates: List[Dict[str, Any]] = []
        for sha256, paths in hash_to_files.items():
            if len(paths) > 1:
                exact_duplicates.append(
                    {
                        "sha256_hash": sha256,
                        "duplicate_count": len(paths),
                        "file_paths": paths,
                    }
                )

        # 2. Augmented / Perceptual similarity analysis
        augmented_by_filename = [
            m.relative_path for m in self.images_metadata if m.is_augmented_filename
        ]

        # Group by perceptual dHash (distance <= 4)
        valid_dhashes = [m for m in self.images_metadata if m.dhash and not m.is_corrupted]
        possible_augmented_groups: List[Dict[str, Any]] = []
        visited: Set[str] = set()

        for i, img1 in enumerate(valid_dhashes):
            if img1.relative_path in visited:
                continue

            cluster = [img1.relative_path]
            for j in range(i + 1, len(valid_dhashes)):
                img2 = valid_dhashes[j]
                if img2.relative_path in visited:
                    continue

                dist = calculate_hamming_distance(img1.dhash, img2.dhash)
                # If dHash hamming distance <= 4 and matching breed
                if dist <= 4 and img1.breed_name == img2.breed_name:
                    cluster.append(img2.relative_path)
                    visited.add(img2.relative_path)

            if len(cluster) > 1:
                visited.add(img1.relative_path)
                possible_augmented_groups.append(
                    {
                        "representative_path": img1.relative_path,
                        "group_size": len(cluster),
                        "member_paths": cluster,
                    }
                )

        return {
            "total_exact_duplicate_groups": len(exact_duplicates),
            "total_exact_duplicate_files": sum(
                item["duplicate_count"] - 1 for item in exact_duplicates
            ),
            "exact_duplicate_groups": exact_duplicates,
            "total_augmented_filename_matches": len(augmented_by_filename),
            "augmented_filename_files": augmented_by_filename,
            "total_perceptual_similar_groups": len(possible_augmented_groups),
            "possible_augmented_groups": possible_augmented_groups,
        }

    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Compute statistical summary across all inspected images."""
        if not self.images_metadata:
            self.scan()

        valid_images = [m for m in self.images_metadata if not m.is_corrupted]
        total_images = len(self.images_metadata)

        # Breed & Animal distribution
        images_per_animal_type: Dict[str, int] = {}
        images_per_breed: Dict[str, int] = {}
        format_counts: Dict[str, int] = {}

        for m in self.images_metadata:
            images_per_animal_type[m.animal_type] = (
                images_per_animal_type.get(m.animal_type, 0) + 1
            )
            images_per_breed[m.breed_name] = images_per_breed.get(m.breed_name, 0) + 1
            if m.image_format:
                format_counts[m.image_format] = format_counts.get(m.image_format, 0) + 1

        # Dimension statistics
        widths = [m.width for m in valid_images if m.width > 0]
        heights = [m.height for m in valid_images if m.height > 0]
        aspect_ratios = [m.aspect_ratio for m in valid_images if m.aspect_ratio > 0]
        file_sizes = [m.file_size_bytes for m in valid_images if m.file_size_bytes > 0]

        dim_stats = {
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "mean_width": round(statistics.mean(widths), 2) if widths else 0.0,
            "median_width": statistics.median(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "mean_height": round(statistics.mean(heights), 2) if heights else 0.0,
            "median_height": statistics.median(heights) if heights else 0,
            "min_aspect_ratio": round(min(aspect_ratios), 4) if aspect_ratios else 0.0,
            "max_aspect_ratio": round(max(aspect_ratios), 4) if aspect_ratios else 0.0,
            "mean_aspect_ratio": round(statistics.mean(aspect_ratios), 4)
            if aspect_ratios
            else 0.0,
            "min_file_size_bytes": min(file_sizes) if file_sizes else 0,
            "max_file_size_bytes": max(file_sizes) if file_sizes else 0,
            "mean_file_size_bytes": round(statistics.mean(file_sizes), 2)
            if file_sizes
            else 0.0,
        }

        # Issue counters
        corrupted_count = sum(1 for m in self.images_metadata if m.is_corrupted)
        unreadable_count = sum(1 for m in self.images_metadata if m.is_unreadable)
        zero_byte_count = sum(1 for m in self.images_metadata if m.is_zero_byte)

        # Duplicate analysis
        duplicate_info = self.analyze_duplicates()

        return {
            "total_images": total_images,
            "valid_images": len(valid_images),
            "corrupted_images": corrupted_count,
            "unreadable_images": unreadable_count,
            "zero_byte_files": zero_byte_count,
            "missing_label_files_count": len(self.missing_label_files),
            "missing_label_files": self.missing_label_files,
            "unexpected_files_count": len(self.unexpected_files),
            "unexpected_files": self.unexpected_files,
            "images_per_animal_type": images_per_animal_type,
            "images_per_breed": images_per_breed,
            "format_counts": format_counts,
            "dimension_statistics": dim_stats,
            "duplicate_analysis": duplicate_info,
        }
