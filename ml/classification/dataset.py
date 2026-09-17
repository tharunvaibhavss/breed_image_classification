"""PyTorch Breed Dataset and DataLoader utilities for EfficientNet-B0.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml.common.breed_registry import BreedRegistry
from ml.preprocessing.opencv_pipeline import OpenCVPreprocessor
from ml.preprocessing.albumentations_pipeline import AugmentationPipeline

# Default breed label to integer class_id mapping
BREED_NAME_TO_CLASS_ID: Dict[str, int] = {
    "gir": 0,
    "ongole": 1,
    "sahiwal": 2,
    "jaffarabadi": 3,
    "murrah": 4,
    "surti": 5,
}


class BreedDataset(Dataset):
    """PyTorch Dataset for Indian Cattle and Buffalo breed images."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        base_dir: Optional[Path] = None,
        transform: Optional[Any] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize BreedDataset.

        Args:
            samples: List of sample dicts containing 'file_path' and 'breed' or 'class_id'.
            base_dir: Base directory path for resolving relative file paths.
            transform: Optional Albumentations transform pipeline.
            target_size: Tuple of (height, width). Default (224, 224).
        """
        self.samples = samples
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.transform = transform
        self.target_size = target_size
        self.preprocessor = OpenCVPreprocessor(target_size=target_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        rel_path = sample.get("file_path", "")
        file_path = self.base_dir / rel_path

        # Determine class_id integer
        if "class_id" in sample:
            class_id = int(sample["class_id"])
        else:
            breed_str = sample.get("breed", "gir").lower()
            class_id = BREED_NAME_TO_CLASS_ID.get(breed_str, 0)

        # Load image via OpenCV preprocessor
        try:
            raw_img = self.preprocessor.load_image(file_path)
            rgb_img = self.preprocessor.ensure_rgb(raw_img)
        except Exception:
            # Fallback black image array if file read fails
            rgb_img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)

        # Apply Albumentations transform if provided, else standard OpenCV resize
        if self.transform is not None:
            augmented = self.transform(image=rgb_img)
            img_processed = augmented["image"]
        else:
            img_processed = self.preprocessor.resize_with_aspect_ratio(rgb_img)
            img_processed = self.preprocessor.normalize(img_processed)

        # Ensure float32 numpy array
        if img_processed.dtype != np.float32:
            img_processed = img_processed.astype(np.float32)

        # Transpose (H, W, C) -> PyTorch (C, H, W)
        tensor_img = torch.from_numpy(img_processed).permute(2, 0, 1)
        return tensor_img, class_id


def create_dataloaders(
    manifest_records: List[Dict[str, Any]],
    base_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (224, 224),
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train, val, and test splits.

    Args:
        manifest_records: List of dict records from dataset manifest.
        base_dir: Root dataset directory path.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker threads count.
        target_size: Input resolution (height, width).

    Returns:
        Dict mapping split name ('train', 'val', 'test') to PyTorch DataLoader.
    """
    pipeline = AugmentationPipeline(target_size=target_size, normalize_to_unit=True)

    train_transform = pipeline.get_train_transforms(enable_bbox=False)
    val_transform = pipeline.get_val_transforms(enable_bbox=False)

    split_samples: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for r in manifest_records:
        split = r.get("split", "train")
        split_samples.setdefault(split, []).append(r)

    train_dataset = BreedDataset(
        samples=split_samples["train"],
        base_dir=base_dir,
        transform=train_transform,
        target_size=target_size,
    )
    val_dataset = BreedDataset(
        samples=split_samples["val"],
        base_dir=base_dir,
        transform=val_transform,
        target_size=target_size,
    )
    test_dataset = BreedDataset(
        samples=split_samples["test"],
        base_dir=base_dir,
        transform=val_transform,
        target_size=target_size,
    )

    train_shuffle = len(train_dataset) > 0

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    return dataloaders
