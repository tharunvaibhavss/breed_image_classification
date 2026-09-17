"""OpenCV Image Preprocessing Pipeline for EfficientNet-B0 and Animal Detection.

Provides deterministic, reusable image loading, RGB conversion, aspect ratio handling,
ROI cropping, resizing, and normalization for training, validation, testing, and inference.
"""

from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OpenCVPreprocessor:
    """Reusable OpenCV image preprocessor for EfficientNet-B0 model inputs."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        keep_aspect_ratio: bool = True,
        pad_color: Tuple[int, int, int] = (0, 0, 0),
        normalize_to_unit: bool = True,
        apply_imagenet_norm: bool = False,
    ):
        """Initialize OpenCVPreprocessor.

        Args:
            target_size: Tuple of (height, width) or (width, height). Default (224, 224).
            keep_aspect_ratio: If True, letterbox pads the image to preserve aspect ratio.
            pad_color: RGB tuple for padding color (default black (0, 0, 0)).
            normalize_to_unit: Scale uint8 pixel values [0, 255] to float32 [0.0, 1.0].
            apply_imagenet_norm: Apply ImageNet mean and std normalization.
        """
        self.target_height = target_size[0]
        self.target_width = target_size[1]
        self.keep_aspect_ratio = keep_aspect_ratio
        self.pad_color = pad_color
        self.normalize_to_unit = normalize_to_unit
        self.apply_imagenet_norm = apply_imagenet_norm

    def load_image(self, source: Union[Path, str, bytes, np.ndarray]) -> np.ndarray:
        """Load an image from path string, Path object, in-memory bytes, or numpy array.

        Args:
            source: Path, string file path, raw bytes, or existing numpy array.

        Returns:
            Loaded OpenCV BGR or Grayscale image array.

        Raises:
            ValueError: If image source cannot be read or is invalid.
        """
        if isinstance(source, np.ndarray):
            return source.copy()

        if isinstance(source, bytes):
            img_np = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_UNCHANGED)
            if img_np is None:
                raise ValueError("Failed to decode image from in-memory bytes payload.")
            return img_np

        path_obj = Path(source)
        if not path_obj.exists() or not path_obj.is_file():
            raise ValueError(f"Image file path does not exist: {path_obj}")

        img_np = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
        if img_np is None:
            raise ValueError(f"OpenCV failed to read image from path: {path_obj}")

        return img_np

    def validate_image(self, image_np: np.ndarray) -> bool:
        """Validate non-empty array, non-zero dimensions, and valid channels.

        Args:
            image_np: Numpy array to validate.

        Returns:
            True if image array is valid.

        Raises:
            ValueError: If image array violates structural integrity.
        """
        if image_np is None or not isinstance(image_np, np.ndarray):
            raise ValueError("Input image is None or not a numpy ndarray.")

        if image_np.size == 0 or len(image_np.shape) < 2:
            raise ValueError("Input image array is empty or has fewer than 2 dimensions.")

        h, w = image_np.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: height={h}, width={w}")

        return True

    def ensure_rgb(self, image_np: np.ndarray) -> np.ndarray:
        """Convert BGR, Grayscale (1-channel), or BGRA (4-channel) arrays to 3-channel RGB.

        Args:
            image_np: Input numpy array.

        Returns:
            3-channel RGB numpy array.
        """
        self.validate_image(image_np)

        if len(image_np.shape) == 2:
            # Grayscale (H, W) -> RGB (H, W, 3)
            return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        if len(image_np.shape) == 3:
            channels = image_np.shape[2]
            if channels == 1:
                return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif channels == 3:
                # OpenCV loads as BGR by default, convert BGR -> RGB
                return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                # BGRA -> RGB
                return cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)

        raise ValueError(f"Unsupported number of channels: {image_np.shape}")

    def crop_roi(
        self,
        image_rgb: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Crop region of interest (ROI) bounding box from RGB image.

        Args:
            image_rgb: 3-channel RGB numpy array.
            bbox: Optional tuple of (xmin, ymin, xmax, ymax) in pixel coordinates.
                  If None, returns original image.

        Returns:
            Cropped RGB image region.
        """
        if bbox is None:
            return image_rgb

        h, w = image_rgb.shape[:2]
        xmin, ymin, xmax, ymax = bbox

        # Clamp bounding box coordinates within image bounds
        xmin = max(0, min(w - 1, int(xmin)))
        ymin = max(0, min(h - 1, int(ymin)))
        xmax = max(xmin + 1, min(w, int(xmax)))
        ymax = max(ymin + 1, min(h, int(ymax)))

        return image_rgb[ymin:ymax, xmin:xmax].copy()

    def resize_with_aspect_ratio(
        self, image_rgb: np.ndarray
    ) -> np.ndarray:
        """Resize image to target size while preserving aspect ratio using letterbox padding.

        Args:
            image_rgb: 3-channel RGB image numpy array.

        Returns:
            Resized and padded RGB image of shape (target_height, target_width, 3).
        """
        if not self.keep_aspect_ratio:
            return cv2.resize(
                image_rgb,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )

        h, w = image_rgb.shape[:2]
        scale = min(self.target_width / w, self.target_height / h)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded background container
        padded = np.full(
            (self.target_height, self.target_width, 3),
            self.pad_color,
            dtype=np.uint8,
        )

        # Center resized image in container
        top = (self.target_height - new_h) // 2
        left = (self.target_width - new_w) // 2

        padded[top : top + new_h, left : left + new_w] = resized
        return padded

    def normalize(self, image_rgb: np.ndarray) -> np.ndarray:
        """Normalize RGB image array to float32 [0.0, 1.0] and optional ImageNet mean/std.

        Args:
            image_rgb: uint8 RGB numpy array.

        Returns:
            float32 RGB array.
        """
        img_float = image_rgb.astype(np.float32)

        if self.normalize_to_unit:
            img_float /= 255.0

        if self.apply_imagenet_norm:
            img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD

        return img_float

    def preprocess_to_numpy(
        self,
        source: Union[Path, str, bytes, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Full end-to-end preprocessing pipeline returning numpy array.

        Args:
            source: Path, string file path, raw bytes, or numpy array.
            bbox: Optional bounding box crop (xmin, ymin, xmax, ymax).

        Returns:
            Preprocessed float32 numpy array of shape (target_height, target_width, 3).
        """
        raw_img = self.load_image(source)
        rgb_img = self.ensure_rgb(raw_img)
        cropped_img = self.crop_roi(rgb_img, bbox=bbox)
        resized_img = self.resize_with_aspect_ratio(cropped_img)
        normalized_img = self.normalize(resized_img)
        return normalized_img

    def preprocess_to_tensor(
        self,
        source: Union[Path, str, bytes, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Any:
        """Full end-to-end preprocessing pipeline returning PyTorch Tensor.

        PyTorch vision models expect tensor shape (Channels, Height, Width) i.e. (3, 224, 224).

        Args:
            source: Path, string file path, raw bytes, or numpy array.
            bbox: Optional bounding box crop (xmin, ymin, xmax, ymax).

        Returns:
            torch.Tensor float32 tensor of shape (3, target_height, target_width).
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch ('torch') package is required for preprocess_to_tensor(). "
                "Please install PyTorch or use preprocess_to_numpy()."
            )

        np_processed = self.preprocess_to_numpy(source, bbox=bbox)
        # Transpose (H, W, C) -> (C, H, W)
        tensor_processed = torch.from_numpy(np_processed).permute(2, 0, 1)
        return tensor_processed
