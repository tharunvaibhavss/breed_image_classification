"""YOLO Animal Detector and Cropping Engine module.

Performs bounding-box animal detection (class 0 = cattle, class 1 = buffalo) and animal ROI crop
extraction for downstream breed classification.
"""

from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ml.preprocessing.opencv_pipeline import OpenCVPreprocessor

# YOLO Class Mapping
YOLO_CLASSES: Dict[int, str] = {
    0: "cattle",
    1: "buffalo",
    19: "cattle",  # COCO cow class support for default yolov8n.pt weights
}


class DetectionResult(BaseModel):
    """Pydantic model representing animal detection bounding box outcome."""

    class_id: int = Field(description="YOLO detection class ID (0 = cattle, 1 = buffalo)")
    class_name: str = Field(description="Animal class name ('cattle' or 'buffalo')")
    confidence: float = Field(description="Detection confidence score [0.0, 1.0]")
    bbox: Tuple[int, int, int, int] = Field(
        description="Pixel bounding box tuple (xmin, ymin, xmax, ymax)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
        }


class YOLOAnimalDetector:
    """YOLO Animal Detector handling model inference, thresholding, and ROI cropping."""

    def __init__(self, model_path: Optional[Union[Path, str]] = None):
        """Initialize YOLOAnimalDetector.

        Args:
            model_path: Optional path to custom trained weights (e.g. models/yolo_best.pt).
                        If None, uses 'yolov8n.pt' lightweight backbone.
        """
        self.model_path = str(model_path) if model_path else "yolov8n.pt"
        self._model = None
        self.preprocessor = OpenCVPreprocessor()

    def _get_model(self):
        """Lazy load Ultralytics YOLO model instance."""
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_path)
            except ImportError:
                raise ImportError(
                    "Ultralytics YOLO package ('ultralytics') is required for YOLOAnimalDetector. "
                    "Please install ultralytics."
                )
        return self._model

    def detect_animals(
        self,
        source: Union[Path, str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[DetectionResult]:
        """Perform object detection on input image source.

        Args:
            source: File path, string path, or RGB/BGR numpy image array.
            conf_threshold: Minimum confidence score threshold.
            iou_threshold: Non-maximum suppression (NMS) IoU threshold.

        Returns:
            List of DetectionResult instances filtered by confidence score.
        """
        raw_img = self.preprocessor.load_image(source)
        rgb_img = self.preprocessor.ensure_rgb(raw_img)
        h, w = rgb_img.shape[:2]

        # Convert RGB to BGR for Ultralytics OpenCV compatibility
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        model = self._get_model()
        results = model.predict(
            source=bgr_img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        for box in first_res.boxes:
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())

            # Filter for cattle (0) or buffalo (1) classes
            if cls_id not in YOLO_CLASSES:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = (
                int(round(xyxy[0])),
                int(round(xyxy[1])),
                int(round(xyxy[2])),
                int(round(xyxy[3])),
            )

            # Clamp coordinates to image dimensions
            xmin = max(0, min(w - 1, xmin))
            ymin = max(0, min(h - 1, ymin))
            xmax = max(xmin + 1, min(w, xmax))
            ymax = max(ymin + 1, min(h, ymax))

            det = DetectionResult(
                class_id=0 if cls_id in (0, 19) else 1,
                class_name=YOLO_CLASSES[cls_id],
                confidence=conf,
                bbox=(xmin, ymin, xmax, ymax),
            )
            detections.append(det)

        return detections

    def crop_animals(
        self,
        image_rgb: np.ndarray,
        detections: List[DetectionResult],
        pad_percent: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Extract cropped animal ROI arrays from RGB image using bounding box coordinates.

        Args:
            image_rgb: 3-channel RGB uint8 image array.
            detections: List of DetectionResult objects.
            pad_percent: Percentage padding to expand bounding box around detected animal.

        Returns:
            List of Dicts containing 'crop' (RGB array), 'detection' (DetectionResult), 'bbox'.
        """
        self.preprocessor.validate_image(image_rgb)
        h, w = image_rgb.shape[:2]
        cropped_results: List[Dict[str, Any]] = []

        for det in detections:
            xmin, ymin, xmax, ymax = det.bbox

            # Add percentage padding around bounding box if specified
            if pad_percent > 0.0:
                bw = xmax - xmin
                bh = ymax - ymin
                pad_w = int(round(bw * pad_percent))
                pad_h = int(round(bh * pad_percent))

                xmin = max(0, xmin - pad_w)
                ymin = max(0, ymin - pad_h)
                xmax = min(w, xmax + pad_w)
                ymax = min(h, ymax + pad_h)

            crop_np = image_rgb[ymin:ymax, xmin:xmax].copy()

            cropped_results.append(
                {
                    "crop": crop_np,
                    "detection": det,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                }
            )

        return cropped_results
