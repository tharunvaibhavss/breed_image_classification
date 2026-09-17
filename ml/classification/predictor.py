"""Breed Classification Inference Predictor module.

Executes EfficientNet-B0 inference on animal image crops and returns Top-1 prediction,
confidence score, and Top-3 predictions.
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from ml.common.breed_registry import BreedRegistry
from ml.preprocessing.opencv_pipeline import OpenCVPreprocessor
from ml.classification.efficientnet import BreedClassifier

# Default class mapping dictionary
DEFAULT_CLASS_MAPPING: Dict[int, Dict[str, str]] = {
    0: {"breed_name": "Gir", "animal_type": "cattle", "display_name": "Gir Cattle"},
    1: {"breed_name": "Ongole", "animal_type": "cattle", "display_name": "Ongole Cattle"},
    2: {"breed_name": "Sahiwal", "animal_type": "cattle", "display_name": "Sahiwal Cattle"},
    3: {"breed_name": "Jaffarabadi", "animal_type": "buffalo", "display_name": "Jaffarabadi Buffalo"},
    4: {"breed_name": "Murrah", "animal_type": "buffalo", "display_name": "Murrah Buffalo"},
    5: {"breed_name": "Surti", "animal_type": "buffalo", "display_name": "Surti Buffalo"},
}


class BreedPredictionItem(BaseModel):
    """Individual breed prediction item for Top-K rankings."""

    class_id: int = Field(description="Class integer index (0..5)")
    breed_name: str = Field(description="Breed name string (e.g. 'Gir')")
    display_name: str = Field(description="Formatted display name (e.g. 'Gir Cattle')")
    animal_type: str = Field(description="Animal classification ('cattle' or 'buffalo')")
    confidence: float = Field(description="Probability confidence score [0.0, 1.0]")


class PredictionResult(BaseModel):
    """Pydantic model representing complete breed prediction output."""

    predicted_breed: str = Field(description="Top-1 predicted breed name")
    display_name: str = Field(description="Top-1 formatted display name")
    animal_type: str = Field(description="Predicted animal classification")
    confidence: float = Field(description="Top-1 prediction confidence score")
    top_3_predictions: List[BreedPredictionItem] = Field(description="Top-3 ranked breed predictions")

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary."""
        return {
            "predicted_breed": self.predicted_breed,
            "display_name": self.display_name,
            "animal_type": self.animal_type,
            "confidence": round(self.confidence, 4),
            "top_3_predictions": [item.model_dump() for item in self.top_3_predictions],
        }


class BreedPredictor:
    """Predictor engine running EfficientNet-B0 inference on animal image crops."""

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        class_mapping_path: Optional[Union[Path, str]] = None,
        device: Optional[str] = None,
    ):
        """Initialize BreedPredictor.

        Args:
            model_path: Path to trained PyTorch weights (.pth). If None, checks models/efficientnet_best.pth.
            class_mapping_path: Path to class_mapping.json file.
            device: Compute device ('cpu' or 'cuda:0').
        """
        self.device = torch.device(
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # Load class mapping
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        self.num_classes = len(self.class_mapping)

        # Initialize preprocessor
        self.preprocessor = OpenCVPreprocessor(target_size=(224, 224))

        # Instantiate model architecture
        self.model = BreedClassifier(num_classes=self.num_classes, pretrained=False)

        # Load weights if checkpoint exists
        if model_path is None:
            default_weights = Path("models/efficientnet_best.pth")
            if default_weights.exists():
                model_path = default_weights

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_class_mapping(
        self, class_mapping_path: Optional[Union[Path, str]]
    ) -> Dict[int, Dict[str, str]]:
        """Load class mapping dictionary from JSON file or default map."""
        if class_mapping_path and Path(class_mapping_path).exists():
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
                return {int(k): v for k, v in raw_json.items()}

        default_json = Path("configs/class_mapping.json")
        if default_json.exists():
            with open(default_json, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
                return {int(k): v for k, v in raw_json.items()}

        return DEFAULT_CLASS_MAPPING

    def predict(
        self,
        source: Union[Path, str, bytes, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        top_k: int = 3,
    ) -> PredictionResult:
        """Run breed classification inference on image or cropped ROI.

        Args:
            source: Image path, string path, bytes, or numpy array.
            bbox: Optional bounding box tuple (xmin, ymin, xmax, ymax) to crop prior to classification.
            top_k: Number of top predictions to return (default 3).

        Returns:
            PredictionResult Pydantic DTO.
        """
        # Preprocess input image to tensor (3, 224, 224)
        tensor_img = self.preprocessor.preprocess_to_tensor(source, bbox=bbox)
        batch_tensor = tensor_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        # Retrieve Top-K probabilities and indices
        k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probs, k=k, largest=True, sorted=True)

        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        top_3_items: List[BreedPredictionItem] = []
        for idx, prob in zip(top_indices, top_probs):
            cls_id = int(idx)
            info = self.class_mapping.get(
                cls_id,
                {
                    "breed_name": f"Breed_{cls_id}",
                    "animal_type": "cattle",
                    "display_name": f"Breed {cls_id}",
                },
            )

            item = BreedPredictionItem(
                class_id=cls_id,
                breed_name=info["breed_name"],
                display_name=info["display_name"],
                animal_type=info["animal_type"],
                confidence=float(prob),
            )
            top_3_items.append(item)

        top_1 = top_3_items[0]
        return PredictionResult(
            predicted_breed=top_1.breed_name,
            display_name=top_1.display_name,
            animal_type=top_1.animal_type,
            confidence=top_1.confidence,
            top_3_predictions=top_3_items,
        )
