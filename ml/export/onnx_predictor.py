"""ONNX Runtime inference predictor for EfficientNet-B0 breed classification."""

from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np
import torch
import onnxruntime as ort
from app.core.logging import setup_logging
from ml.common.breed_registry import BreedRegistry

logger = setup_logging()
default_registry = BreedRegistry()


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """Compute numpy numerically stable Softmax probabilities."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class ONNXBreedPredictor:
    """ONNX Runtime accelerated breed classification predictor."""

    def __init__(
        self,
        onnx_model_path: str = "models/onnx/efficientnet_b0.onnx",
        class_mapping: Optional[Dict[int, str]] = None,
        providers: Optional[List[str]] = None,
    ):
        """Initialize ONNX Runtime inference session.

        Args:
            onnx_model_path: Path to .onnx model file.
            class_mapping: Optional dictionary mapping class index to breed name.
            providers: Execution providers list (e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        """
        self.onnx_path = Path(onnx_model_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

        if class_mapping is None:
            class_mapping = {
                b.class_id: b.breed_name for b in default_registry.get_all_breeds()
            }
        self.class_mapping = class_mapping
        self.id_to_display = {
            b.class_id: b.display_name for b in default_registry.get_all_breeds()
        }

        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]

        logger.info("Initializing ONNX Runtime session with providers: %s", providers)
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(
        self, image_input: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """Perform breed classification using ONNX Runtime.

        Args:
            image_input: PyTorch Tensor or NumPy array of shape (1, 3, 224, 224) or (3, 224, 224).

        Returns:
            Dictionary containing predicted_breed, confidence, top_3_predictions, raw_logits, and probabilities.
        """
        if isinstance(image_input, torch.Tensor):
            input_np = image_input.detach().cpu().numpy()
        else:
            input_np = image_input

        if input_np.ndim == 3:
            input_np = np.expand_dims(input_np, axis=0)

        input_np = input_np.astype(np.float32)

        # Execute ONNX Runtime Forward Pass
        outputs = self.session.run([self.output_name], {self.input_name: input_np})
        logits = outputs[0]  # Shape: (1, 6)

        probabilities = softmax_numpy(logits)[0]  # Shape: (6,)
        top1_idx = int(np.argmax(probabilities))

        raw_val = self.class_mapping.get(top1_idx, f"Class_{top1_idx}")
        if isinstance(raw_val, dict):
            predicted_breed = raw_val.get("breed_name", f"Class_{top1_idx}")
            display_name = raw_val.get("display_name", predicted_breed)
        else:
            predicted_breed = str(raw_val)
            display_name = self.id_to_display.get(top1_idx, predicted_breed)

        confidence = float(probabilities[top1_idx])

        # Top-3 predictions ranking
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3_predictions = []
        for idx in top3_indices:
            v = self.class_mapping.get(int(idx), f"Class_{idx}")
            b_name = v.get("breed_name", f"Class_{idx}") if isinstance(v, dict) else str(v)
            d_name = v.get("display_name", b_name) if isinstance(v, dict) else self.id_to_display.get(int(idx), b_name)
            top3_predictions.append(
                {
                    "class_id": int(idx),
                    "breed_name": b_name,
                    "display_name": d_name,
                    "confidence": float(probabilities[idx]),
                }
            )

        return {
            "predicted_breed": predicted_breed,
            "display_name": display_name,
            "confidence": confidence,
            "top_3_predictions": top3_predictions,
            "raw_logits": logits[0].tolist(),
            "probabilities": probabilities.tolist(),
        }
