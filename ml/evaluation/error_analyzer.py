"""Error Analysis Engine module for analyzing misclassifications and confidence distributions.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from ml.evaluation.evaluator import CLASS_NAMES


class ErrorAnalyzer:
    """Error analysis engine identifying confusion pairs, low confidence, and error patterns."""

    def analyze_errors(
        self,
        y_true: List[int],
        y_pred_probs: List[List[float]],
        manifest_samples: Optional[List[Dict[str, Any]]] = None,
        confidence_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Perform error analysis on evaluation predictions.

        Args:
            y_true: List of ground truth class integer labels.
            y_pred_probs: List of predicted probability lists (N, C).
            manifest_samples: Optional list of test sample metadata dicts.
            confidence_threshold: Confidence threshold for low-confidence flag (default 0.50).

        Returns:
            Dict containing error statistics, confusion pairs, low confidence counts, and sample lists.
        """
        y_true_np = np.array(y_true, dtype=int)
        y_probs_np = np.array(y_pred_probs, dtype=np.float32)

        if len(y_true_np) == 0 or len(y_probs_np) == 0:
            return {
                "total_eval_samples": 0,
                "total_errors": 0,
                "error_rate_percent": 0.0,
                "total_low_confidence_samples": 0,
                "top_breed_confusion_pairs": [],
                "incorrect_predictions": [],
                "correct_predictions": [],
            }

        y_pred_np = np.argmax(y_probs_np, axis=1)
        confidences = np.max(y_probs_np, axis=1)

        total_samples = len(y_true_np)
        incorrect_indices = np.where(y_true_np != y_pred_np)[0]
        correct_indices = np.where(y_true_np == y_pred_np)[0]

        low_conf_indices = np.where(confidences < confidence_threshold)[0]

        # 1. Identify breed confusion pairs (True Breed -> Predicted Breed)
        confusion_pairs_dict: Dict[Tuple[str, str], int] = {}
        for idx in incorrect_indices:
            t_cls = y_true_np[idx]
            p_cls = y_pred_np[idx]
            t_name = CLASS_NAMES[t_cls] if t_cls < len(CLASS_NAMES) else f"Class_{t_cls}"
            p_name = CLASS_NAMES[p_cls] if p_cls < len(CLASS_NAMES) else f"Class_{p_cls}"

            pair_key = (t_name, p_name)
            confusion_pairs_dict[pair_key] = confusion_pairs_dict.get(pair_key, 0) + 1

        # Sort top confusion pairs
        sorted_confusion_pairs = sorted(
            [
                {
                    "true_breed": key[0],
                    "predicted_breed": key[1],
                    "misclassified_count": count,
                }
                for key, count in confusion_pairs_dict.items()
            ],
            key=lambda x: x["misclassified_count"],
            reverse=True,
        )

        # 2. Build list of incorrect predictions
        incorrect_predictions: List[Dict[str, Any]] = []
        for idx in incorrect_indices:
            t_cls = int(y_true_np[idx])
            p_cls = int(y_pred_np[idx])
            t_name = CLASS_NAMES[t_cls] if t_cls < len(CLASS_NAMES) else f"Class_{t_cls}"
            p_name = CLASS_NAMES[p_cls] if p_cls < len(CLASS_NAMES) else f"Class_{p_cls}"
            conf = float(confidences[idx])

            sample_meta = (
                manifest_samples[idx]
                if manifest_samples and idx < len(manifest_samples)
                else {}
            )

            incorrect_predictions.append(
                {
                    "sample_index": int(idx),
                    "file_path": sample_meta.get("file_path", f"sample_{idx}.jpg"),
                    "true_breed": t_name,
                    "predicted_breed": p_name,
                    "confidence": round(conf, 4),
                    "is_low_confidence": bool(conf < confidence_threshold),
                    "width": sample_meta.get("width", 0),
                    "height": sample_meta.get("height", 0),
                    "aspect_ratio": sample_meta.get("aspect_ratio", 0.0),
                }
            )

        # 3. Build sample list of correct predictions
        correct_predictions: List[Dict[str, Any]] = []
        for idx in correct_indices[:10]:  # Cap at top 10 correct samples
            t_cls = int(y_true_np[idx])
            t_name = CLASS_NAMES[t_cls] if t_cls < len(CLASS_NAMES) else f"Class_{t_cls}"
            conf = float(confidences[idx])

            sample_meta = (
                manifest_samples[idx]
                if manifest_samples and idx < len(manifest_samples)
                else {}
            )

            correct_predictions.append(
                {
                    "sample_index": int(idx),
                    "file_path": sample_meta.get("file_path", f"sample_{idx}.jpg"),
                    "breed": t_name,
                    "confidence": round(conf, 4),
                }
            )

        error_rate = (len(incorrect_indices) / total_samples) * 100.0 if total_samples > 0 else 0.0

        return {
            "total_eval_samples": total_samples,
            "total_errors": len(incorrect_indices),
            "error_rate_percent": round(error_rate, 2),
            "total_low_confidence_samples": len(low_conf_indices),
            "confidence_threshold_used": confidence_threshold,
            "top_breed_confusion_pairs": sorted_confusion_pairs,
            "incorrect_predictions": incorrect_predictions,
            "sample_correct_predictions": correct_predictions,
        }
