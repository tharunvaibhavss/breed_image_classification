"""Classification Metrics Calculator module.

Computes Loss, Top-1 Accuracy, Top-3 Accuracy, Precision, Recall, and F1-Score.
"""

from typing import Dict, Any, List, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def calculate_topk_accuracy(
    output_logits: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    topk: Tuple[int, ...] = (1, 3),
) -> Dict[str, float]:
    """Calculate Top-1 and Top-K classification accuracy.

    Args:
        output_logits: Tensor or numpy array of shape (N, C).
        targets: Tensor or numpy array of ground truth labels (N,).
        topk: Tuple of K values (e.g. (1, 3)).

    Returns:
        Dict mapping metric name to accuracy percentage.
    """
    if isinstance(output_logits, np.ndarray):
        output_logits = torch.from_numpy(output_logits)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    maxk = max(topk)
    batch_size = targets.size(0)

    if batch_size == 0:
        return {f"top_{k}_acc": 0.0 for k in topk}

    num_classes = output_logits.size(1)
    actual_k = min(maxk, num_classes)

    _, pred = output_logits.topk(actual_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res: Dict[str, float] = {}
    for k in topk:
        k_val = min(k, num_classes)
        correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
        res[f"top_{k}_acc"] = round((correct_k.item() / batch_size) * 100.0, 2)

    return res


def calculate_classification_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred_probs: Union[np.ndarray, torch.Tensor],
    num_classes: int = 6,
) -> Dict[str, Any]:
    """Compute complete classification evaluation metrics.

    Args:
        y_true: Ground truth integer class labels array (N,).
        y_pred_probs: Predicted class probabilities or logits array (N, C).
        num_classes: Number of classes (default 6).

    Returns:
        Dict containing accuracy, top_3_acc, precision, recall, f1_macro, f1_weighted.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.cpu().numpy()

    y_true = np.array(y_true, dtype=np.int64)
    if len(y_true) == 0 or len(y_pred_probs) == 0:
        return {
            "accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }

    y_pred = np.argmax(y_pred_probs, axis=1)

    # Top-1 and Top-3 accuracy
    topk_res = calculate_topk_accuracy(y_pred_probs, y_true, topk=(1, 3))
    acc = topk_res["top_1_acc"]
    top3_acc = topk_res["top_3_acc"]

    # Precision, Recall, and F1 per class and macro/weighted average
    precisions = []
    recalls = []
    f1s = []
    class_counts = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        class_counts.append(np.sum(y_true == c))

    macro_precision = float(np.mean(precisions)) * 100.0
    macro_recall = float(np.mean(recalls)) * 100.0
    macro_f1 = float(np.mean(f1s)) * 100.0

    total_samples = len(y_true)
    weighted_f1 = (
        float(np.sum(np.array(f1s) * np.array(class_counts)) / total_samples) * 100.0
        if total_samples > 0
        else 0.0
    )

    return {
        "accuracy": round(acc, 2),
        "top_3_accuracy": round(top3_acc, 2),
        "precision_macro": round(macro_precision, 2),
        "recall_macro": round(macro_recall, 2),
        "f1_macro": round(macro_f1, 2),
        "f1_weighted": round(weighted_f1, 2),
    }
