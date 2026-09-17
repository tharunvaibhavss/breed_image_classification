"""ONNX Exporter module for PyTorch EfficientNet-B0 and YOLO model conversion."""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import torch
import torch.nn as nn
import onnx
from app.core.logging import setup_logging

logger = setup_logging()


class ONNXExporter:
    """Handles conversion of PyTorch deep learning models to ONNX format."""

    @staticmethod
    def export_efficientnet(
        model: nn.Module,
        output_path: str = "models/onnx/efficientnet_b0.onnx",
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        dynamic_axes: bool = True,
    ) -> Path:
        """Export PyTorch EfficientNet-B0 breed classification model to ONNX.

        Args:
            model: PyTorch model instance in eval mode.
            output_path: Target destination file path.
            input_shape: Input tensor dimensions (N, C, H, W).
            dynamic_axes: Whether to support dynamic batch sizes.

        Returns:
            Resolved Path to exported ONNX model.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_input = torch.randn(*input_shape, dtype=torch.float32)

        dynamic_dims = None
        if dynamic_axes:
            dynamic_dims = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        logger.info("Exporting EfficientNet-B0 to ONNX format: %s", out_path)
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_dims,
        )

        # Verify graph structure
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        logger.info(
            "ONNX export & graph verification successful. File size: %.2f MB",
            out_path.stat().st_size / (1024 * 1024),
        )

        return out_path

    @staticmethod
    def export_yolo(
        yolo_weights_path: str = "models/yolo/best.pt",
        output_dir: str = "models/onnx",
    ) -> Optional[Path]:
        """Export YOLO detection model to ONNX using Ultralytics exporter.

        Args:
            yolo_weights_path: Path to PyTorch YOLO weights checkpoint.
            output_dir: Output directory.

        Returns:
            Path to exported ONNX file or None if weights not present.
        """
        weights_p = Path(yolo_weights_path)
        if not weights_p.exists():
            logger.warning(
                "YOLO weights '%s' not found. Skipping YOLO ONNX export.", yolo_weights_path
            )
            return None

        try:
            from ultralytics import YOLO

            yolo = YOLO(str(weights_p))
            logger.info("Exporting YOLO model to ONNX: %s", weights_p)
            exported_path = yolo.export(format="onnx", dynamic=True)
            logger.info("YOLO ONNX export successful: %s", exported_path)
            return Path(exported_path)
        except Exception as e:
            logger.error("Failed to export YOLO to ONNX: %s", e)
            return None

    @staticmethod
    def verify_onnx(onnx_path: str) -> bool:
        """Verify ONNX model file graph structure.

        Args:
            onnx_path: File path to ONNX model.

        Returns:
            True if model passes graph validation, False otherwise.
        """
        try:
            onnx_p = Path(onnx_path)
            if not onnx_p.exists():
                return False
            model = onnx.load(str(onnx_p))
            onnx.checker.check_model(model)
            return True
        except Exception as e:
            logger.error("ONNX verification failed for '%s': %s", onnx_path, e)
            return False
