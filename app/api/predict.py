"""API Endpoint for AI Breed Recognition Prediction and Grad-CAM Explainability."""

import base64
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status, Depends

from app.schemas.predict import (
    PredictResponseSchema,
    GradCAMResponseSchema,
    Top3PredictionSchema,
    InferenceTimeSchema,
)
from ml.pipeline.inference_pipeline import BreedRecognitionPipeline, InferenceResult

router = APIRouter(tags=["Breed Prediction"])

# Global inference pipeline instance
_pipeline_instance: Optional[BreedRecognitionPipeline] = None


def get_inference_pipeline() -> BreedRecognitionPipeline:
    """Dependency provider for BreedRecognitionPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = BreedRecognitionPipeline()
    return _pipeline_instance


MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def numpy_to_base64_png(img_rgb: np.ndarray) -> str:
    """Convert RGB numpy array image into Base64 encoded PNG data URL string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@router.post(
    "/predict",
    response_model=PredictResponseSchema,
    summary="Predict Cattle or Buffalo Breed from Image",
    description="Upload an image file (multipart/form-data) to perform YOLO animal detection, EfficientNet-B0 breed classification, and optional Grad-CAM visual explainability.",
)
async def predict_breed(
    file: UploadFile = File(..., description="Uploaded animal image file"),
    generate_gradcam: bool = Query(
        True, description="Whether to compute Grad-CAM explainability heatmap overlay"
    ),
    pipeline: BreedRecognitionPipeline = Depends(get_inference_pipeline),
) -> PredictResponseSchema:
    """Predict Indian Cattle and Buffalo breed from uploaded image file."""
    # 1. Validate File Format & Extension
    filename = file.filename or ""
    extension = filename.lower().split(".")[-1] if "." in filename else ""
    mime_type = file.content_type or ""

    if mime_type.lower() not in ALLOWED_MIME_TYPES and f".{extension}" not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image format '{mime_type}'. "
                "Allowed image formats: JPEG, PNG, WebP, BMP, TIFF."
            ),
        )

    # 2. Safely Read Uploaded File Bytes and Enforce Size Limit
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file payload: {str(e)}",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file payload is zero bytes.",
        )

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size ({len(contents) / (1024*1024):.2f}MB) exceeds maximum limit of 10MB.",
        )

    # 3. Execute End-to-End AI Inference Pipeline
    pipeline_res: InferenceResult = pipeline.predict(
        source=contents, generate_gradcam=generate_gradcam
    )

    if pipeline_res.prediction_status == "invalid_image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be decoded as a valid image.",
        )

    # 4. Construct Grad-CAM Base64 Response Payload
    gradcam_schema: Optional[GradCAMResponseSchema] = None
    if pipeline_res.gradcam_output and "overlay_image" in pipeline_res.gradcam_output:
        overlay_rgb = pipeline_res.gradcam_output["overlay_image"]
        overlay_b64 = numpy_to_base64_png(overlay_rgb)

        heatmap_b64 = None
        if "heatmap" in pipeline_res.gradcam_output:
            heatmap_norm = pipeline_res.gradcam_output["heatmap"]
            heatmap_uint8 = np.uint8(255 * heatmap_norm)
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            heatmap_b64 = numpy_to_base64_png(heatmap_rgb)

        gradcam_schema = GradCAMResponseSchema(
            target_class_id=pipeline_res.gradcam_output.get("target_class_id", 0),
            display_name=pipeline_res.gradcam_output.get("display_name", pipeline_res.predicted_breed),
            overlay_base64=overlay_b64,
            heatmap_base64=heatmap_b64,
        )

    # 5. Build Final Response DTO
    top_3_items = [
        Top3PredictionSchema(
            class_id=p["class_id"],
            breed_name=p["breed_name"],
            display_name=p["display_name"],
            animal_type=p["animal_type"],
            confidence=p["confidence"],
        )
        for p in pipeline_res.top_3_predictions
    ]

    timing_schema = InferenceTimeSchema(
        detection_ms=pipeline_res.inference_time["detection_ms"],
        classification_ms=pipeline_res.inference_time["classification_ms"],
        gradcam_ms=pipeline_res.inference_time["gradcam_ms"],
        total_ms=pipeline_res.inference_time["total_ms"],
    )

    return PredictResponseSchema(
        animal_type=pipeline_res.animal_type,
        animal_confidence=pipeline_res.animal_confidence,
        bounding_box=pipeline_res.bounding_box,
        predicted_breed=pipeline_res.predicted_breed,
        breed_confidence=pipeline_res.breed_confidence,
        top_3_predictions=top_3_items,
        prediction_status=pipeline_res.prediction_status,
        inference_time=timing_schema,
        model_versions=pipeline_res.model_versions,
        gradcam_output=gradcam_schema,
    )
