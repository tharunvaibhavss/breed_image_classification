"""API Endpoint returning AI Model architecture details, version metadata, and environment status."""

from pathlib import Path
from fastapi import APIRouter
from app.schemas.predict import ModelInfoResponseSchema
from ml.common.env_check import get_pytorch_environment_info

router = APIRouter(tags=["Model Metadata"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponseSchema,
    summary="Get AI Model Information and System Status",
    description="Returns AI model architecture versions, checkpoint presence status, supported species/breeds count, and PyTorch hardware compute info.",
)
async def get_model_information() -> ModelInfoResponseSchema:
    """Retrieve AI model architecture and environment details."""
    models_dir = Path("models")
    yolo_checkpoint = models_dir / "yolo_best.pt"
    eff_checkpoint = models_dir / "efficientnet_best.pth"

    env_info = get_pytorch_environment_info()

    return ModelInfoResponseSchema(
        system_name="AI-Powered Breed Recognition System for Indian Cattle and Buffaloes",
        pipeline_version="1.0.0",
        supported_species=["cattle", "buffalo"],
        supported_breeds_count=6,
        checkpoints_status={
            "yolo_detector_best_pt": yolo_checkpoint.exists(),
            "efficientnet_classifier_best_pth": eff_checkpoint.exists(),
        },
        environment_info=env_info,
    )
