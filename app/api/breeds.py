"""API Endpoint listing supported Indian Cattle and Buffalo breeds."""

from fastapi import APIRouter
from app.schemas.predict import BreedsListResponseSchema, BreedInfoSchema
from ml.common.breed_registry import BreedRegistry

router = APIRouter(tags=["Breed Metadata"])

BREED_DESCRIPTIONS = {
    "Gir": {
        "origin_region": "Gujarat (Gir Hills / Kathiawar)",
        "description": "Indigenous dairy cattle breed known for distinctive rounded forehead, long pendulous ears, and high milk yield.",
    },
    "Ongole": {
        "origin_region": "Andhra Pradesh (Prakasam District)",
        "description": "Famous dual-purpose draft and dairy cattle breed recognized by muscular hump, white coat, and high disease resistance.",
    },
    "Sahiwal": {
        "origin_region": "Punjab / Haryana (Indo-Pak Border Region)",
        "description": "Premier indigenous zebu dairy cattle breed known for reddish-brown coat and high fat content milk yield.",
    },
    "Jaffarabadi": {
        "origin_region": "Gujarat (Saurashtra / Gir Forest)",
        "description": "Massive Indian riverine buffalo breed characterized by heavy drooping horns and exceptional butterfat milk.",
    },
    "Murrah": {
        "origin_region": "Haryana / Punjab (Rohtak / Hisar)",
        "description": "World-renowned black riverine buffalo breed recognized by tightly curled horns and top milk productivity.",
    },
    "Surti": {
        "origin_region": "Gujarat (Surat / Kheda)",
        "description": "Medium-sized riverine buffalo breed known for sickle-shaped horns, compact body, and high fat percentage milk.",
    },
}


@router.get(
    "/breeds",
    response_model=BreedsListResponseSchema,
    summary="Get List of Supported Indian Cattle and Buffalo Breeds",
    description="Returns metadata catalog for all supported indigenous Indian cattle and buffalo breeds.",
)
async def get_supported_breeds() -> BreedsListResponseSchema:
    """Retrieve catalog of supported Indian cattle and buffalo breeds."""
    registry = BreedRegistry()
    all_breeds = registry.get_all_breeds(enabled_only=True)

    breed_schemas = []
    for info in all_breeds:
        meta = BREED_DESCRIPTIONS.get(
            info.breed_name,
            {
                "origin_region": "India",
                "description": f"Indigenous Indian {info.animal_type} breed.",
            },
        )

        breed_schemas.append(
            BreedInfoSchema(
                breed_name=info.breed_name,
                display_name=info.display_name,
                animal_type=info.animal_type,
                origin_region=meta["origin_region"],
                description=meta["description"],
            )
        )

    return BreedsListResponseSchema(
        total_breeds=len(breed_schemas), breeds=breed_schemas
    )
