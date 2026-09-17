"""Breed Registry module for Cattle and Buffalo Breeds.

Manages class mapping, metadata, animal type categorization, and dynamic breed
discovery from dataset structures.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class BreedInfo(BaseModel):
    """Data model for individual breed registration entry."""

    class_id: int = Field(description="Unique zero-indexed integer class identifier")
    breed_name: str = Field(description="Standardized breed name (e.g. 'Gir', 'Murrah')")
    animal_type: str = Field(description="Animal classification ('cattle' or 'buffalo')")
    display_name: str = Field(description="Human-readable display name (e.g. 'Gir Cattle')")
    enabled: bool = Field(default=True, description="Flag indicating if breed class is active")


DEFAULT_BREEDS: List[Dict[str, Any]] = [
    {
        "class_id": 0,
        "breed_name": "Gir",
        "animal_type": "cattle",
        "display_name": "Gir Cattle",
        "enabled": True,
    },
    {
        "class_id": 1,
        "breed_name": "Ongole",
        "animal_type": "cattle",
        "display_name": "Ongole Cattle",
        "enabled": True,
    },
    {
        "class_id": 2,
        "breed_name": "Sahiwal",
        "animal_type": "cattle",
        "display_name": "Sahiwal Cattle",
        "enabled": True,
    },
    {
        "class_id": 3,
        "breed_name": "Jaffarabadi",
        "animal_type": "buffalo",
        "display_name": "Jaffarabadi Buffalo",
        "enabled": True,
    },
    {
        "class_id": 4,
        "breed_name": "Murrah",
        "animal_type": "buffalo",
        "display_name": "Murrah Buffalo",
        "enabled": True,
    },
    {
        "class_id": 5,
        "breed_name": "Surti",
        "animal_type": "buffalo",
        "display_name": "Surti Buffalo",
        "enabled": True,
    },
]


class BreedRegistry:
    """Configurable Breed Registry managing class mappings and dynamic discovery."""

    def __init__(self, breeds: Optional[List[Dict[str, Any]]] = None):
        """Initialize BreedRegistry with default or custom breed definitions."""
        self._breeds: Dict[int, BreedInfo] = {}
        self._name_to_id: Dict[str, int] = {}

        initial_breeds = breeds if breeds is not None else DEFAULT_BREEDS
        for entry in initial_breeds:
            self.register_breed(
                class_id=entry["class_id"],
                breed_name=entry["breed_name"],
                animal_type=entry["animal_type"],
                display_name=entry.get("display_name"),
                enabled=entry.get("enabled", True),
            )

    def register_breed(
        self,
        class_id: int,
        breed_name: str,
        animal_type: str,
        display_name: Optional[str] = None,
        enabled: bool = True,
    ) -> BreedInfo:
        """Register or update a breed entry in the registry."""
        normalized_name = breed_name.strip()
        formatted_display = (
            display_name if display_name else f"{normalized_name} {animal_type.capitalize()}"
        )

        breed_info = BreedInfo(
            class_id=class_id,
            breed_name=normalized_name,
            animal_type=animal_type.lower(),
            display_name=formatted_display,
            enabled=enabled,
        )

        self._breeds[class_id] = breed_info
        self._name_to_id[normalized_name.lower()] = class_id
        return breed_info

    def get_by_id(self, class_id: int) -> Optional[BreedInfo]:
        """Lookup breed metadata by integer class_id."""
        return self._breeds.get(class_id)

    def get_by_name(self, breed_name: str) -> Optional[BreedInfo]:
        """Lookup breed metadata by case-insensitive breed_name."""
        class_id = self._name_to_id.get(breed_name.strip().lower())
        if class_id is not None:
            return self._breeds.get(class_id)
        return None

    def get_all_breeds(self, enabled_only: bool = False) -> List[BreedInfo]:
        """Retrieve list of registered breeds ordered by class_id."""
        breeds = sorted(self._breeds.values(), key=lambda b: b.class_id)
        if enabled_only:
            return [b for b in breeds if b.enabled]
        return breeds

    def discover_from_dataset_dir(self, dataset_dir: Path) -> List[BreedInfo]:
        """Scan dataset directory structure (animal_type/breed_name) and register new breeds.

        Args:
            dataset_dir: Path to raw dataset root (e.g. data/raw/dataset)

        Returns:
            List of newly discovered and registered BreedInfo instances.
        """
        dataset_path = Path(dataset_dir)
        newly_registered: List[BreedInfo] = []

        if not dataset_path.exists() or not dataset_path.is_dir():
            return newly_registered

        # Recognized animal top-level directories
        animal_types = ["cattle", "buffalo"]

        for animal_type in animal_types:
            type_dir = dataset_path / animal_type
            if not type_dir.exists() or not type_dir.is_dir():
                continue

            for sub_path in sorted(type_dir.iterdir()):
                if sub_path.is_dir() and not sub_path.name.startswith("."):
                    breed_name = sub_path.name
                    existing = self.get_by_name(breed_name)
                    if existing is None:
                        # Assign next available class_id
                        next_id = (
                            max(self._breeds.keys()) + 1 if self._breeds else 0
                        )
                        new_breed = self.register_breed(
                            class_id=next_id,
                            breed_name=breed_name,
                            animal_type=animal_type,
                        )
                        newly_registered.append(new_breed)

        return newly_registered

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry to dictionary."""
        return {
            "total_classes": len(self._breeds),
            "breeds": [b.model_dump() for b in self.get_all_breeds()],
        }

    def save_json(self, json_path: Path) -> None:
        """Save registry configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
