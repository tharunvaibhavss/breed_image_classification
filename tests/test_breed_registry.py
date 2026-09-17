"""Unit tests for BreedRegistry module."""

from pathlib import Path
from ml.common.breed_registry import BreedRegistry, BreedInfo


def test_default_breed_registry_initialization():
    """Verify registry initializes with expected cattle and buffalo breeds."""
    registry = BreedRegistry()
    breeds = registry.get_all_breeds()

    assert len(breeds) == 6

    # Verify Cattle breeds
    gir = registry.get_by_name("Gir")
    assert gir is not None
    assert gir.class_id == 0
    assert gir.animal_type == "cattle"

    ongole = registry.get_by_name("Ongole")
    assert ongole is not None
    assert ongole.class_id == 1
    assert ongole.animal_type == "cattle"

    sahiwal = registry.get_by_name("Sahiwal")
    assert sahiwal is not None
    assert sahiwal.class_id == 2
    assert sahiwal.animal_type == "cattle"

    # Verify Buffalo breeds
    jaffarabadi = registry.get_by_name("Jaffarabadi")
    assert jaffarabadi is not None
    assert jaffarabadi.class_id == 3
    assert jaffarabadi.animal_type == "buffalo"

    murrah = registry.get_by_name("Murrah")
    assert murrah is not None
    assert murrah.class_id == 4
    assert murrah.animal_type == "buffalo"

    surti = registry.get_by_name("Surti")
    assert surti is not None
    assert surti.class_id == 5
    assert surti.animal_type == "buffalo"


def test_register_new_breed():
    """Test manual registration of a new breed."""
    registry = BreedRegistry()
    new_breed = registry.register_breed(
        class_id=6,
        breed_name="Bhadawari",
        animal_type="buffalo",
        display_name="Bhadawari Buffalo",
    )

    assert new_breed.class_id == 6
    assert registry.get_by_id(6).breed_name == "Bhadawari"
    assert registry.get_by_name("bhadawari").animal_type == "buffalo"


def test_dynamic_discovery_from_temp_dir(tmp_path: Path):
    """Test dynamic breed discovery from folder structure."""
    registry = BreedRegistry()

    # Create dummy dataset directory structure
    cattle_dir = tmp_path / "cattle" / "Kankrej"
    cattle_dir.mkdir(parents=True)

    discovered = registry.discover_from_dataset_dir(tmp_path)
    assert len(discovered) == 1
    assert discovered[0].breed_name == "Kankrej"
    assert discovered[0].animal_type == "cattle"
    assert registry.get_by_name("Kankrej") is not None
