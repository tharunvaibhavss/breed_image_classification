"""Unit and integration tests for Database ORM models, Repositories, and Seeding."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from db.base import Base
from db.models import User, Breed, ImageMetadata, ModelVersion, Prediction
from db.repository import (
    UserRepository,
    BreedRepository,
    ImageMetadataRepository,
    ModelVersionRepository,
    PredictionRepository,
)
from db.seed import seed_database


@pytest.fixture(scope="function")
def db_session() -> Session:
    """Fixture initializing an in-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_user_repository_crud(db_session: Session):
    """Verify UserRepository creation and lookup methods."""
    user = UserRepository.create_user(
        db=db_session,
        email="researcher@cattle.ai",
        hashed_password="hashed_secure_pass_123",
        full_name="Dr. Agricultural Scientist",
    )

    assert user.id is not None
    assert user.email == "researcher@cattle.ai"
    assert user.is_active is True
    assert user.is_superuser is False

    fetched = UserRepository.get_by_email(db_session, "RESEARCHER@cattle.ai")
    assert fetched is not None
    assert fetched.id == user.id


def test_breed_repository_crud(db_session: Session):
    """Verify BreedRepository creation and metadata queries."""
    breed = BreedRepository.create_breed(
        db=db_session,
        breed_name="Gir",
        animal_type="cattle",
        origin="Gir Hills",
        native_state="Gujarat",
        physical_characteristics={"horns": "curved", "ears": "pendulous"},
        milk_production={"average_yield_kg": 2182},
        climate_adaptability="High heat tolerance",
        uses="dairy",
        description="Famous Indian zebu dairy breed",
    )

    assert breed.id is not None
    assert breed.breed_name == "Gir"
    assert breed.animal_type == "cattle"

    fetched = BreedRepository.get_by_name(db_session, "Gir")
    assert fetched is not None
    assert fetched.native_state == "Gujarat"


def test_image_and_model_version_repository(db_session: Session):
    """Verify ImageMetadata and ModelVersion CRUD operations."""
    image = ImageMetadataRepository.create_image(
        db=db_session,
        filename="test_cow.jpg",
        file_path="data/raw/dataset/cattle/Gir/img_1.jpg",
        file_size_bytes=1048576,
        mime_type="image/jpeg",
        width=1920,
        height=1080,
        md5_hash="d41d8cd98f00b204e9800998ecf8427e",
    )
    assert image.id is not None

    version = ModelVersionRepository.create_version(
        db=db_session,
        model_name="BreedNet-v1",
        yolo_version="YOLOv8n",
        efficientnet_version="EfficientNet-B0",
        gradcam_version="Grad-CAM-v1",
    )
    assert version.id is not None

    active_v = ModelVersionRepository.get_active(db_session)
    assert active_v is not None
    assert active_v.id == version.id


def test_prediction_repository_and_relationships(db_session: Session):
    """Verify Prediction creation and foreign key relationships."""
    user = UserRepository.create_user(db_session, "user@test.org", "pass")
    breed = BreedRepository.create_breed(
        db_session, "Murrah", "buffalo", "Rohtak", "Haryana", {}, {}, "High", "dairy", "Desc"
    )
    image = ImageMetadataRepository.create_image(db_session, "buff.png", "/path", 500, "image/png")
    version = ModelVersionRepository.create_version(db_session, "V1", "Y1", "E1", "G1")

    pred = PredictionRepository.create_prediction(
        db=db_session,
        image_id=image.id,
        user_id=user.id,
        model_version_id=version.id,
        predicted_breed_id=breed.id,
        animal_type="buffalo",
        animal_confidence=0.96,
        bounding_box=[10, 20, 200, 200],
        predicted_breed_name="Murrah",
        breed_confidence=0.92,
        top_3_predictions=[{"breed_name": "Murrah", "confidence": 0.92}],
        prediction_status="success",
        inference_time_ms={"total_ms": 42.5},
    )

    assert pred.id is not None
    assert pred.image.filename == "buff.png"
    assert pred.user.email == "user@test.org"
    assert pred.breed_ref.breed_name == "Murrah"
    assert pred.model_version.model_name == "V1"


def test_database_seeding_script(db_session: Session):
    """Verify database seed script populates 6 breeds and active model version."""
    seed_database(db_session)

    all_breeds = BreedRepository.list_all(db_session)
    assert len(all_breeds) == 6

    names = [b.breed_name for b in all_breeds]
    assert "Gir" in names
    assert "Ongole" in names
    assert "Sahiwal" in names
    assert "Jaffarabadi" in names
    assert "Murrah" in names
    assert "Surti" in names

    gir = BreedRepository.get_by_name(db_session, "Gir")
    assert gir.native_state == "Gujarat"
    assert gir.milk_production["average_lactation_yield_kg"] == 2182

    active_v = ModelVersionRepository.get_active(db_session)
    assert active_v is not None
    assert active_v.yolo_version == "YOLOv8n-AnimalDetection-v1.0"
