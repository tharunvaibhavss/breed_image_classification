"""Database Repository CRUD operations for User, Breed, ImageMetadata, ModelVersion, and Prediction."""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from db.models import User, Breed, ImageMetadata, ModelVersion, Prediction


class UserRepository:
    """CRUD repository for User entity."""

    @staticmethod
    def create_user(
        db: Session,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
        is_superuser: bool = False,
    ) -> User:
        user = User(
            email=email.lower().strip(),
            hashed_password=hashed_password,
            full_name=full_name,
            is_superuser=is_superuser,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email.lower().strip())
        return db.scalar(stmt)

    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        return db.get(User, user_id)


class BreedRepository:
    """CRUD repository for Breed entity."""

    @staticmethod
    def create_breed(
        db: Session,
        breed_name: str,
        animal_type: str,
        origin: str,
        native_state: str,
        physical_characteristics: Dict[str, Any],
        milk_production: Dict[str, Any],
        climate_adaptability: str,
        uses: str,
        description: str,
    ) -> Breed:
        breed = Breed(
            breed_name=breed_name.strip(),
            animal_type=animal_type.lower().strip(),
            origin=origin,
            native_state=native_state,
            physical_characteristics=physical_characteristics,
            milk_production=milk_production,
            climate_adaptability=climate_adaptability,
            uses=uses,
            description=description,
        )
        db.add(breed)
        db.commit()
        db.refresh(breed)
        return breed

    @staticmethod
    def get_by_name(db: Session, breed_name: str) -> Optional[Breed]:
        stmt = select(Breed).where(Breed.breed_name == breed_name.strip())
        return db.scalar(stmt)

    @staticmethod
    def get_by_id(db: Session, breed_id: int) -> Optional[Breed]:
        return db.get(Breed, breed_id)

    @staticmethod
    def list_all(db: Session) -> List[Breed]:
        stmt = select(Breed).order_by(Breed.id)
        return list(db.scalars(stmt).all())


class ImageMetadataRepository:
    """CRUD repository for ImageMetadata entity."""

    @staticmethod
    def create_image(
        db: Session,
        filename: str,
        file_path: str,
        file_size_bytes: int,
        mime_type: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        md5_hash: Optional[str] = None,
    ) -> ImageMetadata:
        img = ImageMetadata(
            filename=filename,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            md5_hash=md5_hash,
        )
        db.add(img)
        db.commit()
        db.refresh(img)
        return img

    @staticmethod
    def get_by_id(db: Session, image_id: int) -> Optional[ImageMetadata]:
        return db.get(ImageMetadata, image_id)


class ModelVersionRepository:
    """CRUD repository for ModelVersion entity."""

    @staticmethod
    def create_version(
        db: Session,
        model_name: str,
        yolo_version: str,
        efficientnet_version: str,
        gradcam_version: str,
        is_active: bool = True,
    ) -> ModelVersion:
        version = ModelVersion(
            model_name=model_name,
            yolo_version=yolo_version,
            efficientnet_version=efficientnet_version,
            gradcam_version=gradcam_version,
            is_active=is_active,
        )
        db.add(version)
        db.commit()
        db.refresh(version)
        return version

    @staticmethod
    def get_active(db: Session) -> Optional[ModelVersion]:
        stmt = select(ModelVersion).where(ModelVersion.is_active == True).order_by(ModelVersion.id.desc())
        return db.scalar(stmt)


class PredictionRepository:
    """CRUD repository for Prediction entity."""

    @staticmethod
    def create_prediction(
        db: Session,
        image_id: int,
        model_version_id: int,
        animal_type: str,
        animal_confidence: float,
        bounding_box: List[int],
        predicted_breed_name: str,
        breed_confidence: float,
        top_3_predictions: List[Dict[str, Any]],
        prediction_status: str,
        inference_time_ms: Dict[str, float],
        user_id: Optional[int] = None,
        predicted_breed_id: Optional[int] = None,
    ) -> Prediction:
        pred = Prediction(
            image_id=image_id,
            user_id=user_id,
            model_version_id=model_version_id,
            animal_type=animal_type,
            animal_confidence=animal_confidence,
            bounding_box=bounding_box,
            predicted_breed_id=predicted_breed_id,
            predicted_breed_name=predicted_breed_name,
            breed_confidence=breed_confidence,
            top_3_predictions=top_3_predictions,
            prediction_status=prediction_status,
            inference_time_ms=inference_time_ms,
        )
        db.add(pred)
        db.commit()
        db.refresh(pred)
        return pred

    @staticmethod
    def get_by_id(db: Session, prediction_id: int) -> Optional[Prediction]:
        return db.get(Prediction, prediction_id)

    @staticmethod
    def list_recent(db: Session, limit: int = 50) -> List[Prediction]:
        stmt = select(Prediction).order_by(Prediction.id.desc()).limit(limit)
        return list(db.scalars(stmt).all())
