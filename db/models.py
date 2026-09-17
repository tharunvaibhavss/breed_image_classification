"""SQLAlchemy ORM models for database tables: users, breeds, images, predictions, model_versions."""

import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


def utc_now() -> datetime.datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)


class User(Base):
    """User account entity table."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="user")


class Breed(Base):
    """Breed metadata entity table for Indian Cattle and Buffaloes."""

    __tablename__ = "breeds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    breed_name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    animal_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False)  # 'cattle' or 'buffalo'
    origin: Mapped[str] = mapped_column(String(255), nullable=False)
    native_state: Mapped[str] = mapped_column(String(100), nullable=False)
    physical_characteristics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    milk_production: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    climate_adaptability: Mapped[str] = mapped_column(Text, nullable=False)
    uses: Mapped[str] = mapped_column(String(100), nullable=False)  # 'dairy', 'draft', 'dual-purpose'
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="breed_ref")


class ImageMetadata(Base):
    """Image upload metadata entity table."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    md5_hash: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=utc_now, nullable=False
    )

    # Relationships
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="image")


class ModelVersion(Base):
    """AI Model architecture versioning entity table."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    yolo_version: Mapped[str] = mapped_column(String(50), nullable=False)
    efficientnet_version: Mapped[str] = mapped_column(String(50), nullable=False)
    gradcam_version: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=utc_now, nullable=False
    )

    # Relationships
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="model_version")


class Prediction(Base):
    """AI Breed Recognition prediction record entity table."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("images.id"), nullable=False)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=False)

    animal_type: Mapped[str] = mapped_column(String(50), nullable=False)
    animal_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bounding_box: Mapped[List[int]] = mapped_column(JSON, nullable=False)  # [xmin, ymin, xmax, ymax]

    predicted_breed_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("breeds.id"), nullable=True)
    predicted_breed_name: Mapped[str] = mapped_column(String(100), nullable=False)
    breed_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top_3_predictions: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)

    prediction_status: Mapped[str] = mapped_column(String(50), nullable=False)
    inference_time_ms: Mapped[Dict[str, float]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=utc_now, nullable=False
    )

    # Relationships
    image: Mapped["ImageMetadata"] = relationship("ImageMetadata", back_populates="predictions")
    user: Mapped[Optional["User"]] = relationship("User", back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship("ModelVersion", back_populates="predictions")
    breed_ref: Mapped[Optional["Breed"]] = relationship("Breed", back_populates="predictions")
