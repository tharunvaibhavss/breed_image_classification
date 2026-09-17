"""Database session management and engine creation."""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from db.base import Base

# Determine Database Connection URL from Pydantic Settings
database_url = str(settings.DATABASE_URL)

# Fallback to SQLite in-memory/local file if PostgreSQL URL is invalid or default placeholder
if "postgresql" in database_url and ("localhost" in database_url or "127.0.0.1" in database_url):
    try:
        # Test connection to PostgreSQL server
        test_engine = create_engine(database_url, connect_args={"connect_timeout": 2})
        with test_engine.connect() as conn:
            pass
        engine = test_engine
    except Exception:
        # Fallback to SQLite local db file if local PostgreSQL server is offline
        sqlite_file = "breed_recognition_dev.db"
        database_url = f"sqlite:///{sqlite_file}"
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
else:
    if "sqlite" in database_url:
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(database_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables using Base metadata."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency provider yielding database session context."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
