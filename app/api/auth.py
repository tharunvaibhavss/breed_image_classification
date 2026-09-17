"""Authentication API Endpoints: Register, Login, Logout, Profile."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from db.session import get_db
from db.models import User
from db.repository import UserRepository
from app.core.security import get_password_hash, verify_password, create_access_token
from app.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


class UserRegisterSchema(BaseModel):
    """Schema for user registration request."""

    email: EmailStr = Field(description="User valid email address")
    password: str = Field(min_length=6, description="User password (min 6 chars)")
    full_name: str = Field(description="User full display name")


class UserLoginSchema(BaseModel):
    """Schema for user login request."""

    email: EmailStr = Field(description="User email address")
    password: str = Field(description="User password")


class UserResponseSchema(BaseModel):
    """Schema for user entity response."""

    id: int
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool


class TokenResponseSchema(BaseModel):
    """Schema for JWT authentication token response."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponseSchema


@router.post(
    "/register",
    response_model=UserResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Register a New User Account",
)
def register_user(
    data: UserRegisterSchema, db: Session = Depends(get_db)
) -> Any:
    """Register a new user account with hashed password."""
    existing = UserRepository.get_by_email(db, data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user account with this email address already exists.",
        )

    hashed_pw = get_password_hash(data.password)
    user = UserRepository.create_user(
        db=db,
        email=data.email,
        hashed_password=hashed_pw,
        full_name=data.full_name,
        is_superuser=False,
    )
    return UserResponseSchema(
        id=user.id,
        email=user.email,
        full_name=user.full_name or "",
        is_active=user.is_active,
        is_superuser=user.is_superuser,
    )


@router.post(
    "/login",
    response_model=TokenResponseSchema,
    summary="Authenticate User and Get JWT Access Token",
)
def login_user(
    data: UserLoginSchema, db: Session = Depends(get_db)
) -> Any:
    """Authenticate user credentials and issue signed JWT access token."""
    user = UserRepository.get_by_email(db, data.email)
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(subject=user.email)
    return TokenResponseSchema(
        access_token=token,
        token_type="bearer",
        user=UserResponseSchema(
            id=user.id,
            email=user.email,
            full_name=user.full_name or "",
            is_active=user.is_active,
            is_superuser=user.is_superuser,
        ),
    )


@router.post("/logout", summary="Logout Current User")
def logout_user() -> Any:
    """Logout current user session."""
    return {"message": "Logged out successfully."}


@router.get(
    "/me",
    response_model=UserResponseSchema,
    summary="Get Current User Profile Details",
)
def get_user_profile(
    current_user: User = Depends(get_current_user),
) -> Any:
    """Return profile details for current authenticated user."""
    return UserResponseSchema(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name or "",
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
    )
