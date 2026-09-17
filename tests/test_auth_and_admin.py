"""Security, Authentication, and Role-Based Access Control (RBAC) tests."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.main import app
from db.base import Base
from db.session import get_db
from db.repository import UserRepository
from app.core.security import get_password_hash, create_access_token


# Setup SQLite temporary test database fixture for auth tests
@pytest.fixture(scope="function")
def db_session(tmp_path) -> Session:
    db_file = tmp_path / "test_auth_system.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Override FastAPI get_db dependency
    app.dependency_overrides[get_db] = _override_get_db

    session = TestingSessionLocal()
    yield session
    session.close()

    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(db_session: Session) -> TestClient:
    with TestClient(app) as c:
        yield c


def test_user_registration_and_login(client: TestClient):
    """Verify user registration and authentication flow issuing JWT token."""
    # 1. Register User
    reg_payload = {
        "email": "veterinarian@cattle.ai",
        "password": "SecurePassword123",
        "full_name": "Dr. Livestock Specialist",
    }
    reg_res = client.post("/api/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    user_data = reg_res.json()
    assert user_data["email"] == "veterinarian@cattle.ai"
    assert user_data["is_superuser"] is False

    # 2. Login User
    login_payload = {
        "email": "veterinarian@cattle.ai",
        "password": "SecurePassword123",
    }
    login_res = client.post("/api/auth/login", json=login_payload)
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"


def test_invalid_login_credentials(client: TestClient, db_session: Session):
    """Verify login rejection with 401 Unauthorized for incorrect password."""
    # Create user with hashed password
    UserRepository.create_user(
        db_session,
        email="testuser@cattle.ai",
        hashed_password=get_password_hash("CorrectPassword"),
        full_name="Test User",
    )

    # Attempt login with wrong password
    bad_res = client.post(
        "/api/auth/login",
        json={"email": "testuser@cattle.ai", "password": "WrongPassword"},
    )
    assert bad_res.status_code == 401
    assert "Incorrect email or password" in bad_res.json()["detail"]


def test_unauthorized_access_to_protected_routes(client: TestClient):
    """Verify 401 Unauthorized status when accessing protected route without JWT token."""
    res = client.get("/api/auth/me")
    assert res.status_code == 401


def test_user_profile_access(client: TestClient, db_session: Session):
    """Verify regular authenticated user access to profile endpoint."""
    user = UserRepository.create_user(
        db_session,
        email="regular@cattle.ai",
        hashed_password=get_password_hash("password123"),
        full_name="Regular User",
    )
    token = create_access_token(user.email)

    headers = {"Authorization": f"Bearer {token}"}
    res = client.get("/api/auth/me", headers=headers)
    assert res.status_code == 200
    assert res.json()["email"] == "regular@cattle.ai"
    assert res.json()["is_superuser"] is False


def test_admin_access_to_management_endpoints(client: TestClient, db_session: Session):
    """Verify administrator (is_superuser=True) access to admin endpoints."""
    admin = UserRepository.create_user(
        db_session,
        email="admin@cattle.ai",
        hashed_password=get_password_hash("AdminPass123"),
        full_name="System Admin",
        is_superuser=True,
    )
    token = create_access_token(admin.email)
    headers = {"Authorization": f"Bearer {token}"}

    # Test GET /api/admin/users
    users_res = client.get("/api/admin/users", headers=headers)
    assert users_res.status_code == 200
    assert len(users_res.json()) >= 1

    # Test GET /api/admin/analytics
    analytics_res = client.get("/api/admin/analytics", headers=headers)
    assert analytics_res.status_code == 200
    assert "total_predictions" in analytics_res.json()


def test_privilege_escalation_prevention(client: TestClient, db_session: Session):
    """Verify 403 Forbidden rejection when regular user attempts privilege escalation to admin endpoints."""
    regular_user = UserRepository.create_user(
        db_session,
        email="nonadmin@cattle.ai",
        hashed_password=get_password_hash("UserPass123"),
        full_name="Non Admin User",
        is_superuser=False,
    )
    user_token = create_access_token(regular_user.email)
    headers = {"Authorization": f"Bearer {user_token}"}

    # Attempt to access admin endpoints as regular user
    admin_users_res = client.get("/api/admin/users", headers=headers)
    assert admin_users_res.status_code == 403
    assert "Superuser administrative privileges required" in admin_users_res.json()["detail"]

    admin_analytics_res = client.get("/api/admin/analytics", headers=headers)
    assert admin_analytics_res.status_code == 403
