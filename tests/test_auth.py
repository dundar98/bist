"""Tests for manual-user authentication flow."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.auth import authenticate_user
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from app.main import app
from database.base import Base
from database.models import User, UserRole
from database.session import get_db
from fastapi.testclient import TestClient


def test_password_hash_roundtrip():
    password_hash = hash_password("s3cret")

    assert verify_password("s3cret", password_hash)
    assert not verify_password("wrong", password_hash)


def test_access_token_roundtrip():
    token = create_access_token("42")

    assert decode_access_token(token) == "42"


def test_authenticate_user_by_username_or_email():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        user = User(
            username="demo",
            email="demo@example.com",
            password_hash=hash_password("pass123"),
            role=UserRole.VIEWER,
            is_active=True,
        )
        db.add(user)
        db.commit()

        assert authenticate_user(db, "demo", "pass123") is not None
        assert authenticate_user(db, "demo@example.com", "pass123") is not None
        assert authenticate_user(db, "demo", "bad") is None


def test_protected_endpoints_require_login():
    client = TestClient(app)

    response = client.get("/api/signals/latest")

    assert response.status_code == 401


def test_login_token_allows_protected_endpoint():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as setup_db:
        setup_db.add(
            User(
                username="demo",
                email="demo@example.com",
                password_hash=hash_password("pass123"),
                role=UserRole.VIEWER,
                is_active=True,
            )
        )
        setup_db.commit()

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        login = client.post(
            "/api/auth/login",
            json={"username_or_email": "demo", "password": "pass123"},
        )
        token = login.json()["access_token"]
        response = client.get("/api/signals/latest", headers={"Authorization": f"Bearer {token}"})

        assert login.status_code == 200
        assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()
