"""Tests for admin-only user management endpoints."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import hash_password
from app.main import app
from database.base import Base
from database.models import User, UserRole
from database.session import get_db
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as db:
        db.add_all(
            [
                User(
                    username="admin",
                    email="admin@example.com",
                    password_hash=hash_password("admin123"),
                    role=UserRole.ADMIN,
                    is_active=True,
                ),
                User(
                    username="viewer",
                    email="viewer@example.com",
                    password_hash=hash_password("viewer123"),
                    role=UserRole.VIEWER,
                    is_active=True,
                ),
            ]
        )
        db.commit()

    def override_get_db() -> Iterator[Session]:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/auth/login", json={"username_or_email": username, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_admin_can_create_and_list_users(client: TestClient):
    token = _token(client, "admin", "admin123")

    created = client.post(
        "/api/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "username": "TraderOne",
            "email": "trader@example.com",
            "password": "strongpass",
            "full_name": "Trader One",
            "role": "analyst",
        },
    )
    listed = client.get("/api/users", headers={"Authorization": f"Bearer {token}"})

    assert created.status_code == 201
    assert created.json()["username"] == "traderone"
    assert created.json()["role"] == "analyst"
    assert {user["username"] for user in listed.json()} == {"admin", "viewer", "traderone"}


def test_viewer_cannot_manage_users(client: TestClient):
    token = _token(client, "viewer", "viewer123")

    response = client.get("/api/users", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 403


def test_duplicate_username_or_email_is_rejected(client: TestClient):
    token = _token(client, "admin", "admin123")

    response = client.post(
        "/api/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "username": "admin",
            "email": "new@example.com",
            "password": "strongpass",
            "role": "viewer",
        },
    )

    assert response.status_code == 409


def test_admin_can_update_password_and_active_state(client: TestClient):
    token = _token(client, "admin", "admin123")

    updated = client.patch(
        "/api/users/2",
        headers={"Authorization": f"Bearer {token}"},
        json={"password": "newpass123", "is_active": False},
    )
    failed_login = client.post("/api/auth/login", json={"username_or_email": "viewer", "password": "newpass123"})

    assert updated.status_code == 200
    assert updated.json()["is_active"] is False
    assert failed_login.status_code == 401

