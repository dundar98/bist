"""Tests for dashboard overview endpoints."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import hash_password
from app.main import app
from database.base import Base
from database.models import Symbol, User, UserRole
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
        db.add(
            User(
                username="viewer",
                email="viewer@example.com",
                password_hash=hash_password("viewer123"),
                role=UserRole.VIEWER,
                is_active=True,
            )
        )
        db.add(Symbol(ticker="THYAO", name="Turk Hava Yollari", is_active=True, is_bist100=True))
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


def test_dashboard_requires_login(client: TestClient):
    response = client.get("/api/dashboard/overview")

    assert response.status_code == 401


def test_dashboard_overview_returns_counts(client: TestClient):
    login = client.post("/api/auth/login", json={"username_or_email": "viewer", "password": "viewer123"})
    token = login.json()["access_token"]

    response = client.get("/api/dashboard/overview", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["counts"]["symbols"] == 1
    assert response.json()["counts"]["signals"] == 0
    assert response.json()["latest_portfolios"] == []

