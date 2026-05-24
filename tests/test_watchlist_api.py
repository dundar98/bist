"""Tests for user watchlist endpoints."""

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


def _token(client: TestClient) -> str:
    login = client.post("/api/auth/login", json={"username_or_email": "viewer", "password": "viewer123"})
    return login.json()["access_token"]


def test_watchlist_requires_login(client: TestClient):
    assert client.get("/api/watchlist").status_code == 401


def test_user_can_add_list_and_delete_watchlist_item(client: TestClient):
    token = _token(client)
    headers = {"Authorization": f"Bearer {token}"}

    created = client.post("/api/watchlist", headers=headers, json={"ticker": "THYAO", "note": "izle"})
    listed = client.get("/api/watchlist", headers=headers)
    deleted = client.delete(f"/api/watchlist/{created.json()['id']}", headers=headers)
    listed_after = client.get("/api/watchlist", headers=headers)

    assert created.status_code == 201
    assert listed.json()[0]["ticker"] == "THYAO"
    assert deleted.status_code == 204
    assert listed_after.json() == []
