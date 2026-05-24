"""Tests for market radar endpoint."""

from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import hash_password
from app.main import app
from database.base import Base
from database.models import PriceBar, Symbol, Timeframe, User, UserRole
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
        xu100 = Symbol(ticker="XU100", name="BIST 100", market="BIST", is_active=True, is_bist100=False)
        db.add(xu100)
        db.flush()
        start = datetime(2026, 1, 1)
        for index in range(22):
            close = 1000 + index * 5
            db.add(
                PriceBar(
                    symbol_id=xu100.id,
                    timeframe=Timeframe.DAILY,
                    timestamp=start + timedelta(days=index),
                    open=close,
                    high=close + 5,
                    low=close - 5,
                    close=close,
                    volume=1000,
                    source="test",
                )
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


def test_market_radar_requires_login(client: TestClient):
    response = client.get("/api/market/radar")

    assert response.status_code == 401


def test_market_radar_returns_index_summary(client: TestClient):
    login = client.post("/api/auth/login", json={"username_or_email": "viewer", "password": "viewer123"})
    token = login.json()["access_token"]

    response = client.get("/api/market/radar", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["indices"][0]["ticker"] == "XU100"
    assert payload["indices"][0]["close"] == 1105
    assert "Piyasa modu" in payload["breadth_summary"]
