"""Tests for frontend-ready recommendation endpoints."""

from collections.abc import Iterator
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import hash_password
from app.main import app
from database.base import Base
from database.models import (
    Horizon,
    PortfolioItem,
    PortfolioSnapshot,
    PriceBar,
    Signal,
    SignalDirection,
    SignalStatus,
    Symbol,
    Timeframe,
    User,
    UserRole,
)
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
        user = User(
            username="viewer",
            email="viewer@example.com",
            password_hash=hash_password("viewer123"),
            role=UserRole.VIEWER,
            is_active=True,
        )
        symbol = Symbol(
            ticker="THYAO",
            name="Turk Hava Yollari",
            sector="Ulastirma",
            market="BIST",
            is_active=True,
            is_bist100=True,
        )
        db.add_all([user, symbol])
        db.flush()
        db.add(
            PriceBar(
                symbol_id=symbol.id,
                timeframe=Timeframe.DAILY,
                timestamp=datetime(2026, 5, 19),
                open=308.0,
                high=312.0,
                low=304.0,
                close=309.0,
                volume=1000,
                source="test",
            )
        )
        signal = Signal(
            symbol_id=symbol.id,
            signal_time=datetime(2026, 5, 18),
            timeframe=Timeframe.DAILY,
            horizon=Horizon.MEDIUM,
            strategy="technical_selective_v1",
            direction=SignalDirection.BUY,
            status=SignalStatus.OPEN,
            final_score=78.5,
            model_score=0.0,
            trend_score=80.0,
            volume_score=70.0,
            relative_strength_score=75.0,
            risk_score=82.0,
            entry_price=300.0,
            stop_price=285.0,
            target_price=330.0,
            reason="test recommendation",
        )
        snapshot = PortfolioSnapshot(
            name="technical_selective_v1:medium:1d",
            snapshot_time=datetime(2026, 5, 19),
            timeframe=Timeframe.DAILY,
            horizon=Horizon.MEDIUM,
            strategy="technical_selective_v1",
        )
        db.add_all([signal, snapshot])
        db.flush()
        db.add(
            PortfolioItem(
                portfolio_snapshot_id=snapshot.id,
                symbol_id=symbol.id,
                signal_id=signal.id,
                rank=1,
                score=78.5,
                suggested_weight=0.15,
                reason="test recommendation",
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


def test_recommendations_require_login(client: TestClient):
    response = client.get("/api/signals/recommendations")

    assert response.status_code == 401


def test_recommendations_include_symbol_context(client: TestClient):
    login = client.post("/api/auth/login", json={"username_or_email": "viewer", "password": "viewer123"})
    token = login.json()["access_token"]

    response = client.get(
        "/api/signals/recommendations?timeframe=1d&horizon=medium",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()[0]["ticker"] == "THYAO"
    assert response.json()[0]["name"] == "Turk Hava Yollari"
    assert response.json()[0]["horizon"] == "medium"
    assert response.json()[0]["entry_price"] == 300.0
    assert response.json()[0]["current_price"] == 309.0
    assert response.json()[0]["return_pct"] == 3.0
    assert response.json()[0]["horizon_days"] == 20
