"""Tests for symbol-level analysis endpoint and decision logs."""

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
    DecisionLog,
    FeatureValue,
    Horizon,
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
        symbol = Symbol(ticker="THYAO", name="Turk Hava Yollari", sector="Ulastirma", is_active=True, is_bist100=True)
        db.add_all([user, symbol])
        db.flush()

        for index, close in enumerate([100, 102, 104, 108, 110, 115], start=1):
            db.add(
                PriceBar(
                    symbol_id=symbol.id,
                    timeframe=Timeframe.DAILY,
                    timestamp=datetime(2026, 5, index),
                    open=close - 1,
                    high=close + 1,
                    low=close - 2,
                    close=close,
                    volume=1000,
                    source="test",
                )
            )

        db.add(
            FeatureValue(
                symbol_id=symbol.id,
                timeframe=Timeframe.DAILY,
                timestamp=datetime(2026, 5, 6),
                feature_set="technical_v1",
                rsi=58,
                trend_score=82,
                volume_score=70,
                momentum_score=76,
            )
        )
        signal = Signal(
            symbol_id=symbol.id,
            signal_time=datetime(2026, 5, 6),
            timeframe=Timeframe.DAILY,
            horizon=Horizon.MEDIUM,
            strategy="technical_selective_v1",
            direction=SignalDirection.BUY,
            status=SignalStatus.OPEN,
            final_score=78,
            model_score=0,
            trend_score=82,
            volume_score=70,
            relative_strength_score=76,
            risk_score=65,
            entry_price=115,
            stop_price=109.25,
            target_price=126.5,
            reason="test buy",
        )
        db.add(signal)
        db.flush()
        db.add(
            DecisionLog(
                signal_id=signal.id,
                symbol_id=symbol.id,
                decision_time=datetime(2026, 5, 6, 18),
                signal_time=signal.signal_time,
                timeframe=Timeframe.DAILY,
                horizon=Horizon.MEDIUM,
                strategy=signal.strategy,
                direction=SignalDirection.BUY,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                final_score=signal.final_score,
                model_score=0,
                trend_score=82,
                volume_score=70,
                relative_strength_score=76,
                risk_score=65,
                reason="test buy",
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


def test_symbol_analysis_requires_login(client: TestClient):
    response = client.get("/api/symbols/THYAO/analysis")

    assert response.status_code == 401


def test_symbol_analysis_returns_price_feature_signal_and_logs(client: TestClient):
    login = client.post("/api/auth/login", json={"username_or_email": "viewer", "password": "viewer123"})
    token = login.json()["access_token"]

    response = client.get("/api/symbols/THYAO/analysis", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "THYAO"
    assert payload["price"]["close"] == 115
    assert payload["price"]["return_5d"] == 15
    assert payload["feature"]["trend_score"] == 82
    assert payload["latest_signals"][0]["direction"] == "BUY"
    assert payload["decision_logs"][0]["entry_price"] == 115
    assert "trend" in payload["summary"]
