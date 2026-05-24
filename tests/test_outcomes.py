"""Tests for signal outcome calculation."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.models import Horizon, PriceBar, Signal, SignalDirection, SignalStatus, Symbol, Timeframe
from database.repositories.outcomes import calculate_signal_outcome, update_signal_outcomes


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def _seed_signal_with_future_bars(db_session):
    symbol = Symbol(ticker="TEST", market="BIST", is_active=True, is_bist100=True)
    db_session.add(symbol)
    db_session.commit()

    signal_time = datetime(2024, 1, 1)
    signal = Signal(
        symbol_id=symbol.id,
        signal_time=signal_time,
        timeframe=Timeframe.DAILY,
        horizon=Horizon.MEDIUM,
        strategy="test",
        direction=SignalDirection.BUY,
        status=SignalStatus.OPEN,
        final_score=70,
        model_score=70,
        trend_score=80,
        volume_score=60,
        relative_strength_score=75,
        risk_score=55,
        entry_price=100,
        stop_price=95,
        target_price=110,
    )
    db_session.add(signal)

    closes = [102, 105, 111, 108, 112]
    for index, close in enumerate(closes, start=1):
        db_session.add(
            PriceBar(
                symbol_id=symbol.id,
                timeframe=Timeframe.DAILY,
                timestamp=signal_time + timedelta(days=index),
                open=close - 1,
                high=close + 1,
                low=close - 2,
                close=close,
                volume=1000,
                source="test",
            )
        )
    db_session.commit()
    return signal


def test_calculate_signal_outcome_for_buy_signal(db_session):
    signal = _seed_signal_with_future_bars(db_session)

    result = calculate_signal_outcome(db_session, signal)

    assert result is not None
    assert result.return_1d == pytest.approx(0.02)
    assert result.return_3d == pytest.approx(0.11)
    assert result.max_gain == pytest.approx(0.13)
    assert result.hit_target is True
    assert result.hit_stop is False


def test_update_signal_outcomes_persists_rows(db_session):
    _seed_signal_with_future_bars(db_session)

    changed = update_signal_outcomes(db_session)
    db_session.commit()

    assert changed == 1
