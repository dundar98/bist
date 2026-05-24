"""Tests for price persistence helpers."""

from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.models import Symbol, Timeframe
from database.repositories.prices import (
    PriceDataError,
    get_last_price_timestamp,
    list_price_bars,
    upsert_price_bars,
    validate_price_frame,
)


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


@pytest.fixture
def symbol(db_session):
    item = Symbol(ticker="THYAO", market="BIST", is_active=True, is_bist100=True)
    db_session.add(item)
    db_session.commit()
    return item


def _sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [104.0, 102.0],
            "volume": [1000, 1200],
        }
    )


def test_upsert_price_bars_inserts_and_updates_without_duplicates(db_session, symbol):
    changed = upsert_price_bars(db_session, symbol, Timeframe.DAILY, _sample_prices(), source="test")
    db_session.commit()

    assert changed == 2
    assert len(list_price_bars(db_session, symbol.id, Timeframe.DAILY)) == 2
    assert get_last_price_timestamp(db_session, symbol.id, Timeframe.DAILY) == datetime(2024, 1, 2)

    updated = _sample_prices()
    updated.loc[1, "close"] = 103.5
    changed = upsert_price_bars(db_session, symbol, Timeframe.DAILY, updated, source="test")
    db_session.commit()

    bars = list_price_bars(db_session, symbol.id, Timeframe.DAILY)
    assert changed == 2
    assert len(bars) == 2
    assert bars[-1].close == 103.5


def test_validate_price_frame_rejects_invalid_ranges():
    prices = _sample_prices()
    prices.loc[0, "high"] = 98.0

    with pytest.raises(PriceDataError):
        validate_price_frame(prices)
