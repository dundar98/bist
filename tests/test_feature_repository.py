"""Tests for feature store helpers."""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.models import Symbol, Timeframe
from database.repositories.features import compute_feature_frame, list_feature_values, upsert_feature_values
from database.repositories.prices import upsert_price_bars


def _price_frame(rows: int = 90) -> pd.DataFrame:
    close = 100 + np.cumsum(np.sin(np.arange(rows) / 5) + 0.2)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000, 2000, rows),
        }
    )


def test_compute_and_store_features_from_price_bars():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        symbol = Symbol(ticker="THYAO", market="BIST", is_active=True, is_bist100=True)
        db.add(symbol)
        db.commit()

        upsert_price_bars(db, symbol, Timeframe.DAILY, _price_frame(), source="test")
        db.commit()

        bars = symbol.price_bars
        features = compute_feature_frame(bars)
        changed = upsert_feature_values(db, symbol, Timeframe.DAILY, features)
        db.commit()

        stored = list_feature_values(db, symbol.id, Timeframe.DAILY, limit=5)
        assert changed == len(features)
        assert len(stored) == 5
        assert stored[-1].trend_score is not None
        assert 0 <= stored[-1].trend_score <= 100
