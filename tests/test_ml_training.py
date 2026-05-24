"""Tests for lightweight baseline ML training."""

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.models import FeatureValue, PriceBar, Symbol, Timeframe
from ml.training import build_training_frame, train_baseline_models


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_build_training_frame_labels_future_returns():
    db = _session()
    try:
        symbol = Symbol(ticker="TEST", is_active=True, is_bist100=True)
        db.add(symbol)
        db.flush()
        start = datetime(2026, 1, 1)
        for index in range(12):
            timestamp = start + timedelta(days=index)
            close = 100 + index
            db.add(
                PriceBar(
                    symbol_id=symbol.id,
                    timeframe=Timeframe.DAILY,
                    timestamp=timestamp,
                    open=close,
                    high=close + 1,
                    low=close - 1,
                    close=close,
                    volume=1000,
                    source="test",
                )
            )
            db.add(
                FeatureValue(
                    symbol_id=symbol.id,
                    timeframe=Timeframe.DAILY,
                    timestamp=timestamp,
                    feature_set="technical_v1",
                    rsi=50 + index,
                    macd=0.1,
                    macd_signal=0.05,
                    atr_pct=0.02,
                    volatility=0.1,
                    volume_ratio=1.1,
                    trend_score=60 + index,
                    volume_score=55,
                    momentum_score=58,
                )
            )
        db.commit()

        frame = build_training_frame(db, timeframe=Timeframe.DAILY, horizon_bars=3, target_return=0.02)

        assert len(frame) == 9
        assert frame["target"].sum() > 0
    finally:
        db.close()


def test_train_baseline_models_can_skip_when_data_is_small(tmp_path):
    db = _session()
    try:
        result = train_baseline_models(db, artifact_dir=tmp_path, min_rows=80)

        assert result.metrics["status"] == "skipped"
        assert result.model_run.model_type == "skipped"
    finally:
        db.close()
