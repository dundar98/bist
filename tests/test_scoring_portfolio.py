"""Tests for rule-based scoring and portfolio construction."""

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.models import FeatureValue, Horizon, PriceBar, SignalDirection, Symbol, Timeframe
from database.repositories.portfolio import _capped_weights, create_portfolio_snapshot
from signals.scoring import score_feature


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_score_feature_promotes_strong_profiles():
    feature = FeatureValue(
        symbol_id=1,
        timeframe=Timeframe.DAILY,
        timestamp=datetime(2024, 1, 1),
        trend_score=80,
        volume_score=75,
        momentum_score=78,
        volatility=0.15,
        atr_pct=0.02,
        rsi=62,
    )

    result = score_feature(feature)

    assert result.direction == "BUY"
    assert result.final_score >= 62
    assert "passed selective filters" in result.reason
    assert "horizon medium" in result.reason


def test_score_feature_blocks_low_risk_profiles_even_when_momentum_is_strong():
    feature = FeatureValue(
        symbol_id=1,
        timeframe=Timeframe.DAILY,
        timestamp=datetime(2024, 1, 1),
        trend_score=100,
        volume_score=75,
        momentum_score=100,
        volatility=0.65,
        atr_pct=0.08,
        rsi=62,
    )

    result = score_feature(feature, horizon=Horizon.MEDIUM)

    assert result.direction == "HOLD"
    assert "blocked: risk below" in result.reason


def test_create_portfolio_snapshot_selects_highest_scores():
    db = _session()
    try:
        weak = Symbol(ticker="WEAK", market="BIST", is_active=True, is_bist100=True)
        strong = Symbol(ticker="STRONG", market="BIST", is_active=True, is_bist100=True)
        db.add_all([weak, strong])
        db.commit()

        for symbol, trend, volume, momentum in [(weak, 45, 45, 40), (strong, 82, 74, 78)]:
            db.add(
                FeatureValue(
                    symbol_id=symbol.id,
                    timeframe=Timeframe.DAILY,
                    timestamp=datetime(2024, 1, 2),
                    feature_set="technical_v1",
                    trend_score=trend,
                    volume_score=volume,
                    momentum_score=momentum,
                    volatility=0.12,
                    atr_pct=0.02,
                    rsi=60,
                )
            )
            db.add(
                PriceBar(
                    symbol_id=symbol.id,
                    timeframe=Timeframe.DAILY,
                    timestamp=datetime(2024, 1, 2),
                    open=100,
                    high=105,
                    low=99,
                    close=102,
                    volume=1000,
                    source="test",
                )
            )
        db.commit()

        snapshot = create_portfolio_snapshot(db, timeframe=Timeframe.DAILY, max_positions=1, min_score=55)
        db.commit()

        assert len(snapshot.items) == 1
        assert snapshot.horizon == Horizon.MEDIUM
        assert snapshot.items[0].symbol_id == strong.id
        assert snapshot.items[0].signal.direction == SignalDirection.BUY
        assert snapshot.items[0].signal.horizon == Horizon.MEDIUM
        assert "market_risk=not_used" in snapshot.items[0].signal.reason
    finally:
        db.close()


def test_market_regime_makes_portfolio_more_conservative():
    db = _session()
    try:
        for index in range(8):
            symbol = Symbol(ticker=f"STK{index}", market="BIST", is_active=True, is_bist100=True)
            db.add(symbol)
            db.flush()
            db.add(
                FeatureValue(
                    symbol_id=symbol.id,
                    timeframe=Timeframe.DAILY,
                    timestamp=datetime(2024, 1, 2),
                    feature_set="technical_v1",
                    trend_score=90,
                    volume_score=80,
                    momentum_score=85,
                    volatility=0.12,
                    atr_pct=0.02,
                    rsi=60,
                )
            )
            db.add(
                PriceBar(
                    symbol_id=symbol.id,
                    timeframe=Timeframe.DAILY,
                    timestamp=datetime(2024, 1, 2),
                    open=100,
                    high=105,
                    low=99,
                    close=100,
                    volume=1000,
                    source="test",
                )
            )
        db.commit()

        snapshot = create_portfolio_snapshot(
            db,
            timeframe=Timeframe.DAILY,
            max_positions=10,
            min_score=55,
            market_risk_mode="riskli",
        )
        db.commit()

        assert len(snapshot.items) <= 5
        assert snapshot.items[0].signal.target_price == 107
        assert "market_risk=riskli" in snapshot.items[0].signal.reason
    finally:
        db.close()


def test_create_portfolio_snapshot_excludes_non_buy_candidates():
    db = _session()
    try:
        risky = Symbol(ticker="RISKY", market="BIST", is_active=True, is_bist100=True)
        db.add(risky)
        db.commit()

        db.add(
            FeatureValue(
                symbol_id=risky.id,
                timeframe=Timeframe.DAILY,
                timestamp=datetime(2024, 1, 2),
                feature_set="technical_v1",
                trend_score=100,
                volume_score=80,
                momentum_score=100,
                volatility=0.8,
                atr_pct=0.08,
                rsi=60,
            )
        )
        db.add(
            PriceBar(
                symbol_id=risky.id,
                timeframe=Timeframe.DAILY,
                timestamp=datetime(2024, 1, 2),
                open=100,
                high=105,
                low=99,
                close=102,
                volume=1000,
                source="test",
            )
        )
        db.commit()

        snapshot = create_portfolio_snapshot(db, timeframe=Timeframe.DAILY, max_positions=10, min_score=55)
        db.commit()

        assert snapshot.items == []
    finally:
        db.close()


def test_capped_weights_limits_single_position_weight():
    weights = _capped_weights([90, 80, 70, 60, 50, 40], max_weight=0.15)

    assert max(weights) <= 0.15
    assert sum(weights) <= 1.0


def test_long_horizon_is_stricter_about_risk_and_rsi():
    feature = FeatureValue(
        symbol_id=1,
        timeframe=Timeframe.DAILY,
        timestamp=datetime(2024, 1, 1),
        trend_score=90,
        volume_score=50,
        momentum_score=70,
        volatility=0.12,
        atr_pct=0.02,
        rsi=74,
    )

    medium = score_feature(feature, horizon=Horizon.MEDIUM)
    long = score_feature(feature, horizon=Horizon.LONG)

    assert medium.direction == "BUY"
    assert long.direction == "HOLD"
    assert "RSI above 72" in long.reason
