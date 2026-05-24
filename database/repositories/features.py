"""Repository helpers for feature value persistence."""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from data.features import FeatureEngine
from database.models import FeatureValue, PriceBar, Symbol, Timeframe
from database.repositories.prices import list_price_bars


CORE_FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "macd_signal",
    "atr_pct",
    "volatility",
    "volume_ratio",
    "trend_score",
    "volume_score",
    "momentum_score",
]


class FeatureStoreError(ValueError):
    """Raised when features cannot be computed or persisted."""


def price_bars_to_frame(bars: list[PriceBar]) -> pd.DataFrame:
    """Convert ORM price bars into the feature-engine DataFrame format."""

    return pd.DataFrame(
        [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )


def _bounded_score(value: float) -> float:
    if not math.isfinite(value):
        return 50.0
    return float(max(0.0, min(100.0, value)))


def add_selection_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable 0-100 selection scores for early filtering."""

    scored = df.copy()

    rsi = scored.get("rsi", pd.Series(50.0, index=scored.index)).fillna(50.0)
    trend = scored.get("price_to_sma_long", pd.Series(0.0, index=scored.index)).fillna(0.0)
    macd_hist = scored.get("macd_histogram", pd.Series(0.0, index=scored.index)).fillna(0.0)
    volume_ratio = scored.get("volume_ratio", pd.Series(1.0, index=scored.index)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return_20d = scored.get("return_20d", pd.Series(0.0, index=scored.index)).fillna(0.0)

    scored["trend_score"] = (50 + trend * 500 + macd_hist.rank(pct=True).fillna(0.5) * 20 - 10).map(_bounded_score)
    scored["volume_score"] = (50 + (volume_ratio - 1.0) * 25).map(_bounded_score)
    scored["momentum_score"] = (50 + return_20d * 250 + (rsi - 50) * 0.4).map(_bounded_score)
    return scored


def compute_feature_frame(bars: list[PriceBar]) -> pd.DataFrame:
    """Compute technical features and selection scores from price bars."""

    if len(bars) < 60:
        raise FeatureStoreError(f"At least 60 bars are required for stable features, got {len(bars)}")

    prices = price_bars_to_frame(bars)
    engine = FeatureEngine()
    features = engine.compute_all_features(prices)
    features = add_selection_scores(features)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def _to_nullable_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _row_raw_json(row: pd.Series) -> str:
    raw = {}
    for key, value in row.items():
        if key in {"timestamp", "open", "high", "low", "close", "volume"}:
            continue
        if pd.isna(value):
            continue
        if isinstance(value, (np.integer, np.floating)):
            raw[key] = float(value)
        else:
            raw[key] = value
    return json.dumps(raw, ensure_ascii=False, sort_keys=True)


def upsert_feature_values(
    db: Session,
    symbol: Symbol,
    timeframe: Timeframe,
    feature_frame: pd.DataFrame,
    *,
    feature_set: str = "technical_v1",
) -> int:
    """Insert or update computed features for a symbol/timeframe."""

    if feature_frame.empty:
        return 0

    clean = feature_frame.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"]).dt.tz_localize(None)
    clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp")

    timestamps = [row.to_pydatetime() for row in clean["timestamp"]]
    existing = {
        item.timestamp: item
        for item in db.scalars(
            select(FeatureValue).where(
                FeatureValue.symbol_id == symbol.id,
                FeatureValue.timeframe == timeframe,
                FeatureValue.feature_set == feature_set,
                FeatureValue.timestamp.in_(timestamps),
            )
        ).all()
    }

    changed = 0
    for _, row in clean.iterrows():
        timestamp = row["timestamp"].to_pydatetime()
        item = existing.get(timestamp)
        values = {
            "rsi": _to_nullable_float(row.get("rsi")),
            "macd": _to_nullable_float(row.get("macd")),
            "macd_signal": _to_nullable_float(row.get("macd_signal")),
            "atr_pct": _to_nullable_float(row.get("atr_pct")),
            "volatility": _to_nullable_float(row.get("volatility")),
            "volume_ratio": _to_nullable_float(row.get("volume_ratio")),
            "trend_score": _to_nullable_float(row.get("trend_score")),
            "volume_score": _to_nullable_float(row.get("volume_score")),
            "momentum_score": _to_nullable_float(row.get("momentum_score")),
            "raw_json": _row_raw_json(row),
        }

        if item is None:
            db.add(
                FeatureValue(
                    symbol_id=symbol.id,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    feature_set=feature_set,
                    **values,
                )
            )
        else:
            for key, value in values.items():
                setattr(item, key, value)
        changed += 1

    return changed


def compute_and_store_features(
    db: Session,
    symbol: Symbol,
    timeframe: Timeframe,
    *,
    lookback_bars: int = 260,
    feature_set: str = "technical_v1",
) -> int:
    """Compute features from stored price bars and persist them."""

    bars = list_price_bars(db, symbol.id, timeframe, limit=lookback_bars)
    feature_frame = compute_feature_frame(bars)
    return upsert_feature_values(db, symbol, timeframe, feature_frame, feature_set=feature_set)


def list_feature_values(
    db: Session,
    symbol_id: int,
    timeframe: Timeframe,
    *,
    feature_set: str = "technical_v1",
    limit: int = 200,
) -> list[FeatureValue]:
    """Return recent feature rows in chronological order."""

    subquery = (
        select(FeatureValue.id)
        .where(
            FeatureValue.symbol_id == symbol_id,
            FeatureValue.timeframe == timeframe,
            FeatureValue.feature_set == feature_set,
        )
        .order_by(FeatureValue.timestamp.desc())
        .limit(limit)
        .subquery()
    )
    stmt = (
        select(FeatureValue)
        .where(FeatureValue.id.in_(select(subquery.c.id)))
        .order_by(FeatureValue.timestamp.asc())
    )
    return list(db.scalars(stmt).all())
