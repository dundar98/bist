"""Feature API schemas."""

from datetime import datetime

from pydantic import BaseModel


class FeatureValueRead(BaseModel):
    timestamp: datetime
    timeframe: str
    feature_set: str
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    atr_pct: float | None = None
    volatility: float | None = None
    volume_ratio: float | None = None
    trend_score: float | None = None
    volume_score: float | None = None
    momentum_score: float | None = None

    model_config = {"from_attributes": True}
