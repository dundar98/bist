"""Signal and portfolio API schemas."""

from datetime import datetime

from pydantic import BaseModel


class SignalRead(BaseModel):
    id: int
    symbol_id: int
    signal_time: datetime
    timeframe: str
    horizon: str
    strategy: str
    direction: str
    status: str
    final_score: float
    trend_score: float
    volume_score: float
    relative_strength_score: float
    risk_score: float
    entry_price: float
    stop_price: float | None = None
    target_price: float | None = None
    reason: str | None = None

    model_config = {"from_attributes": True}


class RecommendationRead(BaseModel):
    signal_id: int
    symbol_id: int
    ticker: str
    name: str | None = None
    sector: str | None = None
    signal_time: datetime
    timeframe: str
    horizon: str
    direction: str
    status: str
    horizon_days: int
    days_open: int
    final_score: float
    trend_score: float
    volume_score: float
    relative_strength_score: float
    risk_score: float
    entry_price: float
    current_price: float | None = None
    return_pct: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    target_return_pct: float | None = None
    stop_return_pct: float | None = None
    reason: str | None = None


class SignalOutcomeRead(BaseModel):
    id: int
    signal_id: int
    return_1d: float | None = None
    return_3d: float | None = None
    return_5d: float | None = None
    return_10d: float | None = None
    return_20d: float | None = None
    max_gain: float | None = None
    max_loss: float | None = None
    hit_target: bool
    hit_stop: bool

    model_config = {"from_attributes": True}


class PortfolioItemRead(BaseModel):
    rank: int
    symbol_id: int
    ticker: str
    signal_id: int | None = None
    score: float
    suggested_weight: float
    reason: str | None = None


class PortfolioSnapshotRead(BaseModel):
    id: int
    name: str
    snapshot_time: datetime
    timeframe: str
    horizon: str
    strategy: str
    items: list[PortfolioItemRead]
