"""Market and index radar schemas."""

from datetime import datetime

from pydantic import BaseModel


class IndexRadarRead(BaseModel):
    ticker: str
    label: str
    timestamp: datetime | None = None
    close: float | None = None
    return_1d: float | None = None
    return_5d: float | None = None
    return_20d: float | None = None
    trend_score: float | None = None
    risk_mode: str
    outlook: str
    summary: str


class MarketRadarRead(BaseModel):
    generated_at: datetime
    indices: list[IndexRadarRead]
    breadth_summary: str
    risk_mode: str
