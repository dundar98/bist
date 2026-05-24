"""Analysis API schemas for symbol detail and audit logs."""

from datetime import datetime

from pydantic import BaseModel


class SymbolPriceSummary(BaseModel):
    timestamp: datetime
    close: float
    return_1d: float | None = None
    return_5d: float | None = None
    return_20d: float | None = None


class SymbolFeatureSummary(BaseModel):
    timestamp: datetime
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    atr_pct: float | None = None
    volatility: float | None = None
    volume_ratio: float | None = None
    trend_score: float | None = None
    volume_score: float | None = None
    momentum_score: float | None = None


class SymbolSignalSummary(BaseModel):
    id: int
    signal_time: datetime
    horizon: str
    direction: str
    status: str
    final_score: float
    entry_price: float
    stop_price: float | None = None
    target_price: float | None = None
    reason: str | None = None


class DecisionLogRead(BaseModel):
    id: int
    decision_time: datetime
    signal_time: datetime
    timeframe: str
    horizon: str
    strategy: str
    direction: str
    entry_price: float
    stop_price: float | None = None
    target_price: float | None = None
    final_score: float
    reason: str | None = None


class SymbolAnalysisRead(BaseModel):
    symbol_id: int
    ticker: str
    name: str | None = None
    sector: str | None = None
    timeframe: str
    price: SymbolPriceSummary | None = None
    feature: SymbolFeatureSummary | None = None
    latest_signals: list[SymbolSignalSummary]
    decision_logs: list[DecisionLogRead]
    summary: str
