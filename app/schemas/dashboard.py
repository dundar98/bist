"""Dashboard API schemas."""

from datetime import datetime

from pydantic import BaseModel

from app.schemas.signals import PortfolioSnapshotRead


class DashboardCounts(BaseModel):
    symbols: int
    signals: int
    open_signals: int
    portfolios: int


class DashboardOverview(BaseModel):
    generated_at: datetime
    counts: DashboardCounts
    latest_portfolios: list[PortfolioSnapshotRead]

