"""Authenticated dashboard endpoints for the web application."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.routers.portfolios import _to_snapshot_read
from app.schemas.dashboard import DashboardCounts, DashboardOverview
from database.models import Horizon, PortfolioSnapshot, Signal, SignalStatus, Symbol, Timeframe, User
from database.repositories.portfolio import get_latest_portfolio_snapshot
from database.repositories.prices import timeframe_from_string
from database.session import get_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/overview", response_model=DashboardOverview)
def dashboard_overview(
    timeframe: str = Query(default=Timeframe.DAILY.value),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DashboardOverview:
    """Return the compact overview needed by the first web dashboard screen."""

    tf = timeframe_from_string(timeframe)
    symbol_count = db.scalar(select(func.count()).select_from(Symbol)) or 0
    signal_count = db.scalar(select(func.count()).select_from(Signal)) or 0
    open_signal_count = db.scalar(select(func.count()).select_from(Signal).where(Signal.status == SignalStatus.OPEN)) or 0
    portfolio_count = db.scalar(select(func.count()).select_from(PortfolioSnapshot)) or 0

    latest_portfolios = []
    for horizon in (Horizon.SHORT, Horizon.MEDIUM, Horizon.LONG):
        snapshot = get_latest_portfolio_snapshot(db, timeframe=tf, horizon=horizon)
        if snapshot is not None:
            latest_portfolios.append(_to_snapshot_read(snapshot))

    return DashboardOverview(
        generated_at=datetime.now(timezone.utc),
        counts=DashboardCounts(
            symbols=symbol_count,
            signals=signal_count,
            open_signals=open_signal_count,
            portfolios=portfolio_count,
        ),
        latest_portfolios=latest_portfolios,
    )

