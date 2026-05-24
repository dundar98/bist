"""Signal endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.schemas.signals import RecommendationRead, SignalOutcomeRead, SignalRead
from database.models import Horizon, PortfolioItem, PriceBar, SignalDirection, Timeframe, User
from database.repositories.outcomes import list_signal_outcomes
from database.repositories.portfolio import get_latest_portfolio_snapshot, list_latest_signals
from database.repositories.prices import PriceDataError, timeframe_from_string
from database.session import get_db
from signals.scoring import horizon_from_string

router = APIRouter(prefix="/signals", tags=["signals"])

HORIZON_DAYS = {
    Horizon.SHORT: 5,
    Horizon.MEDIUM: 20,
    Horizon.LONG: 90,
}


def _latest_close(db: Session, symbol_id: int, timeframe: Timeframe) -> float | None:
    price = db.scalar(
        select(PriceBar)
        .where(PriceBar.symbol_id == symbol_id, PriceBar.timeframe == timeframe)
        .order_by(PriceBar.timestamp.desc())
        .limit(1)
    )
    return price.close if price else None


def _directional_return_pct(direction: SignalDirection, start: float, end: float | None) -> float | None:
    if end is None or start == 0:
        return None
    raw_return = ((end - start) / start) * 100
    if direction == SignalDirection.SELL:
        return -raw_return
    return raw_return


def _portfolio_item_to_recommendation(db: Session, item: PortfolioItem) -> RecommendationRead | None:
    signal = item.signal
    if signal is None:
        return None

    symbol = item.symbol or signal.symbol
    current_price = _latest_close(db, item.symbol_id, signal.timeframe)
    return_pct = _directional_return_pct(signal.direction, signal.entry_price, current_price)
    target_return_pct = _directional_return_pct(signal.direction, signal.entry_price, signal.target_price)
    stop_return_pct = _directional_return_pct(signal.direction, signal.entry_price, signal.stop_price)
    days_open = max((datetime.now(timezone.utc).date() - signal.signal_time.date()).days, 0)

    return RecommendationRead(
        signal_id=signal.id,
        symbol_id=item.symbol_id,
        ticker=symbol.ticker if symbol else str(item.symbol_id),
        name=symbol.name if symbol else None,
        sector=symbol.sector if symbol else None,
        signal_time=signal.signal_time,
        timeframe=signal.timeframe.value,
        horizon=signal.horizon.value,
        direction=signal.direction.value,
        status=signal.status.value,
        horizon_days=HORIZON_DAYS.get(signal.horizon, 20),
        days_open=days_open,
        final_score=signal.final_score,
        trend_score=signal.trend_score,
        volume_score=signal.volume_score,
        relative_strength_score=signal.relative_strength_score,
        risk_score=signal.risk_score,
        entry_price=signal.entry_price,
        current_price=current_price,
        return_pct=return_pct,
        stop_price=signal.stop_price,
        target_price=signal.target_price,
        target_return_pct=target_return_pct,
        stop_return_pct=stop_return_pct,
        reason=signal.reason,
    )


@router.get("/latest", response_model=list[SignalRead])
def latest_signals(
    timeframe: str | None = Query(default=None),
    horizon: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list:
    """Return latest generated signals."""

    tf = None
    if timeframe is not None:
        tf = timeframe_from_string(timeframe)
    hz = horizon_from_string(horizon) if horizon else None
    return list_latest_signals(db, timeframe=tf, horizon=hz, limit=limit)


@router.get("/recommendations", response_model=list[RecommendationRead])
def recommendations(
    timeframe: str | None = Query(default="1d"),
    horizon: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[RecommendationRead]:
    """Return current frontend-ready single-stock recommendations."""

    tf = timeframe_from_string(timeframe) if timeframe else None
    horizons = [horizon_from_string(horizon)] if horizon else [Horizon.SHORT, Horizon.MEDIUM, Horizon.LONG]
    recommendations: list[RecommendationRead] = []

    for hz in horizons:
        snapshot = get_latest_portfolio_snapshot(db, timeframe=tf, horizon=hz)
        if snapshot is None:
            continue

        for item in sorted(snapshot.items, key=lambda row: row.rank):
            recommendation = _portfolio_item_to_recommendation(db, item)
            if recommendation is not None:
                recommendations.append(recommendation)

    return recommendations[:limit]


@router.get("/outcomes", response_model=list[SignalOutcomeRead])
def latest_outcomes(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list:
    """Return latest calculated signal outcomes."""

    return list_signal_outcomes(db, limit=limit)
