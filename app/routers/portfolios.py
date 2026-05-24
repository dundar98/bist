"""Portfolio endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.schemas.signals import PortfolioItemRead, PortfolioSnapshotRead
from database.models import PortfolioSnapshot, User
from database.repositories.portfolio import get_latest_portfolio_snapshot
from database.repositories.prices import timeframe_from_string
from database.session import get_db
from signals.scoring import horizon_from_string

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


def _to_snapshot_read(snapshot: PortfolioSnapshot) -> PortfolioSnapshotRead:
    items = []
    for item in sorted(snapshot.items, key=lambda row: row.rank):
        items.append(
            PortfolioItemRead(
                rank=item.rank,
                symbol_id=item.symbol_id,
                ticker=item.symbol.ticker if item.symbol else str(item.symbol_id),
                signal_id=item.signal_id,
                score=item.score,
                suggested_weight=item.suggested_weight,
                reason=item.reason,
            )
        )

    return PortfolioSnapshotRead(
        id=snapshot.id,
        name=snapshot.name,
        snapshot_time=snapshot.snapshot_time,
        timeframe=snapshot.timeframe.value,
        horizon=snapshot.horizon.value,
        strategy=snapshot.strategy,
        items=items,
    )


@router.get("/latest", response_model=PortfolioSnapshotRead)
def latest_portfolio(
    timeframe: str | None = Query(default=None),
    horizon: str | None = Query(default=None),
    strategy: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PortfolioSnapshotRead:
    """Return the latest generated portfolio snapshot."""

    tf = timeframe_from_string(timeframe) if timeframe else None
    hz = horizon_from_string(horizon) if horizon else None
    snapshot = get_latest_portfolio_snapshot(db, timeframe=tf, horizon=hz, strategy=strategy)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No portfolio snapshot found")
    return _to_snapshot_read(snapshot)
