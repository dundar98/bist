"""Market radar endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.schemas.market import MarketRadarRead
from database.models import User
from database.repositories.market import build_market_radar
from database.repositories.prices import timeframe_from_string
from database.session import get_db

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/radar", response_model=MarketRadarRead)
def market_radar(
    timeframe: str = Query(default="1d"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MarketRadarRead:
    """Return Turkish market/index risk radar."""

    return build_market_radar(db, timeframe=timeframe_from_string(timeframe))
