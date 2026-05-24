"""Symbol endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.schemas.analysis import SymbolAnalysisRead
from app.schemas.features import FeatureValueRead
from app.schemas.prices import PriceBarRead
from app.schemas.symbols import SymbolRead
from database.models import Symbol, User
from database.repositories.analysis import build_symbol_analysis
from database.repositories.features import list_feature_values
from database.repositories.prices import (
    PriceDataError,
    get_symbol_by_ticker,
    list_price_bars,
    timeframe_from_string,
)
from database.session import get_db

router = APIRouter(prefix="/symbols", tags=["symbols"])


@router.get("", response_model=list[SymbolRead])
def list_symbols(
    active_only: bool = True,
    bist100_only: bool = True,
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[Symbol]:
    """List tradable symbols known by the platform."""

    stmt = select(Symbol).order_by(Symbol.ticker).limit(limit)

    if active_only:
        stmt = stmt.where(Symbol.is_active.is_(True))

    if bist100_only:
        stmt = stmt.where(Symbol.is_bist100.is_(True))

    return list(db.scalars(stmt).all())


@router.get("/{ticker}/prices", response_model=list[PriceBarRead])
def get_symbol_prices(
    ticker: str,
    timeframe: str = Query(default="1d"),
    limit: int = Query(default=200, ge=1, le=5000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list:
    """Return recent OHLCV bars for one symbol."""

    symbol = get_symbol_by_ticker(db, ticker)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{ticker}' not found")

    try:
        tf = timeframe_from_string(timeframe)
    except PriceDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bars = list_price_bars(db, symbol.id, tf, limit=limit)
    return bars


@router.get("/{ticker}/features", response_model=list[FeatureValueRead])
def get_symbol_features(
    ticker: str,
    timeframe: str = Query(default="1d"),
    feature_set: str = Query(default="technical_v1"),
    limit: int = Query(default=200, ge=1, le=5000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list:
    """Return recent computed features for one symbol."""

    symbol = get_symbol_by_ticker(db, ticker)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{ticker}' not found")

    try:
        tf = timeframe_from_string(timeframe)
    except PriceDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return list_feature_values(db, symbol.id, tf, feature_set=feature_set, limit=limit)


@router.get("/{ticker}/analysis", response_model=SymbolAnalysisRead)
def get_symbol_analysis(
    ticker: str,
    timeframe: str = Query(default="1d"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SymbolAnalysisRead:
    """Return a user-facing symbol analysis summary and decision audit trail."""

    symbol = get_symbol_by_ticker(db, ticker)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{ticker}' not found")

    try:
        tf = timeframe_from_string(timeframe)
    except PriceDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_symbol_analysis(db, symbol, tf)
