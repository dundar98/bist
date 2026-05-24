"""User watchlist endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.core.auth import get_current_user
from app.schemas.watchlist import WatchlistCreate, WatchlistItemRead
from database.models import User, WatchlistItem
from database.repositories.prices import get_symbol_by_ticker
from database.session import get_db

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


def _to_read(item: WatchlistItem) -> WatchlistItemRead:
    return WatchlistItemRead(
        id=item.id,
        symbol_id=item.symbol_id,
        ticker=item.symbol.ticker,
        name=item.symbol.name,
        sector=item.symbol.sector,
        note=item.note,
        created_at=item.created_at,
    )


@router.get("", response_model=list[WatchlistItemRead])
def list_watchlist(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[WatchlistItemRead]:
    items = list(
        db.scalars(
            select(WatchlistItem)
            .options(selectinload(WatchlistItem.symbol))
            .where(WatchlistItem.user_id == current_user.id)
            .order_by(WatchlistItem.created_at.desc())
        ).all()
    )
    return [_to_read(item) for item in items]


@router.post("", response_model=WatchlistItemRead, status_code=201)
def add_watchlist_item(
    payload: WatchlistCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WatchlistItemRead:
    symbol = get_symbol_by_ticker(db, payload.ticker)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{payload.ticker}' not found")
    item = WatchlistItem(user_id=current_user.id, symbol_id=symbol.id, note=payload.note)
    db.add(item)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.scalar(
            select(WatchlistItem)
            .options(selectinload(WatchlistItem.symbol))
            .where(WatchlistItem.user_id == current_user.id, WatchlistItem.symbol_id == symbol.id)
        )
        if existing is None:
            raise
        return _to_read(existing)
    db.refresh(item)
    item = db.scalar(select(WatchlistItem).options(selectinload(WatchlistItem.symbol)).where(WatchlistItem.id == item.id))
    return _to_read(item)


@router.delete("/{item_id}", status_code=204)
def delete_watchlist_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    item = db.get(WatchlistItem, item_id)
    if item is None or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    db.delete(item)
    db.commit()
