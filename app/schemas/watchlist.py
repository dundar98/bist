"""Watchlist schemas."""

from datetime import datetime

from pydantic import BaseModel


class WatchlistCreate(BaseModel):
    ticker: str
    note: str | None = None


class WatchlistItemRead(BaseModel):
    id: int
    symbol_id: int
    ticker: str
    name: str | None = None
    sector: str | None = None
    note: str | None = None
    created_at: datetime
