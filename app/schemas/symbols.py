"""Symbol API schemas."""

from pydantic import BaseModel


class SymbolRead(BaseModel):
    id: int
    ticker: str
    name: str | None = None
    sector: str | None = None
    market: str
    is_active: bool
    is_bist100: bool

    model_config = {"from_attributes": True}
