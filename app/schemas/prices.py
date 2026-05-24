"""Price API schemas."""

from datetime import datetime

from pydantic import BaseModel


class PriceBarRead(BaseModel):
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str

    model_config = {"from_attributes": True}
