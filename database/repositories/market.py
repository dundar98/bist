"""Market radar repository helpers."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas.market import IndexRadarRead, MarketRadarRead
from database.models import PriceBar, Symbol, Timeframe

INDEX_LABELS = {
    "XU100": "BIST 100",
    "XU030": "BIST 30",
    "XBANK": "Bankacilik",
    "XUSIN": "Sanayi",
}


def _pct(current: float, previous: float | None) -> float | None:
    if previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * 100


def _bars(db: Session, ticker: str, timeframe: Timeframe, limit: int = 30) -> list[PriceBar]:
    symbol = db.scalar(select(Symbol).where(Symbol.ticker == ticker))
    if symbol is None:
        return []
    rows = list(
        db.scalars(
            select(PriceBar)
            .where(PriceBar.symbol_id == symbol.id, PriceBar.timeframe == timeframe)
            .order_by(PriceBar.timestamp.desc())
            .limit(limit)
        ).all()
    )
    return list(reversed(rows))


def _outlook(return_5d: float | None, return_20d: float | None) -> tuple[str, str]:
    score = (return_5d or 0) * 0.6 + (return_20d or 0) * 0.4
    if score >= 3:
        return "pozitif", "risk alma ortami destekleniyor"
    if score <= -3:
        return "negatif", "piyasa baskisi belirgin"
    return "notr", "piyasa secici ilerliyor"


def _risk_mode(return_5d: float | None, return_20d: float | None) -> str:
    if (return_5d or 0) <= -4 or (return_20d or 0) <= -8:
        return "riskli"
    if (return_5d or 0) >= 3 and (return_20d or 0) >= 5:
        return "uygun"
    return "dikkat"


def build_market_radar(db: Session, timeframe: Timeframe = Timeframe.DAILY) -> MarketRadarRead:
    indices = []
    for ticker, label in INDEX_LABELS.items():
        bars = _bars(db, ticker, timeframe)
        if not bars:
            indices.append(
                IndexRadarRead(
                    ticker=ticker,
                    label=label,
                    risk_mode="veri_yok",
                    outlook="bilinmiyor",
                    summary=f"{label} icin veri yok. Endeks seed/veri akisi eklenmeli.",
                )
            )
            continue

        closes = [bar.close for bar in bars]
        latest = bars[-1]
        r1 = _pct(closes[-1], closes[-2] if len(closes) >= 2 else None)
        r5 = _pct(closes[-1], closes[-6] if len(closes) >= 6 else None)
        r20 = _pct(closes[-1], closes[-21] if len(closes) >= 21 else None)
        outlook, phrase = _outlook(r5, r20)
        risk = _risk_mode(r5, r20)
        trend_score = max(0.0, min(100.0, 50 + (r5 or 0) * 4 + (r20 or 0) * 2))
        indices.append(
            IndexRadarRead(
                ticker=ticker,
                label=label,
                timestamp=latest.timestamp,
                close=latest.close,
                return_1d=r1,
                return_5d=r5,
                return_20d=r20,
                trend_score=trend_score,
                risk_mode=risk,
                outlook=outlook,
                summary=f"{label}: {outlook} gorunum, {phrase}.",
            )
        )

    available = [item for item in indices if item.risk_mode != "veri_yok"]
    if not available:
        mode = "veri_yok"
        summary = "Endeks verisi henuz yok. XU100/XU030 verileri eklenince piyasa radari aktiflesir."
    else:
        risky = sum(1 for item in available if item.risk_mode == "riskli")
        suitable = sum(1 for item in available if item.risk_mode == "uygun")
        mode = "riskli" if risky >= 2 else "uygun" if suitable >= 2 else "dikkat"
        summary = f"Piyasa modu: {mode}. {len(available)} endeks uzerinden hesaplandi."

    return MarketRadarRead(generated_at=datetime.now(timezone.utc), indices=indices, breadth_summary=summary, risk_mode=mode)
