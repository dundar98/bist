"""Signal outcome calculation and persistence."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import PriceBar, Signal, SignalDirection, SignalOutcome

RETURN_HORIZONS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class OutcomeResult:
    return_1d: float | None = None
    return_3d: float | None = None
    return_5d: float | None = None
    return_10d: float | None = None
    return_20d: float | None = None
    max_gain: float | None = None
    max_loss: float | None = None
    hit_target: bool = False
    hit_stop: bool = False


def _future_bars(db: Session, signal: Signal, *, max_bars: int = 20) -> list[PriceBar]:
    return list(
        db.scalars(
            select(PriceBar)
            .where(
                PriceBar.symbol_id == signal.symbol_id,
                PriceBar.timeframe == signal.timeframe,
                PriceBar.timestamp > signal.signal_time,
            )
            .order_by(PriceBar.timestamp.asc())
            .limit(max_bars)
        ).all()
    )


def calculate_signal_outcome(db: Session, signal: Signal) -> OutcomeResult | None:
    """Calculate outcome metrics for one signal from future price bars."""

    bars = _future_bars(db, signal, max_bars=max(RETURN_HORIZONS))
    if not bars:
        return None

    entry = signal.entry_price
    if entry <= 0:
        return None

    def bar_return(bar: PriceBar) -> float:
        raw = (bar.close - entry) / entry
        if signal.direction == SignalDirection.SELL:
            return -raw
        return raw

    returns_by_horizon = {}
    for horizon in RETURN_HORIZONS:
        if len(bars) >= horizon:
            returns_by_horizon[horizon] = bar_return(bars[horizon - 1])

    if signal.direction == SignalDirection.SELL:
        gains = [(entry - bar.low) / entry for bar in bars]
        losses = [(entry - bar.high) / entry for bar in bars]
        hit_target = signal.target_price is not None and any(bar.low <= signal.target_price for bar in bars)
        hit_stop = signal.stop_price is not None and any(bar.high >= signal.stop_price for bar in bars)
    else:
        gains = [(bar.high - entry) / entry for bar in bars]
        losses = [(bar.low - entry) / entry for bar in bars]
        hit_target = signal.target_price is not None and any(bar.high >= signal.target_price for bar in bars)
        hit_stop = signal.stop_price is not None and any(bar.low <= signal.stop_price for bar in bars)

    return OutcomeResult(
        return_1d=returns_by_horizon.get(1),
        return_3d=returns_by_horizon.get(3),
        return_5d=returns_by_horizon.get(5),
        return_10d=returns_by_horizon.get(10),
        return_20d=returns_by_horizon.get(20),
        max_gain=max(gains) if gains else None,
        max_loss=min(losses) if losses else None,
        hit_target=hit_target,
        hit_stop=hit_stop,
    )


def upsert_signal_outcome(db: Session, signal: Signal, result: OutcomeResult) -> SignalOutcome:
    """Insert or update one signal outcome row."""

    outcome = db.scalar(select(SignalOutcome).where(SignalOutcome.signal_id == signal.id))
    values = {
        "return_1d": result.return_1d,
        "return_3d": result.return_3d,
        "return_5d": result.return_5d,
        "return_10d": result.return_10d,
        "return_20d": result.return_20d,
        "max_gain": result.max_gain,
        "max_loss": result.max_loss,
        "hit_target": result.hit_target,
        "hit_stop": result.hit_stop,
    }

    if outcome is None:
        outcome = SignalOutcome(signal_id=signal.id, **values)
        db.add(outcome)
    else:
        for key, value in values.items():
            setattr(outcome, key, value)

    return outcome


def update_signal_outcomes(
    db: Session,
    *,
    limit: int = 500,
    only_buy: bool = True,
) -> int:
    """Update outcomes for recent signals that have future bars available."""

    stmt = select(Signal).order_by(Signal.signal_time.desc()).limit(limit)
    if only_buy:
        stmt = stmt.where(Signal.direction == SignalDirection.BUY)

    changed = 0
    for signal in db.scalars(stmt).all():
        result = calculate_signal_outcome(db, signal)
        if result is None:
            continue
        upsert_signal_outcome(db, signal, result)
        changed += 1

    return changed


def list_signal_outcomes(db: Session, *, limit: int = 100) -> list[SignalOutcome]:
    """Return latest outcome rows."""

    return list(
        db.scalars(
            select(SignalOutcome)
            .order_by(SignalOutcome.updated_at.desc())
            .limit(limit)
        ).all()
    )
