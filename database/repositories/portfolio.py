"""Portfolio construction and signal persistence."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from database.models import (
    FeatureValue,
    Horizon,
    DecisionLog,
    PortfolioItem,
    PortfolioSnapshot,
    PriceBar,
    Signal,
    SignalDirection,
    SignalStatus,
    Symbol,
    Timeframe,
)
from database.repositories.prices import list_active_symbols
from signals.scoring import ScoreResult, score_feature


@dataclass(frozen=True)
class Candidate:
    symbol: Symbol
    feature: FeatureValue
    latest_price: float
    score: ScoreResult


def _latest_feature(
    db: Session,
    symbol_id: int,
    timeframe: Timeframe,
    feature_set: str,
) -> FeatureValue | None:
    return db.scalar(
        select(FeatureValue)
        .where(
            FeatureValue.symbol_id == symbol_id,
            FeatureValue.timeframe == timeframe,
            FeatureValue.feature_set == feature_set,
        )
        .order_by(FeatureValue.timestamp.desc())
        .limit(1)
    )


def _latest_price(db: Session, symbol_id: int, timeframe: Timeframe) -> PriceBar | None:
    return db.scalar(
        select(PriceBar)
        .where(PriceBar.symbol_id == symbol_id, PriceBar.timeframe == timeframe)
        .order_by(PriceBar.timestamp.desc())
        .limit(1)
    )


def build_candidates(
    db: Session,
    *,
    timeframe: Timeframe,
    horizon: Horizon = Horizon.MEDIUM,
    feature_set: str = "technical_v1",
    symbol_limit: int | None = None,
    min_score: float = 55.0,
) -> list[Candidate]:
    """Score active symbols and return candidates above the watch threshold."""

    symbols = list_active_symbols(db, limit=symbol_limit)
    candidates: list[Candidate] = []

    for symbol in symbols:
        feature = _latest_feature(db, symbol.id, timeframe, feature_set)
        price = _latest_price(db, symbol.id, timeframe)
        if feature is None or price is None:
            continue

        score = score_feature(feature, horizon=horizon)
        if score.final_score < min_score:
            continue

        candidates.append(
            Candidate(
                symbol=symbol,
                feature=feature,
                latest_price=price.close,
                score=score,
            )
        )

    return sorted(candidates, key=lambda item: item.score.final_score, reverse=True)


def _capped_weights(scores: list[float], *, max_weight: float = 0.15) -> list[float]:
    """Allocate score-proportional weights with a per-position cap."""

    if not scores:
        return []

    weights = [0.0 for _ in scores]
    remaining_indices = set(range(len(scores)))
    remaining_weight = 1.0

    while remaining_indices and remaining_weight > 0:
        remaining_score = sum(scores[index] for index in remaining_indices) or float(len(remaining_indices))
        changed = False

        for index in list(remaining_indices):
            raw_weight = remaining_weight * (scores[index] / remaining_score)
            if raw_weight >= max_weight:
                weights[index] = max_weight
                remaining_weight -= max_weight
                remaining_indices.remove(index)
                changed = True

        if not changed:
            for index in remaining_indices:
                weights[index] = remaining_weight * (scores[index] / remaining_score)
            break

    return weights


def create_portfolio_snapshot(
    db: Session,
    *,
    timeframe: Timeframe,
    horizon: Horizon = Horizon.MEDIUM,
    strategy: str = "technical_selective_v1",
    feature_set: str = "technical_v1",
    symbol_limit: int | None = None,
    max_positions: int = 10,
    min_score: float = 55.0,
    market_risk_mode: str | None = None,
) -> PortfolioSnapshot:
    """Create a portfolio snapshot from the highest scoring candidates."""

    target_multiplier = 1.10
    stop_multiplier = 0.95
    if market_risk_mode == "riskli":
        min_score += 7
        max_positions = max(3, min(max_positions, 5))
        target_multiplier = 1.07
        stop_multiplier = 0.965
    elif market_risk_mode == "dikkat":
        min_score += 3
        max_positions = max(4, min(max_positions, 7))
        target_multiplier = 1.08
        stop_multiplier = 0.96

    candidates = build_candidates(
        db,
        timeframe=timeframe,
        horizon=horizon,
        feature_set=feature_set,
        symbol_limit=symbol_limit,
        min_score=min_score,
    )
    selected = [candidate for candidate in candidates if candidate.score.direction == "BUY"][:max_positions]

    snapshot = PortfolioSnapshot(
        name=f"{strategy}:{horizon.value}:{timeframe.value}",
        snapshot_time=datetime.now(timezone.utc),
        timeframe=timeframe,
        horizon=horizon,
        strategy=strategy,
    )
    db.add(snapshot)
    db.flush()

    weights = _capped_weights([candidate.score.final_score for candidate in selected])
    for rank, candidate in enumerate(selected, start=1):
        score = candidate.score
        direction = SignalDirection(score.direction)
        signal = Signal(
            symbol_id=candidate.symbol.id,
            signal_time=candidate.feature.timestamp,
            timeframe=timeframe,
            horizon=horizon,
            strategy=strategy,
            direction=direction,
            status=SignalStatus.OPEN,
            final_score=score.final_score,
            model_score=score.model_score,
            trend_score=score.trend_score,
            volume_score=score.volume_score,
            relative_strength_score=score.momentum_score,
            risk_score=score.risk_score,
            entry_price=candidate.latest_price,
            stop_price=candidate.latest_price * stop_multiplier if direction == SignalDirection.BUY else None,
            target_price=candidate.latest_price * target_multiplier if direction == SignalDirection.BUY else None,
            reason=f"{score.reason}; market_risk={market_risk_mode or 'not_used'}",
        )
        db.add(signal)
        db.flush()
        db.add(
            DecisionLog(
                signal_id=signal.id,
                symbol_id=candidate.symbol.id,
                decision_time=datetime.now(timezone.utc),
                signal_time=signal.signal_time,
                timeframe=timeframe,
                horizon=horizon,
                strategy=strategy,
                direction=direction,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                final_score=signal.final_score,
                model_score=signal.model_score,
                trend_score=signal.trend_score,
                volume_score=signal.volume_score,
                relative_strength_score=signal.relative_strength_score,
                risk_score=signal.risk_score,
                reason=signal.reason,
                raw_json=json.dumps(
                    {
                        "ticker": candidate.symbol.ticker,
                        "feature_timestamp": candidate.feature.timestamp.isoformat(),
                        "feature_set": candidate.feature.feature_set,
                        "rsi": candidate.feature.rsi,
                        "atr_pct": candidate.feature.atr_pct,
                        "volatility": candidate.feature.volatility,
                        "volume_ratio": candidate.feature.volume_ratio,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            )
        )

        db.add(
            PortfolioItem(
                portfolio_snapshot_id=snapshot.id,
                symbol_id=candidate.symbol.id,
                signal_id=signal.id,
                rank=rank,
                score=score.final_score,
                suggested_weight=weights[rank - 1] if weights else 0.0,
                reason=score.reason,
            )
        )

    return snapshot


def get_latest_portfolio_snapshot(
    db: Session,
    *,
    timeframe: Timeframe | None = None,
    horizon: Horizon | None = None,
    strategy: str | None = None,
) -> PortfolioSnapshot | None:
    """Return the latest saved portfolio snapshot with items and symbols."""

    stmt = (
        select(PortfolioSnapshot)
        .options(
            selectinload(PortfolioSnapshot.items).selectinload(PortfolioItem.symbol),
            selectinload(PortfolioSnapshot.items).selectinload(PortfolioItem.signal),
        )
        .order_by(PortfolioSnapshot.snapshot_time.desc())
        .limit(1)
    )
    if timeframe is not None:
        stmt = stmt.where(PortfolioSnapshot.timeframe == timeframe)
    if horizon is not None:
        stmt = stmt.where(PortfolioSnapshot.horizon == horizon)
    if strategy is not None:
        stmt = stmt.where(PortfolioSnapshot.strategy == strategy)
    return db.scalar(stmt)


def list_latest_signals(
    db: Session,
    *,
    timeframe: Timeframe | None = None,
    horizon: Horizon | None = None,
    limit: int = 50,
) -> list[Signal]:
    """Return latest saved signals."""

    stmt = select(Signal).options(selectinload(Signal.symbol)).order_by(Signal.created_at.desc()).limit(limit)
    if timeframe is not None:
        stmt = stmt.where(Signal.timeframe == timeframe)
    if horizon is not None:
        stmt = stmt.where(Signal.horizon == horizon)
    return list(db.scalars(stmt).all())
