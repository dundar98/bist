"""Core database models for market data, signals, portfolios, and users."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


class Timeframe(str, Enum):
    DAILY = "1d"
    HOURLY = "1h"
    FIFTEEN_MIN = "15m"


class Horizon(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStatus(str, Enum):
    OPEN = "OPEN"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    TRIAL = "trial"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class JobRun(Base, TimestampMixin):
    __tablename__ = "job_runs"
    __table_args__ = (Index("ix_job_runs_name_started", "job_name", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    summary_json: Mapped[Optional[str]] = mapped_column(Text)
    error: Mapped[Optional[str]] = mapped_column(Text)


class Symbol(Base, TimestampMixin):
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    sector: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    market: Mapped[str] = mapped_column(String(32), default="BIST", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_bist100: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    price_bars: Mapped[list["PriceBar"]] = relationship(back_populates="symbol")
    feature_values: Mapped[list["FeatureValue"]] = relationship(back_populates="symbol")
    signals: Mapped[list["Signal"]] = relationship(back_populates="symbol")


class WatchlistItem(Base, TimestampMixin):
    __tablename__ = "watchlist_items"
    __table_args__ = (
        UniqueConstraint("user_id", "symbol_id", name="uq_watchlist_user_symbol"),
        Index("ix_watchlist_user", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(String(255))

    user: Mapped[User] = relationship()
    symbol: Mapped[Symbol] = relationship()


class PriceBar(Base):
    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint("symbol_id", "timeframe", "timestamp", name="uq_price_bar_symbol_timeframe_ts"),
        Index("ix_price_bars_symbol_timeframe_ts", "symbol_id", "timeframe", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    timeframe: Mapped[Timeframe] = mapped_column(SAEnum(Timeframe), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="yfinance", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    symbol: Mapped[Symbol] = relationship(back_populates="price_bars")


class FeatureValue(Base):
    __tablename__ = "feature_values"
    __table_args__ = (
        UniqueConstraint("symbol_id", "timeframe", "timestamp", "feature_set", name="uq_feature_symbol_timeframe_ts_set"),
        Index("ix_feature_values_symbol_timeframe_ts", "symbol_id", "timeframe", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    timeframe: Mapped[Timeframe] = mapped_column(SAEnum(Timeframe), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    feature_set: Mapped[str] = mapped_column(String(64), default="technical_v1", nullable=False)
    rsi: Mapped[Optional[float]] = mapped_column(Float)
    macd: Mapped[Optional[float]] = mapped_column(Float)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float)
    atr_pct: Mapped[Optional[float]] = mapped_column(Float)
    volatility: Mapped[Optional[float]] = mapped_column(Float)
    volume_ratio: Mapped[Optional[float]] = mapped_column(Float)
    trend_score: Mapped[Optional[float]] = mapped_column(Float)
    volume_score: Mapped[Optional[float]] = mapped_column(Float)
    momentum_score: Mapped[Optional[float]] = mapped_column(Float)
    raw_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    symbol: Mapped[Symbol] = relationship(back_populates="feature_values")


class ModelRun(Base, TimestampMixin):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[Timeframe] = mapped_column(SAEnum(Timeframe), nullable=False)
    horizon_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    train_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    train_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512))


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("model_run_id", "symbol_id", "prediction_time", name="uq_prediction_run_symbol_time"),
        Index("ix_predictions_symbol_time", "symbol_id", "prediction_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    prediction_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    raw_score: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class Signal(Base, TimestampMixin):
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signals_date_score", "signal_time", "final_score"),
        Index("ix_signals_symbol_status", "symbol_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    prediction_id: Mapped[Optional[int]] = mapped_column(ForeignKey("predictions.id"))
    signal_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    timeframe: Mapped[Timeframe] = mapped_column(SAEnum(Timeframe), nullable=False)
    horizon: Mapped[Horizon] = mapped_column(SAEnum(Horizon), default=Horizon.MEDIUM, nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    direction: Mapped[SignalDirection] = mapped_column(SAEnum(SignalDirection), nullable=False)
    status: Mapped[SignalStatus] = mapped_column(SAEnum(SignalStatus), default=SignalStatus.OPEN, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    model_score: Mapped[float] = mapped_column(Float, nullable=False)
    trend_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    volume_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    relative_strength_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_price: Mapped[Optional[float]] = mapped_column(Float)
    target_price: Mapped[Optional[float]] = mapped_column(Float)
    reason: Mapped[Optional[str]] = mapped_column(Text)

    symbol: Mapped[Symbol] = relationship(back_populates="signals")


class DecisionLog(Base):
    __tablename__ = "decision_logs"
    __table_args__ = (
        Index("ix_decision_logs_symbol_time", "symbol_id", "decision_time"),
        Index("ix_decision_logs_signal", "signal_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[Optional[int]] = mapped_column(ForeignKey("signals.id"))
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    decision_time: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    signal_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    timeframe: Mapped[Timeframe] = mapped_column(SAEnum(Timeframe), nullable=False)
    horizon: Mapped[Horizon] = mapped_column(SAEnum(Horizon), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    direction: Mapped[SignalDirection] = mapped_column(SAEnum(SignalDirection), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_price: Mapped[Optional[float]] = mapped_column(Float)
    target_price: Mapped[Optional[float]] = mapped_column(Float)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    model_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    trend_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    volume_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    relative_strength_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    raw_json: Mapped[Optional[str]] = mapped_column(Text)

    signal: Mapped[Optional[Signal]] = relationship()
    symbol: Mapped[Symbol] = relationship()


class SignalOutcome(Base, TimestampMixin):
    __tablename__ = "signal_outcomes"
    __table_args__ = (UniqueConstraint("signal_id", name="uq_signal_outcome_signal"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"), nullable=False)
    return_1d: Mapped[Optional[float]] = mapped_column(Float)
    return_3d: Mapped[Optional[float]] = mapped_column(Float)
    return_5d: Mapped[Optional[float]] = mapped_column(Float)
    return_10d: Mapped[Optional[float]] = mapped_column(Float)
    return_20d: Mapped[Optional[float]] = mapped_column(Float)
    max_gain: Mapped[Optional[float]] = mapped_column(Float)
    max_loss: Mapped[Optional[float]] = mapped_column(Float)
    hit_target: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    hit_stop: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class PortfolioSnapshot(Base, TimestampMixin):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    timeframe: Mapped[Timeframe] = mapped_column(SAEnum(Timeframe), nullable=False)
    horizon: Mapped[Horizon] = mapped_column(SAEnum(Horizon), default=Horizon.MEDIUM, nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)

    items: Mapped[list["PortfolioItem"]] = relationship(back_populates="snapshot")


class PortfolioItem(Base):
    __tablename__ = "portfolio_items"
    __table_args__ = (UniqueConstraint("portfolio_snapshot_id", "symbol_id", name="uq_portfolio_item_snapshot_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_snapshot_id: Mapped[int] = mapped_column(ForeignKey("portfolio_snapshots.id"), nullable=False)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    signal_id: Mapped[Optional[int]] = mapped_column(ForeignKey("signals.id"))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    suggested_weight: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text)

    snapshot: Mapped[PortfolioSnapshot] = relationship(back_populates="items")
    symbol: Mapped[Symbol] = relationship()
    signal: Mapped[Optional[Signal]] = relationship()
