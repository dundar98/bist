"""Backtest module for the BIST100 trading system."""

from .engine import (
    Trade,
    Position,
    BacktestState,
    BacktestEngine,
)
from .metrics import (
    PerformanceMetrics,
    calculate_metrics,
    calculate_rolling_metrics,
)
from .reporter import (
    BacktestReporter,
    generate_report,
)

__all__ = [
    "Trade",
    "Position",
    "BacktestState",
    "BacktestEngine",
    "PerformanceMetrics",
    "calculate_metrics",
    "calculate_rolling_metrics",
    "BacktestReporter",
    "generate_report",
]
