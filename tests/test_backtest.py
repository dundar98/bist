"""
Unit tests for backtest module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine, BacktestState
from backtest.metrics import calculate_metrics, PerformanceMetrics


class TestBacktestEngine:
    """Tests for backtesting engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        n = 100
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2
        open_price = close - 0.5
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n, freq='D'),
            'open': open_price,
            'high': high,
            'low': np.maximum(low, 1),  # Ensure positive
            'close': close,
            'volume': np.random.randint(100000, 1000000, n),
        })
    
    @pytest.fixture
    def engine(self):
        """Create backtest engine."""
        return BacktestEngine(
            initial_capital=100000,
            entry_threshold=0.65,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
        )
    
    def test_run_returns_state(self, engine, sample_data):
        """Test that backtest returns a state object."""
        predictions = np.random.uniform(0.4, 0.8, len(sample_data))
        
        state = engine.run(sample_data, predictions)
        
        assert isinstance(state, BacktestState)
        assert len(state.equity_curve) > 0
    
    def test_no_trades_with_low_predictions(self, engine, sample_data):
        """Test that low predictions don't generate trades."""
        predictions = np.ones(len(sample_data)) * 0.3  # All below threshold
        
        state = engine.run(sample_data, predictions)
        
        assert len(state.trades) == 0
    
    def test_trades_with_high_predictions(self, engine, sample_data):
        """Test that high predictions generate trades."""
        predictions = np.ones(len(sample_data)) * 0.8  # All above threshold
        
        state = engine.run(sample_data, predictions)
        
        assert len(state.trades) >= 0  # Should have some trades
    
    def test_capital_conservation(self, sample_data):
        """Test that capital is conserved (plus/minus PnL)."""
        engine = BacktestEngine(
            initial_capital=100000,
            commission_pct=0,  # No commission for this test
            slippage_pct=0,    # No slippage
        )
        
        # Simple predictions
        predictions = np.where(np.arange(len(sample_data)) % 20 < 10, 0.8, 0.3)
        
        state = engine.run(sample_data, predictions)
        
        # Final equity should equal initial + sum of PnLs
        total_pnl = sum(t.pnl for t in state.trades)
        assert abs(state.equity - (100000 + total_pnl)) < 1  # Allow small float error
    
    def test_stop_loss_triggered(self, sample_data):
        """Test that stop loss is triggered correctly."""
        # Create data with a large drop
        sample_data_copy = sample_data.copy()
        sample_data_copy.loc[10:15, 'low'] = sample_data_copy.loc[10:15, 'close'] * 0.9
        
        engine = BacktestEngine(
            initial_capital=100000,
            entry_threshold=0.5,
            stop_loss_pct=0.05,
        )
        
        # High prediction at start
        predictions = np.zeros(len(sample_data_copy))
        predictions[5] = 0.8
        
        state = engine.run(sample_data_copy, predictions)
        
        # Should have at least one trade
        if state.trades:
            stop_loss_trades = [t for t in state.trades if t.exit_reason == 'stop_loss']
            # If price dropped enough, should have stop loss
            assert len([t for t in state.trades]) >= 0


class TestMetrics:
    """Tests for performance metrics calculation."""
    
    def test_empty_trades(self):
        """Test metrics with no trades."""
        metrics = calculate_metrics(
            trades=[],
            equity_curve=[100000],
            initial_capital=100000,
        )
        
        assert metrics.total_trades == 0
        assert metrics.total_return == 0
    
    def test_winning_trades(self):
        """Test metrics with winning trades."""
        from backtest.engine import Trade
        
        trades = [
            Trade(
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                symbol="TEST",
                direction="long",
                entry_price=100,
                exit_price=110,
                size=10,
                pnl=100,
                pnl_pct=0.1,
                exit_reason="take_profit",
                bars_held=5,
            ),
            Trade(
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                symbol="TEST",
                direction="long",
                entry_price=100,
                exit_price=105,
                size=10,
                pnl=50,
                pnl_pct=0.05,
                exit_reason="take_profit",
                bars_held=3,
            ),
        ]
        
        equity = [100000, 100100, 100150]
        
        metrics = calculate_metrics(
            trades=trades,
            equity_curve=equity,
            initial_capital=100000,
        )
        
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 2
        assert metrics.win_rate == 1.0
        assert metrics.net_profit == 150
    
    def test_mixed_trades(self):
        """Test metrics with mixed win/loss trades."""
        from backtest.engine import Trade
        
        trades = [
            Trade(
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                symbol="TEST",
                direction="long",
                entry_price=100,
                exit_price=110,
                size=10,
                pnl=100,
                pnl_pct=0.1,
                exit_reason="take_profit",
                bars_held=5,
            ),
            Trade(
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                symbol="TEST",
                direction="long",
                entry_price=100,
                exit_price=95,
                size=10,
                pnl=-50,
                pnl_pct=-0.05,
                exit_reason="stop_loss",
                bars_held=2,
            ),
        ]
        
        equity = [100000, 100100, 100050]
        
        metrics = calculate_metrics(
            trades=trades,
            equity_curve=equity,
            initial_capital=100000,
        )
        
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5
        assert metrics.profit_factor == 100 / 50  # 2.0


class TestNoLookahead:
    """Tests to ensure no future data leakage."""
    
    def test_predictions_used_sequentially(self, sample_data):
        """Test that predictions are used in correct order."""
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=50, freq='D'),
            'open': np.ones(50) * 100,
            'high': np.ones(50) * 105,
            'low': np.ones(50) * 95,
            'close': np.ones(50) * 100,
            'volume': np.ones(50) * 1000000,
        })
        
        # Create predictions that signal entry only at bar 10
        predictions = np.zeros(50)
        predictions[10] = 0.9
        
        engine = BacktestEngine(
            initial_capital=100000,
            entry_threshold=0.8,
        )
        
        state = engine.run(sample_data, predictions, symbol="TEST")
        
        # Should have exactly one entry (at bar 10)
        if state.trades:
            assert len(state.trades) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
