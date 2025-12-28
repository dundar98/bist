"""
Performance Metrics Module.

Calculates comprehensive trading performance metrics
from backtest results.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk metrics
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # PnL analysis
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    
    # Average trade
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    avg_win_loss_ratio: float
    
    # Expectancy
    expectancy: float
    expectancy_pct: float
    
    # Duration
    avg_bars_held: float
    avg_winning_bars: float
    avg_losing_bars: float
    
    # Best/Worst
    best_trade: float
    worst_trade: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'net_profit': self.net_profit,
            'profit_factor': self.profit_factor,
            'avg_trade_pnl': self.avg_trade_pnl,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_win_loss_ratio': self.avg_win_loss_ratio,
            'expectancy': self.expectancy,
            'expectancy_pct': self.expectancy_pct,
            'avg_bars_held': self.avg_bars_held,
            'avg_winning_bars': self.avg_winning_bars,
            'avg_losing_bars': self.avg_losing_bars,
            'best_trade': self.best_trade,
            'worst_trade': self.worst_trade,
        }
    
    def summary(self) -> str:
        """Return formatted summary string."""
        return f"""
========== BACKTEST RESULTS ==========

RETURNS
  Total Return:      {self.total_return:>10.2%}
  Annualized Return: {self.annualized_return:>10.2%}
  Net Profit:        {self.net_profit:>10.2f}

RISK METRICS
  Max Drawdown:      {self.max_drawdown:>10.2%}
  Volatility:        {self.volatility:>10.2%}
  Sharpe Ratio:      {self.sharpe_ratio:>10.2f}
  Sortino Ratio:     {self.sortino_ratio:>10.2f}
  Calmar Ratio:      {self.calmar_ratio:>10.2f}

TRADE STATISTICS
  Total Trades:      {self.total_trades:>10d}
  Win Rate:          {self.win_rate:>10.2%}
  Profit Factor:     {self.profit_factor:>10.2f}

AVERAGE TRADE
  Avg Trade PnL:     {self.avg_trade_pnl:>10.2f}
  Avg Win:           {self.avg_win:>10.2f}
  Avg Loss:          {self.avg_loss:>10.2f}
  Win/Loss Ratio:    {self.avg_win_loss_ratio:>10.2f}

EXPECTANCY
  Expectancy $:      {self.expectancy:>10.2f}
  Expectancy %:      {self.expectancy_pct:>10.2%}

DURATION
  Avg Bars Held:     {self.avg_bars_held:>10.1f}

EXTREMES
  Best Trade:        {self.best_trade:>10.2f}
  Worst Trade:       {self.worst_trade:>10.2f}

==========================================
"""


def calculate_metrics(
    trades: List,
    equity_curve: List[float],
    initial_capital: float,
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        trades: List of Trade objects
        equity_curve: List of equity values over time
        initial_capital: Starting capital
        trading_days_per_year: Number of trading days per year
        
    Returns:
        PerformanceMetrics object
    """
    # Handle empty trades
    if not trades:
        return _empty_metrics()
    
    # Extract trade PnLs
    pnls = np.array([t.pnl for t in trades])
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    bars_held = np.array([t.bars_held for t in trades])
    
    # Identify winners and losers
    winners_mask = pnls > 0
    losers_mask = pnls < 0
    
    winning_pnls = pnls[winners_mask]
    losing_pnls = pnls[losers_mask]
    
    # Returns
    equity = np.array(equity_curve)
    total_return = (equity[-1] - initial_capital) / initial_capital if len(equity) > 0 else 0
    
    n_periods = len(equity)
    if n_periods > 1:
        annualized_return = (1 + total_return) ** (trading_days_per_year / n_periods) - 1
    else:
        annualized_return = 0
    
    # Risk metrics
    max_drawdown = _calculate_max_drawdown(equity)
    
    # Daily returns for volatility calculation
    if len(equity) > 1:
        returns = np.diff(equity) / equity[:-1]
        volatility = np.std(returns) * np.sqrt(trading_days_per_year)
        
        # Sharpe (assuming risk-free rate = 0)
        mean_return = np.mean(returns)
        sharpe_ratio = (mean_return * trading_days_per_year) / (volatility + 1e-8)
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(trading_days_per_year) if len(downside_returns) > 0 else 0.01
        sortino_ratio = (mean_return * trading_days_per_year) / (downside_std + 1e-8)
    else:
        volatility = 0
        sharpe_ratio = 0
        sortino_ratio = 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / (max_drawdown + 1e-8) if max_drawdown > 0 else 0
    
    # Trade statistics
    total_trades = len(trades)
    winning_trades = len(winning_pnls)
    losing_trades = len(losing_pnls)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # PnL analysis
    gross_profit = winning_pnls.sum() if len(winning_pnls) > 0 else 0
    gross_loss = abs(losing_pnls.sum()) if len(losing_pnls) > 0 else 0
    net_profit = pnls.sum()
    profit_factor = gross_profit / (gross_loss + 1e-8) if gross_loss > 0 else float('inf')
    
    # Average trade
    avg_trade_pnl = pnls.mean() if len(pnls) > 0 else 0
    avg_win = winning_pnls.mean() if len(winning_pnls) > 0 else 0
    avg_loss = losing_pnls.mean() if len(losing_pnls) > 0 else 0
    avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    expectancy_pct = pnl_pcts.mean() if len(pnl_pcts) > 0 else 0
    
    # Duration
    avg_bars_held = bars_held.mean() if len(bars_held) > 0 else 0
    avg_winning_bars = bars_held[winners_mask].mean() if winners_mask.sum() > 0 else 0
    avg_losing_bars = bars_held[losers_mask].mean() if losers_mask.sum() > 0 else 0
    
    # Extremes
    best_trade = pnls.max() if len(pnls) > 0 else 0
    worst_trade = pnls.min() if len(pnls) > 0 else 0
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_win_loss_ratio=avg_win_loss_ratio,
        expectancy=expectancy,
        expectancy_pct=expectancy_pct,
        avg_bars_held=avg_bars_held,
        avg_winning_bars=avg_winning_bars,
        avg_losing_bars=avg_losing_bars,
        best_trade=best_trade,
        worst_trade=worst_trade,
    )


def _calculate_max_drawdown(equity: np.ndarray) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity) < 2:
        return 0.0
    
    peak = equity[0]
    max_dd = 0.0
    
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def _empty_metrics() -> PerformanceMetrics:
    """Return empty metrics for no trades."""
    return PerformanceMetrics(
        total_return=0,
        annualized_return=0,
        max_drawdown=0,
        volatility=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        calmar_ratio=0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0,
        gross_profit=0,
        gross_loss=0,
        net_profit=0,
        profit_factor=0,
        avg_trade_pnl=0,
        avg_win=0,
        avg_loss=0,
        avg_win_loss_ratio=0,
        expectancy=0,
        expectancy_pct=0,
        avg_bars_held=0,
        avg_winning_bars=0,
        avg_losing_bars=0,
        best_trade=0,
        worst_trade=0,
    )


def calculate_rolling_metrics(
    equity_curve: List[float],
    window: int = 20,
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        equity_curve: Equity values over time
        window: Rolling window size
        
    Returns:
        DataFrame with rolling metrics
    """
    equity = pd.Series(equity_curve)
    returns = equity.pct_change()
    
    df = pd.DataFrame({
        'equity': equity,
        'returns': returns,
        'cumulative_return': (1 + returns).cumprod() - 1,
    })
    
    # Rolling metrics
    df['rolling_return'] = returns.rolling(window).mean() * 252
    df['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
    df['rolling_sharpe'] = df['rolling_return'] / (df['rolling_volatility'] + 1e-8)
    
    # Running max for drawdown
    df['running_max'] = equity.cummax()
    df['drawdown'] = (df['running_max'] - equity) / df['running_max']
    
    return df
