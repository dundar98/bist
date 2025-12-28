"""
Backtest Reporter Module.

Generates reports and visualizations from backtest results.
"""

import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

import pandas as pd
import numpy as np

from .engine import BacktestState
from .metrics import PerformanceMetrics, calculate_metrics, calculate_rolling_metrics

logger = logging.getLogger(__name__)


class BacktestReporter:
    """
    Generates comprehensive reports from backtest results.
    """
    
    def __init__(
        self,
        state: BacktestState,
        initial_capital: float,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize reporter.
        
        Args:
            state: BacktestState from backtest run
            initial_capital: Initial capital
            output_dir: Directory to save reports
        """
        self.state = state
        self.initial_capital = initial_capital
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
        
        # Calculate metrics
        self.metrics = calculate_metrics(
            trades=state.trades,
            equity_curve=state.equity_curve,
            initial_capital=initial_capital,
        )
    
    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print(self.metrics.summary())
    
    def get_metrics(self) -> PerformanceMetrics:
        """Return performance metrics."""
        return self.metrics
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.state.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.state.trades])
    
    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame({
            'equity': self.state.equity_curve,
            'drawdown': self.state.drawdown_curve,
        })
    
    def save_report(self, name: str = "backtest") -> None:
        """
        Save all reports to files.
        
        Args:
            name: Base name for report files
        """
        if self.output_dir is None:
            logger.warning("No output directory specified, skipping save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{name}_{timestamp}"
        
        # Save metrics
        metrics_path = self.output_dir / f"{prefix}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save trades
        trades_df = self.get_trades_df()
        if not trades_df.empty:
            trades_path = self.output_dir / f"{prefix}_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Saved trades to {trades_path}")
        
        # Save equity curve
        equity_df = self.get_equity_df()
        equity_path = self.output_dir / f"{prefix}_equity.csv"
        equity_df.to_csv(equity_path, index=False)
        logger.info(f"Saved equity curve to {equity_path}")
        
        # Save summary text
        summary_path = self.output_dir / f"{prefix}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(self.metrics.summary())
        logger.info(f"Saved summary to {summary_path}")
    
    def plot_equity_curve(self, save: bool = False) -> None:
        """
        Plot equity curve with drawdown.
        
        Args:
            save: Whether to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Equity curve
        ax1.plot(self.state.equity_curve, label='Equity', color='blue')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_ylabel('Equity ($)')
        ax1.set_title('Backtest Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(
            range(len(self.state.drawdown_curve)),
            [d * 100 for d in self.state.drawdown_curve],
            color='red',
            alpha=0.5,
            label='Drawdown'
        )
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Bar')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save and self.output_dir:
            path = self.output_dir / 'equity_curve.png'
            plt.savefig(path, dpi=150)
            logger.info(f"Saved equity curve plot to {path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_trade_analysis(self, save: bool = False) -> None:
        """
        Plot trade analysis charts.
        
        Args:
            save: Whether to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
            return
        
        trades_df = self.get_trades_df()
        if trades_df.empty:
            logger.warning("No trades to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # PnL distribution
        ax1 = axes[0, 0]
        ax1.hist(trades_df['pnl'], bins=30, color='steelblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--')
        ax1.set_xlabel('PnL ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Trade PnL Distribution')
        
        # Win/Loss by exit reason
        ax2 = axes[0, 1]
        if 'exit_reason' in trades_df.columns:
            reason_pnl = trades_df.groupby('exit_reason')['pnl'].sum()
            colors = ['green' if x > 0 else 'red' for x in reason_pnl.values]
            reason_pnl.plot(kind='bar', ax=ax2, color=colors)
            ax2.set_ylabel('Total PnL ($)')
            ax2.set_title('PnL by Exit Reason')
            ax2.tick_params(axis='x', rotation=45)
        
        # Cumulative PnL
        ax3 = axes[1, 0]
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax3.plot(cumulative_pnl, color='green' if cumulative_pnl.iloc[-1] > 0 else 'red')
        ax3.axhline(y=0, color='gray', linestyle='--')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Cumulative PnL ($)')
        ax3.set_title('Cumulative PnL by Trade')
        
        # Trade duration vs PnL
        ax4 = axes[1, 1]
        colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
        ax4.scatter(trades_df['bars_held'], trades_df['pnl'], c=colors, alpha=0.6)
        ax4.axhline(y=0, color='gray', linestyle='--')
        ax4.set_xlabel('Bars Held')
        ax4.set_ylabel('PnL ($)')
        ax4.set_title('Trade Duration vs PnL')
        
        plt.tight_layout()
        
        if save and self.output_dir:
            path = self.output_dir / 'trade_analysis.png'
            plt.savefig(path, dpi=150)
            logger.info(f"Saved trade analysis plot to {path}")
        else:
            plt.show()
        
        plt.close()


def generate_report(
    state: BacktestState,
    initial_capital: float,
    output_dir: Optional[str] = None,
    print_summary: bool = True,
    save_files: bool = True,
    plot: bool = False,
) -> BacktestReporter:
    """
    Convenience function to generate full backtest report.
    
    Args:
        state: BacktestState from backtest
        initial_capital: Initial capital
        output_dir: Directory to save files
        print_summary: Whether to print summary
        save_files: Whether to save output files
        plot: Whether to generate plots
        
    Returns:
        BacktestReporter instance
    """
    reporter = BacktestReporter(
        state=state,
        initial_capital=initial_capital,
        output_dir=output_dir,
    )
    
    if print_summary:
        reporter.print_summary()
    
    if save_files and output_dir:
        reporter.save_report()
    
    if plot:
        reporter.plot_equity_curve(save=bool(output_dir))
        reporter.plot_trade_analysis(save=bool(output_dir))
    
    return reporter
