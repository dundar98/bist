"""
Backtest Engine.

Simulates trading using model predictions with proper handling of
transaction costs, slippage, and position management.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_signal: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'entry_signal': self.entry_signal,
            'exit_reason': self.exit_reason,
            'bars_held': self.bars_held,
        }


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str
    entry_price: float
    size: float
    entry_time: datetime
    entry_signal: float
    bars_held: int = 0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    
    def update_bars(self, current_high: float, current_low: float) -> None:
        """Update position state with new bar."""
        self.bars_held += 1
        self.highest_price = max(self.highest_price, current_high)
        self.lowest_price = min(self.lowest_price, current_low) if self.lowest_price > 0 else current_low


@dataclass 
class BacktestState:
    """Current state of the backtest."""
    cash: float
    equity: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    peak_equity: float = 0.0
    daily_pnl: float = 0.0


class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    
    Processes historical data bar-by-bar and simulates trading
    based on model predictions.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        entry_threshold: float = 0.65,
        exit_threshold: float = 0.35,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        position_size_pct: float = 0.1,
        max_positions: int = 5,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            entry_threshold: Min probability to enter trade
            exit_threshold: Probability below which to exit
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size_pct: Fraction of capital per position
            max_positions: Maximum concurrent positions
            commission_pct: Commission per trade
            slippage_pct: Slippage per trade
        """
        self.initial_capital = initial_capital
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        # State
        self.state: Optional[BacktestState] = None
    
    def run(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        symbol: str = "UNKNOWN",
    ) -> BacktestState:
        """
        Run backtest on data with predictions.
        
        Args:
            data: DataFrame with OHLCV data (must have timestamp, open, high, low, close)
            predictions: Array of model predictions (probabilities)
            symbol: Symbol being backtested
            
        Returns:
            BacktestState with results
        """
        # Validate inputs
        if len(data) != len(predictions):
            raise ValueError(
                f"Data length ({len(data)}) != predictions length ({len(predictions)})"
            )
        
        # Initialize state
        self.state = BacktestState(
            cash=self.initial_capital,
            equity=self.initial_capital,
            peak_equity=self.initial_capital,
        )
        
        logger.info(f"Starting backtest for {symbol} with {len(data)} bars")
        
        # Process each bar
        for i in range(len(data)):
            bar = data.iloc[i]
            prob = predictions[i]
            
            timestamp = bar.get('timestamp', i)
            open_price = bar['open']
            high = bar['high']
            low = bar['low']
            close = bar['close']
            
            # 1. Check existing positions for stop/take profit
            self._check_exits(timestamp, high, low, close)
            
            # 2. Check for signal-based exits
            if symbol in self.state.positions and prob < self.exit_threshold:
                self._close_position(
                    symbol=symbol,
                    exit_time=timestamp,
                    exit_price=close,
                    reason="signal_exit"
                )
            
            # 3. Check for new entries
            if (prob >= self.entry_threshold and 
                symbol not in self.state.positions and
                len(self.state.positions) < self.max_positions):
                
                self._open_position(
                    symbol=symbol,
                    entry_time=timestamp,
                    entry_price=close,
                    signal=prob,
                )
            
            # 4. Update position bars
            for pos in self.state.positions.values():
                pos.update_bars(high, low)
            
            # 5. Update equity
            self._update_equity(data.iloc[:i+1])
            
            # 6. Record equity curve
            self.state.equity_curve.append(self.state.equity)
            
            # Update peak and drawdown
            if self.state.equity > self.state.peak_equity:
                self.state.peak_equity = self.state.equity
            
            drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            self.state.drawdown_curve.append(drawdown)
        
        # Close any remaining positions at end
        for symbol in list(self.state.positions.keys()):
            self._close_position(
                symbol=symbol,
                exit_time=data.iloc[-1].get('timestamp', len(data)),
                exit_price=data.iloc[-1]['close'],
                reason="end_of_data"
            )
        
        logger.info(
            f"Backtest complete: {len(self.state.trades)} trades, "
            f"final equity: {self.state.equity:.2f}"
        )
        
        return self.state
    
    def run_multi_symbol(
        self,
        data_dict: Dict[str, pd.DataFrame],
        predictions_dict: Dict[str, np.ndarray],
    ) -> BacktestState:
        """
        Run backtest on multiple symbols.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            predictions_dict: Dictionary mapping symbol to predictions
            
        Returns:
            BacktestState with combined results
        """
        # Combine all data into single timeline
        all_events = []
        
        for symbol, df in data_dict.items():
            preds = predictions_dict[symbol]
            for i in range(len(df)):
                all_events.append({
                    'timestamp': df.iloc[i].get('timestamp', i),
                    'symbol': symbol,
                    'data': df.iloc[i],
                    'prediction': preds[i],
                })
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x['timestamp'])
        
        # Initialize state
        self.state = BacktestState(
            cash=self.initial_capital,
            equity=self.initial_capital,
            peak_equity=self.initial_capital,
        )
        
        # Process each event
        for event in all_events:
            symbol = event['symbol']
            bar = event['data']
            prob = event['prediction']
            timestamp = event['timestamp']
            
            high = bar['high']
            low = bar['low']
            close = bar['close']
            
            # Check exits
            self._check_exits(timestamp, high, low, close, symbol)
            
            # Signal exit
            if symbol in self.state.positions and prob < self.exit_threshold:
                self._close_position(symbol, timestamp, close, "signal_exit")
            
            # New entry
            if (prob >= self.entry_threshold and 
                symbol not in self.state.positions and
                len(self.state.positions) < self.max_positions):
                self._open_position(symbol, timestamp, close, prob)
            
            # Update bars
            if symbol in self.state.positions:
                self.state.positions[symbol].update_bars(high, low)
            
            # Update equity
            self._update_equity_multi(data_dict)
            self.state.equity_curve.append(self.state.equity)
        
        return self.state
    
    def _open_position(
        self,
        symbol: str,
        entry_time: datetime,
        entry_price: float,
        signal: float,
    ) -> None:
        """Open a new position."""
        # Apply slippage
        actual_price = entry_price * (1 + self.slippage_pct)
        
        # Calculate position size
        capital_for_trade = self.state.cash * self.position_size_pct
        
        # Apply commission
        commission = capital_for_trade * self.commission_pct
        capital_for_trade -= commission
        
        size = capital_for_trade / actual_price
        
        if size <= 0:
            return
        
        # Create position
        position = Position(
            symbol=symbol,
            direction='long',
            entry_price=actual_price,
            size=size,
            entry_time=entry_time,
            entry_signal=signal,
            highest_price=actual_price,
            lowest_price=actual_price,
        )
        
        self.state.positions[symbol] = position
        self.state.cash -= (capital_for_trade + commission)
        
        logger.debug(
            f"OPEN {symbol}: price={actual_price:.2f}, size={size:.2f}, "
            f"signal={signal:.3f}"
        )
    
    def _close_position(
        self,
        symbol: str,
        exit_time: datetime,
        exit_price: float,
        reason: str,
    ) -> None:
        """Close an existing position."""
        if symbol not in self.state.positions:
            return
        
        position = self.state.positions[symbol]
        
        # Apply slippage
        actual_price = exit_price * (1 - self.slippage_pct)
        
        # Calculate PnL
        gross_value = position.size * actual_price
        commission = gross_value * self.commission_pct
        net_value = gross_value - commission
        
        cost_basis = position.size * position.entry_price
        pnl = net_value - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0
        
        # Create trade record
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=actual_price,
            size=position.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_signal=position.entry_signal,
            exit_reason=reason,
            bars_held=position.bars_held,
        )
        
        self.state.trades.append(trade)
        self.state.cash += net_value
        
        del self.state.positions[symbol]
        
        logger.debug(
            f"CLOSE {symbol}: price={actual_price:.2f}, pnl={pnl:.2f} ({pnl_pct:.1%}), "
            f"reason={reason}"
        )
    
    def _check_exits(
        self,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        symbol: Optional[str] = None,
    ) -> None:
        """Check stop loss and take profit for open positions."""
        symbols_to_check = [symbol] if symbol else list(self.state.positions.keys())
        
        for sym in symbols_to_check:
            if sym not in self.state.positions:
                continue
            
            position = self.state.positions[sym]
            
            # Stop loss check (using low)
            stop_price = position.entry_price * (1 - self.stop_loss_pct)
            if low <= stop_price:
                self._close_position(sym, timestamp, stop_price, "stop_loss")
                continue
            
            # Take profit check (using high)
            tp_price = position.entry_price * (1 + self.take_profit_pct)
            if high >= tp_price:
                self._close_position(sym, timestamp, tp_price, "take_profit")
    
    def _update_equity(self, data: pd.DataFrame) -> None:
        """Update current equity value."""
        positions_value = 0.0
        
        for symbol, position in self.state.positions.items():
            # Use latest close price
            current_price = data.iloc[-1]['close']
            positions_value += position.size * current_price
        
        self.state.equity = self.state.cash + positions_value
    
    def _update_equity_multi(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Update equity for multi-symbol backtest."""
        positions_value = 0.0
        
        for symbol, position in self.state.positions.items():
            if symbol in data_dict:
                current_price = data_dict[symbol].iloc[-1]['close']
                positions_value += position.size * current_price
        
        self.state.equity = self.state.cash + positions_value
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.state or not self.state.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.state.trades])
