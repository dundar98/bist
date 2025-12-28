"""
Paper Trading Simulator.

Simulates real-time trading bar-by-bar with no look-ahead.
Logs all decisions with full reasoning for debugging.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch

from models.base import BaseModel
from data.features import FeatureEngine, FeatureNormalizer
from strategy.signals import SignalGenerator, Signal
from strategy.position import PositionSizer, get_position_sizer
from risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """A trading decision with full reasoning."""
    timestamp: datetime
    symbol: str
    bar_index: int
    probability: float
    signal: str
    action: str  # 'buy', 'sell', 'hold', 'skip'
    reason: str
    position_size: Optional[float] = None
    entry_price: Optional[float] = None
    risk_check: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': str(self.timestamp),
            'symbol': self.symbol,
            'bar_index': self.bar_index,
            'probability': self.probability,
            'signal': self.signal,
            'action': self.action,
            'reason': self.reason,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'risk_check': self.risk_check,
            'confidence': self.confidence,
        }


@dataclass
class PaperPosition:
    """An open paper trading position."""
    symbol: str
    entry_time: datetime
    entry_price: float
    size: float
    entry_signal: float
    bars_held: int = 0
    highest_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PaperTrade:
    """A completed paper trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    bars_held: int


class PaperTradingSimulator:
    """
    Simulates paper trading with bar-by-bar prediction.
    
    Key features:
    1. Processes data sequentially (no look-ahead)
    2. Computes features in real-time
    3. Logs every decision with reasoning
    4. Integrates with risk manager
    """
    
    def __init__(
        self,
        model: BaseModel,
        feature_columns: List[str],
        lookback: int = 60,
        initial_capital: float = 100000.0,
        entry_threshold: float = 0.65,
        exit_threshold: float = 0.35,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        position_size_method: str = "fixed_fraction",
        log_file: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize paper trading simulator.
        
        Args:
            model: Trained model for predictions
            feature_columns: List of feature column names
            lookback: Sequence length for model input
            initial_capital: Starting capital
            entry_threshold: Probability threshold for entry
            exit_threshold: Probability threshold for exit
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size_method: Position sizing method
            log_file: Path to save decision log
            device: Device for model inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_columns = feature_columns
        self.lookback = lookback
        self.device = device
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Components
        self.signal_generator = SignalGenerator(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
        self.position_sizer = get_position_sizer(position_size_method)
        self.risk_manager = RiskManager(initial_capital=initial_capital)
        
        # State
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        self.decisions: List[Decision] = []
        self.equity_history: List[float] = []
        
        # Feature history for each symbol
        self.feature_history: Dict[str, List[np.ndarray]] = {}
        
        # Logging
        self.log_file = log_file
        if log_file:
            self.log_path = Path(log_file)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def simulate(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """
        Run paper trading simulation on historical data.
        
        Args:
            data: DataFrame with OHLCV and features
            symbol: Symbol being traded
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting paper trading simulation for {symbol}")
        
        # Reset state
        self._reset()
        
        # Ensure features exist
        if not all(col in data.columns for col in self.feature_columns):
            missing = [c for c in self.feature_columns if c not in data.columns]
            raise ValueError(f"Missing feature columns: {missing}")
        
        # Process each bar
        for i in range(len(data)):
            bar = data.iloc[i]
            timestamp = bar.get('timestamp', i)
            
            # Update feature history
            features = bar[self.feature_columns].values.astype(np.float32)
            if symbol not in self.feature_history:
                self.feature_history[symbol] = []
            self.feature_history[symbol].append(features)
            
            # Wait for enough history
            if len(self.feature_history[symbol]) < self.lookback:
                self._log_decision(Decision(
                    timestamp=timestamp,
                    symbol=symbol,
                    bar_index=i,
                    probability=0.5,
                    signal="hold",
                    action="skip",
                    reason=f"Building history: {len(self.feature_history[symbol])}/{self.lookback}",
                ))
                continue
            
            # Check existing positions for stop/take profit
            self._check_position_exits(symbol, bar, timestamp)
            
            # Skip prediction if we don't have a position and can't trade
            if self.risk_manager.is_circuit_breaker_triggered():
                self._log_decision(Decision(
                    timestamp=timestamp,
                    symbol=symbol,
                    bar_index=i,
                    probability=0.5,
                    signal="hold",
                    action="skip",
                    reason="Circuit breaker active",
                ))
                continue
            
            # Get prediction
            probability = self._predict(symbol)
            
            # Generate signal
            current_position = 'long' if symbol in self.positions else None
            signal_result = self.signal_generator.generate(
                probability=probability,
                current_position=current_position,
            )
            
            # Make trading decision
            self._process_signal(
                signal_result=signal_result,
                symbol=symbol,
                bar=bar,
                timestamp=timestamp,
                bar_index=i,
            )
            
            # Update equity
            self._update_equity(data.iloc[:i+1])
            self.equity_history.append(self.capital + self._get_positions_value(bar))
        
        # Close remaining positions
        for sym in list(self.positions.keys()):
            final_bar = data.iloc[-1]
            self._close_position(
                symbol=sym,
                exit_price=final_bar['close'],
                exit_time=final_bar.get('timestamp', len(data)),
                reason="simulation_end"
            )
        
        # Save log
        if self.log_file:
            self._save_log()
        
        # Compile results
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'total_trades': len(self.trades),
            'winning_trades': sum(1 for t in self.trades if t.pnl > 0),
            'losing_trades': sum(1 for t in self.trades if t.pnl < 0),
            'total_decisions': len(self.decisions),
            'equity_history': self.equity_history,
        }
        
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            results['avg_pnl'] = np.mean(pnls)
            results['max_pnl'] = np.max(pnls)
            results['min_pnl'] = np.min(pnls)
            results['win_rate'] = results['winning_trades'] / results['total_trades']
        
        logger.info(
            f"Simulation complete: {results['total_trades']} trades, "
            f"return: {results['total_return']:.2%}"
        )
        
        return results
    
    def _predict(self, symbol: str) -> float:
        """Make prediction for current bar."""
        # Get last lookback features
        features = np.array(self.feature_history[symbol][-self.lookback:])
        
        # Handle NaN
        features = np.nan_to_num(features, nan=0.0)
        
        # Convert to tensor
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prob = self.model(x)
        
        return prob.item()
    
    def _process_signal(
        self,
        signal_result,
        symbol: str,
        bar: pd.Series,
        timestamp: datetime,
        bar_index: int,
    ) -> None:
        """Process a trading signal and make a decision."""
        probability = signal_result.probability
        signal = signal_result.signal
        
        close_price = bar['close']
        
        # No position - check for entry
        if symbol not in self.positions:
            if signal == Signal.BUY:
                # Calculate position size
                size_result = self.position_sizer.calculate(
                    capital=self.capital,
                    price=close_price,
                    stop_loss_pct=self.stop_loss_pct,
                    probability=probability,
                )
                
                # Check risk
                risk_check = self.risk_manager.check_trade_allowed(
                    position_size=size_result.size,
                    price=close_price,
                    stop_loss_pct=self.stop_loss_pct,
                    capital=self.capital,
                )
                
                if risk_check.allowed:
                    # Open position
                    self._open_position(
                        symbol=symbol,
                        entry_price=close_price,
                        size=size_result.size,
                        entry_time=timestamp,
                        entry_signal=probability,
                    )
                    
                    self._log_decision(Decision(
                        timestamp=timestamp,
                        symbol=symbol,
                        bar_index=bar_index,
                        probability=probability,
                        signal=signal.value,
                        action="buy",
                        reason=signal_result.reason,
                        position_size=size_result.size,
                        entry_price=close_price,
                        risk_check=risk_check.reason,
                        confidence=signal_result.confidence,
                    ))
                else:
                    self._log_decision(Decision(
                        timestamp=timestamp,
                        symbol=symbol,
                        bar_index=bar_index,
                        probability=probability,
                        signal=signal.value,
                        action="skip",
                        reason=f"Risk check failed: {risk_check.reason}",
                        risk_check=risk_check.reason,
                        confidence=signal_result.confidence,
                    ))
            else:
                self._log_decision(Decision(
                    timestamp=timestamp,
                    symbol=symbol,
                    bar_index=bar_index,
                    probability=probability,
                    signal=signal.value,
                    action="hold",
                    reason=signal_result.reason,
                    confidence=signal_result.confidence,
                ))
        
        # Has position - check for exit
        else:
            position = self.positions[symbol]
            position.bars_held += 1
            position.highest_price = max(position.highest_price, bar['high'])
            position.unrealized_pnl = (close_price - position.entry_price) * position.size
            
            if signal == Signal.SELL:
                self._close_position(
                    symbol=symbol,
                    exit_price=close_price,
                    exit_time=timestamp,
                    reason="signal_exit"
                )
                
                self._log_decision(Decision(
                    timestamp=timestamp,
                    symbol=symbol,
                    bar_index=bar_index,
                    probability=probability,
                    signal=signal.value,
                    action="sell",
                    reason=signal_result.reason,
                    confidence=signal_result.confidence,
                ))
            else:
                self._log_decision(Decision(
                    timestamp=timestamp,
                    symbol=symbol,
                    bar_index=bar_index,
                    probability=probability,
                    signal=signal.value,
                    action="hold",
                    reason=f"Holding position, {position.bars_held} bars",
                    confidence=signal_result.confidence,
                ))
    
    def _check_position_exits(
        self,
        symbol: str,
        bar: pd.Series,
        timestamp: datetime,
    ) -> None:
        """Check stop loss and take profit for position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        low = bar['low']
        high = bar['high']
        
        # Stop loss
        stop_price = position.entry_price * (1 - self.stop_loss_pct)
        if low <= stop_price:
            self._close_position(
                symbol=symbol,
                exit_price=stop_price,
                exit_time=timestamp,
                reason="stop_loss"
            )
            return
        
        # Take profit
        tp_price = position.entry_price * (1 + self.take_profit_pct)
        if high >= tp_price:
            self._close_position(
                symbol=symbol,
                exit_price=tp_price,
                exit_time=timestamp,
                reason="take_profit"
            )
    
    def _open_position(
        self,
        symbol: str,
        entry_price: float,
        size: float,
        entry_time: datetime,
        entry_signal: float,
    ) -> None:
        """Open a new position."""
        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            size=size,
            entry_signal=entry_signal,
            highest_price=entry_price,
        )
        
        self.capital -= size * entry_price
        logger.debug(f"OPEN {symbol}: price={entry_price:.2f}, size={size:.2f}")
    
    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ) -> None:
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        pnl = (exit_price - position.entry_price) * position.size
        pnl_pct = (exit_price - position.entry_price) / position.entry_price
        
        trade = PaperTrade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            bars_held=position.bars_held,
        )
        
        self.trades.append(trade)
        self.capital += position.size * exit_price
        self.risk_manager.update_pnl(pnl)
        
        del self.positions[symbol]
        
        logger.debug(
            f"CLOSE {symbol}: price={exit_price:.2f}, pnl={pnl:.2f}, reason={reason}"
        )
    
    def _update_equity(self, data: pd.DataFrame) -> None:
        """Update equity tracking."""
        positions_value = 0
        for symbol, position in self.positions.items():
            current_price = data.iloc[-1]['close']
            positions_value += position.size * current_price
        
        total_equity = self.capital + positions_value
        self.risk_manager.update_equity(total_equity)
    
    def _get_positions_value(self, bar: pd.Series) -> float:
        """Get total value of open positions."""
        value = 0
        for position in self.positions.values():
            value += position.size * bar['close']
        return value
    
    def _log_decision(self, decision: Decision) -> None:
        """Log a decision."""
        self.decisions.append(decision)
    
    def _save_log(self) -> None:
        """Save decision log to file."""
        if not self.log_file:
            return
        
        log_data = {
            'decisions': [d.to_dict() for d in self.decisions],
            'trades': [asdict(t) for t in self.trades],
            'final_capital': self.capital,
            'initial_capital': self.initial_capital,
        }
        
        with open(self.log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Saved decision log to {self.log_path}")
    
    def _reset(self) -> None:
        """Reset simulator state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.decisions = []
        self.equity_history = []
        self.feature_history = {}
        self.risk_manager.reset()
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])
    
    def get_decisions_df(self) -> pd.DataFrame:
        """Get decisions as DataFrame."""
        if not self.decisions:
            return pd.DataFrame()
        return pd.DataFrame([d.to_dict() for d in self.decisions])
