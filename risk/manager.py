"""
Risk Management Module.

Implements risk controls including per-trade limits,
daily drawdown limits, and circuit breakers.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Current risk management state."""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown: float = 0.0
    circuit_breaker_triggered: bool = False
    circuit_breaker_reason: str = ""
    cooldown_remaining: int = 0
    violations: List[str] = field(default_factory=list)
    
    def reset_daily(self) -> None:
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.daily_trades = 0


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    allowed: bool
    reason: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'


class RiskManager:
    """
    Manages trading risk with multiple safety layers.
    
    Implements:
    1. Per-trade risk limits
    2. Daily drawdown limits
    3. Portfolio-level limits
    4. Circuit breaker
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = 0.02,
        max_daily_drawdown: float = 0.05,
        max_daily_loss: float = 0.03,
        max_portfolio_drawdown: float = 0.15,
        max_daily_trades: int = 20,
        max_position_pct: float = 0.25,
        max_correlation: float = 0.7,
        cooldown_periods: int = 10,
        initial_capital: float = 100000.0,
    ):
        """
        Initialize risk manager.
        
        Args:
            max_risk_per_trade: Maximum risk per trade (fraction of capital)
            max_daily_drawdown: Maximum daily drawdown before halt
            max_daily_loss: Maximum daily loss before halt
            max_portfolio_drawdown: Maximum total drawdown before halt
            max_daily_trades: Maximum trades per day
            max_position_pct: Maximum position size as fraction of capital
            max_correlation: Maximum correlation between positions
            cooldown_periods: Cooldown after circuit breaker
            initial_capital: Initial capital amount
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_daily_trades = max_daily_trades
        self.max_position_pct = max_position_pct
        self.max_correlation = max_correlation
        self.cooldown_periods = cooldown_periods
        self.initial_capital = initial_capital
        
        # State
        self.state = RiskState(
            peak_equity=initial_capital,
            current_equity=initial_capital,
        )
        
        self._current_date: Optional[date] = None
    
    def check_trade_allowed(
        self,
        position_size: float,
        price: float,
        stop_loss_pct: float,
        capital: float,
        current_date: Optional[date] = None,
    ) -> RiskCheckResult:
        """
        Check if a new trade is allowed under risk rules.
        
        Args:
            position_size: Proposed position size (shares)
            price: Entry price
            stop_loss_pct: Stop loss percentage
            capital: Current capital
            current_date: Current date for daily tracking
            
        Returns:
            RiskCheckResult indicating if trade is allowed
        """
        # Handle date change
        if current_date and current_date != self._current_date:
            self._on_new_day(current_date)
        
        # Check circuit breaker
        if self.state.circuit_breaker_triggered:
            if self.state.cooldown_remaining > 0:
                self.state.cooldown_remaining -= 1
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Circuit breaker active. Cooldown: {self.state.cooldown_remaining}",
                    risk_level="critical"
                )
            else:
                # Reset circuit breaker
                self.state.circuit_breaker_triggered = False
                self.state.circuit_breaker_reason = ""
                logger.info("Circuit breaker reset")
        
        # Check daily trade limit
        if self.state.daily_trades >= self.max_daily_trades:
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily trade limit reached: {self.state.daily_trades}/{self.max_daily_trades}",
                risk_level="high"
            )
        
        # Check daily loss limit
        daily_loss_pct = abs(min(0, self.state.daily_pnl)) / capital
        if daily_loss_pct >= self.max_daily_loss:
            self._trigger_circuit_breaker("Daily loss limit exceeded")
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily loss {daily_loss_pct:.2%} >= limit {self.max_daily_loss:.2%}",
                risk_level="critical"
            )
        
        # Check portfolio drawdown
        if self.state.current_drawdown >= self.max_portfolio_drawdown:
            self._trigger_circuit_breaker("Portfolio drawdown limit exceeded")
            return RiskCheckResult(
                allowed=False,
                reason=f"Drawdown {self.state.current_drawdown:.2%} >= limit {self.max_portfolio_drawdown:.2%}",
                risk_level="critical"
            )
        
        # Check per-trade risk
        trade_value = position_size * price
        trade_risk = trade_value * stop_loss_pct
        risk_pct = trade_risk / capital
        
        if risk_pct > self.max_risk_per_trade:
            return RiskCheckResult(
                allowed=False,
                reason=f"Trade risk {risk_pct:.2%} > max {self.max_risk_per_trade:.2%}",
                risk_level="high"
            )
        
        # Check position size
        position_pct = trade_value / capital
        if position_pct > self.max_position_pct:
            return RiskCheckResult(
                allowed=False,
                reason=f"Position size {position_pct:.2%} > max {self.max_position_pct:.2%}",
                risk_level="medium"
            )
        
        # Determine risk level
        if risk_pct > self.max_risk_per_trade * 0.8:
            risk_level = "medium"
        elif daily_loss_pct > self.max_daily_loss * 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return RiskCheckResult(
            allowed=True,
            reason="All risk checks passed",
            risk_level=risk_level
        )
    
    def update_pnl(self, pnl: float) -> None:
        """
        Update PnL tracking.
        
        Args:
            pnl: PnL from closed trade
        """
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        
        # Update equity
        self.state.current_equity += pnl
        
        # Update peak and drawdown
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity
        
        self.state.current_drawdown = (
            (self.state.peak_equity - self.state.current_equity) /
            self.state.peak_equity
        )
        
        # Check if we need to trigger circuit breaker
        if self.state.current_drawdown >= self.max_portfolio_drawdown:
            self._trigger_circuit_breaker("Portfolio drawdown limit exceeded")
    
    def update_equity(self, equity: float) -> None:
        """
        Update current equity value.
        
        Args:
            equity: Current portfolio equity
        """
        self.state.current_equity = equity
        
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        
        self.state.current_drawdown = (
            (self.state.peak_equity - equity) / self.state.peak_equity
        )
    
    def is_circuit_breaker_triggered(self) -> bool:
        """Check if circuit breaker is active."""
        return self.state.circuit_breaker_triggered
    
    def get_risk_status(self) -> Dict:
        """Get current risk status."""
        return {
            'daily_pnl': self.state.daily_pnl,
            'daily_trades': self.state.daily_trades,
            'current_drawdown': self.state.current_drawdown,
            'peak_equity': self.state.peak_equity,
            'current_equity': self.state.current_equity,
            'circuit_breaker': self.state.circuit_breaker_triggered,
            'circuit_breaker_reason': self.state.circuit_breaker_reason,
            'violations': self.state.violations,
        }
    
    def get_max_position_size(
        self,
        price: float,
        stop_loss_pct: float,
        capital: float,
    ) -> float:
        """
        Calculate maximum allowed position size.
        
        Args:
            price: Entry price
            stop_loss_pct: Stop loss percentage
            capital: Current capital
            
        Returns:
            Maximum position size in shares
        """
        # Based on max risk per trade
        max_risk_dollars = capital * self.max_risk_per_trade
        risk_per_share = price * stop_loss_pct
        max_shares_risk = max_risk_dollars / risk_per_share if risk_per_share > 0 else 0
        
        # Based on max position percentage
        max_dollars_position = capital * self.max_position_pct
        max_shares_position = max_dollars_position / price
        
        return min(max_shares_risk, max_shares_position)
    
    def _on_new_day(self, new_date: date) -> None:
        """Handle transition to new trading day."""
        logger.debug(f"New trading day: {new_date}")
        self._current_date = new_date
        self.state.reset_daily()
    
    def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger the circuit breaker."""
        self.state.circuit_breaker_triggered = True
        self.state.circuit_breaker_reason = reason
        self.state.cooldown_remaining = self.cooldown_periods
        self.state.violations.append(f"{datetime.now()}: {reason}")
        
        logger.warning(f"CIRCUIT BREAKER TRIGGERED: {reason}")
    
    def reset(self) -> None:
        """Reset risk manager state."""
        self.state = RiskState(
            peak_equity=self.initial_capital,
            current_equity=self.initial_capital,
        )
        self._current_date = None
        logger.info("Risk manager reset")
