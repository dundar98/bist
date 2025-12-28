"""
Position Sizing Module.

Implements various position sizing strategies.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing result."""
    size: float  # Number of shares/units
    capital_used: float  # Dollar amount
    risk_amount: float  # Dollar amount at risk
    sizing_method: str
    reason: str


class PositionSizer(ABC):
    """Abstract base for position sizing strategies."""
    
    @abstractmethod
    def calculate(
        self,
        capital: float,
        price: float,
        stop_loss_pct: float,
        probability: float,
        volatility: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate position size.
        
        Args:
            capital: Available capital
            price: Entry price
            stop_loss_pct: Stop loss percentage
            probability: Signal probability
            volatility: Optional volatility measure
            
        Returns:
            PositionSize object
        """
        pass


class FixedFractionSizer(PositionSizer):
    """
    Fixed fraction of capital per trade.
    
    Simple and safe approach - always uses same percentage.
    """
    
    def __init__(
        self,
        fraction: float = 0.1,
        max_fraction: float = 0.25,
    ):
        """
        Initialize fixed fraction sizer.
        
        Args:
            fraction: Fraction of capital per trade
            max_fraction: Maximum fraction regardless of other factors
        """
        self.fraction = fraction
        self.max_fraction = max_fraction
    
    def calculate(
        self,
        capital: float,
        price: float,
        stop_loss_pct: float,
        probability: float,
        volatility: Optional[float] = None,
    ) -> PositionSize:
        """Calculate position using fixed fraction."""
        capital_used = capital * min(self.fraction, self.max_fraction)
        size = capital_used / price
        risk_amount = capital_used * stop_loss_pct
        
        return PositionSize(
            size=size,
            capital_used=capital_used,
            risk_amount=risk_amount,
            sizing_method="fixed_fraction",
            reason=f"Using {self.fraction:.1%} of capital"
        )


class RiskBasedSizer(PositionSizer):
    """
    Risk-based position sizing.
    
    Sizes position so that stop loss represents a fixed % of capital.
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.02,
        max_fraction: float = 0.25,
    ):
        """
        Initialize risk-based sizer.
        
        Args:
            risk_per_trade: Maximum risk per trade as fraction of capital
            max_fraction: Maximum capital fraction per trade
        """
        self.risk_per_trade = risk_per_trade
        self.max_fraction = max_fraction
    
    def calculate(
        self,
        capital: float,
        price: float,
        stop_loss_pct: float,
        probability: float,
        volatility: Optional[float] = None,
    ) -> PositionSize:
        """Calculate position based on risk."""
        if stop_loss_pct <= 0:
            stop_loss_pct = 0.01  # Default 1% stop
        
        # Maximum dollar risk
        max_risk = capital * self.risk_per_trade
        
        # Position size to achieve target risk
        risk_per_share = price * stop_loss_pct
        size = max_risk / risk_per_share
        capital_used = size * price
        
        # Apply maximum fraction constraint
        max_capital = capital * self.max_fraction
        if capital_used > max_capital:
            capital_used = max_capital
            size = capital_used / price
        
        actual_risk = size * risk_per_share
        
        return PositionSize(
            size=size,
            capital_used=capital_used,
            risk_amount=actual_risk,
            sizing_method="risk_based",
            reason=f"Targeting {self.risk_per_trade:.1%} risk per trade"
        )


class KellySizer(PositionSizer):
    """
    Kelly Criterion position sizing.
    
    Optimal sizing based on edge and probability.
    Uses fractional Kelly for safety.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_fraction: float = 0.25,
        min_edge: float = 0.0,
    ):
        """
        Initialize Kelly sizer.
        
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
            max_fraction: Maximum position size
            min_edge: Minimum required edge to take position
        """
        self.kelly_fraction = kelly_fraction
        self.max_fraction = max_fraction
        self.min_edge = min_edge
    
    def calculate(
        self,
        capital: float,
        price: float,
        stop_loss_pct: float,
        probability: float,
        volatility: Optional[float] = None,
    ) -> PositionSize:
        """Calculate position using Kelly Criterion."""
        # Estimate win/loss ratio from stop loss
        # Assuming symmetric take profit
        take_profit_pct = stop_loss_pct * 2  # 2:1 reward to risk
        
        win_prob = probability
        loss_prob = 1 - probability
        
        # Calculate Kelly fraction
        # f* = (bp - q) / b
        # where b = win/loss ratio, p = win prob, q = loss prob
        b = take_profit_pct / stop_loss_pct
        
        kelly = (b * win_prob - loss_prob) / b
        
        # Check minimum edge
        if kelly < self.min_edge:
            return PositionSize(
                size=0,
                capital_used=0,
                risk_amount=0,
                sizing_method="kelly",
                reason=f"Kelly {kelly:.3f} below minimum edge {self.min_edge}"
            )
        
        # Apply fractional Kelly and cap
        fraction = min(kelly * self.kelly_fraction, self.max_fraction)
        fraction = max(0, fraction)  # No negative sizing
        
        capital_used = capital * fraction
        size = capital_used / price
        risk_amount = capital_used * stop_loss_pct
        
        return PositionSize(
            size=size,
            capital_used=capital_used,
            risk_amount=risk_amount,
            sizing_method="kelly",
            reason=f"Kelly={kelly:.3f}, using {fraction:.1%} of capital"
        )


class VolatilityAdjustedSizer(PositionSizer):
    """
    Volatility-adjusted position sizing.
    
    Larger positions in low-volatility markets, smaller in high-volatility.
    """
    
    def __init__(
        self,
        base_fraction: float = 0.1,
        target_volatility: float = 0.02,
        max_fraction: float = 0.25,
    ):
        """
        Initialize volatility-adjusted sizer.
        
        Args:
            base_fraction: Base position size fraction
            target_volatility: Target daily volatility
            max_fraction: Maximum position fraction
        """
        self.base_fraction = base_fraction
        self.target_volatility = target_volatility
        self.max_fraction = max_fraction
    
    def calculate(
        self,
        capital: float,
        price: float,
        stop_loss_pct: float,
        probability: float,
        volatility: Optional[float] = None,
    ) -> PositionSize:
        """Calculate position adjusted for volatility."""
        if volatility is None or volatility <= 0:
            volatility = self.target_volatility
        
        # Scale position inversely with volatility
        vol_scalar = self.target_volatility / volatility
        fraction = self.base_fraction * vol_scalar
        
        # Apply cap
        fraction = min(fraction, self.max_fraction)
        
        capital_used = capital * fraction
        size = capital_used / price
        risk_amount = capital_used * stop_loss_pct
        
        return PositionSize(
            size=size,
            capital_used=capital_used,
            risk_amount=risk_amount,
            sizing_method="volatility_adjusted",
            reason=f"Volatility {volatility:.3f}, scaling by {vol_scalar:.2f}x"
        )


def get_position_sizer(
    method: str = "fixed_fraction",
    **kwargs
) -> PositionSizer:
    """
    Factory function for position sizers.
    
    Args:
        method: Sizing method name
        **kwargs: Method-specific parameters
        
    Returns:
        PositionSizer instance
    """
    sizers = {
        "fixed_fraction": FixedFractionSizer,
        "risk_based": RiskBasedSizer,
        "kelly": KellySizer,
        "volatility_adjusted": VolatilityAdjustedSizer,
    }
    
    if method not in sizers:
        raise ValueError(f"Unknown sizing method: {method}. Available: {list(sizers.keys())}")
    
    return sizers[method](**kwargs)
