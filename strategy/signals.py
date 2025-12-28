"""
Trading Strategy Signals.

Converts model probabilities into actionable trading signals (BUY/SELL/HOLD).
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class SignalResult:
    signal: SignalType
    confidence: float
    reason: str
    metadata: Dict[str, Any] = None


class SignalGenerator:
    """
    Generates trading signals from probability predictions.
    
    Features:
    - Configurable thresholds
    - Dynamic thresholds based on volatility
    - Trend filtering
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.65,
        exit_threshold: float = 0.35,
        use_dynamic_threshold: bool = False,
        min_volatility: float = 0.0,
        max_volatility: float = 1.0,
    ):
        """
        Initialize signal generator.
        
        Args:
            entry_threshold: Base probability threshold for BUY
            exit_threshold: Base probability threshold for SELL
            use_dynamic_threshold: Adjust thresholds based on volatility
            min_volatility: Min volatility to trade
            max_volatility: Max volatility to trade
        """
        self.base_entry_threshold = entry_threshold
        self.base_exit_threshold = exit_threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        
    def generate(
        self,
        probability: float,
        current_position: Optional[str] = None,
        volatility: Optional[float] = None,
        trend_strength: Optional[float] = None,
    ) -> SignalResult:
        """
        Generate signal.
        
        Args:
            probability: distinct model probability (0-1)
            current_position: 'long' or None
            volatility: current volatility (e.g. normalized ATR or std dev)
            trend_strength: trend indicator (0-1)
            
        Returns:
            SignalResult
        """
        # Determine effective thresholds
        entry_threshold = self.base_entry_threshold
        exit_threshold = self.base_exit_threshold
        
        meta = {
            "base_entry": self.base_entry_threshold,
            "effective_entry": entry_threshold,
            "volatility": volatility
        }
        
        # Dynamic Threshold Adjustment
        if self.use_dynamic_threshold and volatility is not None:
            # If volatility is high, require higher confidence to enter
            # If volatility is low, we can be more aggressive
            
            # Simple scaling: volatility 0.5 -> threshold + 0.05
            # volatility 0.1 -> threshold - 0.05
            adjustment = (volatility - 0.3) * 0.2  # Assumes vol centered around 0.3
            entry_threshold = np.clip(entry_threshold + adjustment, 0.55, 0.90)
            meta["effective_entry"] = entry_threshold
            meta["threshold_adj"] = adjustment
            
            # Check volatility bounds
            if volatility < self.min_volatility:
                return SignalResult(SignalType.HOLD, 0.0, "Volatility too low", meta)
            if volatility > self.max_volatility:
                return SignalResult(SignalType.HOLD, 0.0, "Volatility too high", meta)
        
        # Generate Signal
        if probability >= entry_threshold:
            if current_position != 'long':
                return SignalResult(
                    SignalType.BUY,
                    probability,
                    f"Strong bullish signal ({probability:.1%}) > {entry_threshold:.1%}",
                    meta
                )
            else:
                return SignalResult(SignalType.HOLD, probability, "Already long", meta)
                
        elif probability <= exit_threshold:
            if current_position == 'long':
                return SignalResult(
                    SignalType.SELL,
                    probability,
                    f"Bearish signal ({probability:.1%}) < {exit_threshold:.1%}",
                    meta
                )
            else:
                return SignalResult(SignalType.HOLD, probability, "Bearish but no position", meta)
        
        return SignalResult(SignalType.HOLD, probability, "Indecisive signal", meta)


class TrailingStop:
    """
    Trailing Stop Loss Logic.
    
    Tracks high-water mark of price and signals exit if price drops
    by a certain percentage or ATR multiple.
    """
    
    def __init__(
        self,
        pct_stop: float = 0.05,  # 5% trailing stop
        use_atr: bool = False,
        atr_multiplier: float = 3.0,
    ):
        self.pct_stop = pct_stop
        self.use_atr = use_atr
        self.atr_multiplier = atr_multiplier
        
        # State
        self.high_water_mark = -float('inf')
        self.entry_price = 0.0
        self.active = False
        self.stop_price = 0.0
        
    def reset(self):
        """Reset state."""
        self.high_water_mark = -float('inf')
        self.entry_price = 0.0
        self.active = False
        self.stop_price = 0.0
        
    def update(
        self,
        current_price: float,
        atr: float = 0.0
    ) -> bool:
        """
        Update with current price.
        
        Returns:
            True if stop loss is triggered.
        """
        if not self.active:
            # Activate on first update (entry)
            self.active = True
            self.entry_price = current_price
            self.high_water_mark = current_price
            self._update_stop_price(current_price, atr)
            return False
        
        # Update high water mark
        if current_price > self.high_water_mark:
            self.high_water_mark = current_price
            self._update_stop_price(current_price, atr)
            
        # Check stop
        if current_price < self.stop_price:
            return True
            
        return False
        
    def _update_stop_price(self, price: float, atr: float):
        """Calculate new stop price."""
        if self.use_atr and atr > 0:
            dist = atr * self.atr_multiplier
            self.stop_price = price - dist
        else:
            dist = price * self.pct_stop
            self.stop_price = price - dist
