"""
Trading Rules Module.

Defines entry and exit rules for the trading strategy.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result of a rule evaluation."""
    triggered: bool
    rule_name: str
    reason: str
    value: Optional[float] = None


class TradingRule(ABC):
    """Abstract base for trading rules."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name."""
        pass
    
    @abstractmethod
    def evaluate(self, **kwargs) -> RuleResult:
        """Evaluate the rule."""
        pass


class ProbabilityThresholdRule(TradingRule):
    """Entry/exit based on probability threshold."""
    
    def __init__(self, threshold: float, direction: str = "above"):
        """
        Initialize threshold rule.
        
        Args:
            threshold: Probability threshold
            direction: 'above' or 'below'
        """
        self.threshold = threshold
        self.direction = direction
    
    @property
    def name(self) -> str:
        return f"probability_{self.direction}_{self.threshold}"
    
    def evaluate(self, probability: float, **kwargs) -> RuleResult:
        """Check if probability crosses threshold."""
        if self.direction == "above":
            triggered = probability >= self.threshold
        else:
            triggered = probability <= self.threshold
        
        return RuleResult(
            triggered=triggered,
            rule_name=self.name,
            reason=f"Probability {probability:.3f} {'≥' if self.direction == 'above' else '≤'} {self.threshold}",
            value=probability
        )


class StopLossRule(TradingRule):
    """Stop loss exit rule."""
    
    def __init__(self, stop_loss_pct: float):
        """
        Initialize stop loss rule.
        
        Args:
            stop_loss_pct: Stop loss percentage from entry
        """
        self.stop_loss_pct = stop_loss_pct
    
    @property
    def name(self) -> str:
        return f"stop_loss_{self.stop_loss_pct:.1%}"
    
    def evaluate(
        self,
        current_price: float,
        entry_price: float,
        **kwargs
    ) -> RuleResult:
        """Check if stop loss is hit."""
        stop_price = entry_price * (1 - self.stop_loss_pct)
        triggered = current_price <= stop_price
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        return RuleResult(
            triggered=triggered,
            rule_name=self.name,
            reason=f"Current {current_price:.2f} {'≤' if triggered else '>'} stop {stop_price:.2f}",
            value=pnl_pct
        )


class TakeProfitRule(TradingRule):
    """Take profit exit rule."""
    
    def __init__(self, take_profit_pct: float):
        """
        Initialize take profit rule.
        
        Args:
            take_profit_pct: Take profit percentage from entry
        """
        self.take_profit_pct = take_profit_pct
    
    @property
    def name(self) -> str:
        return f"take_profit_{self.take_profit_pct:.1%}"
    
    def evaluate(
        self,
        current_price: float,
        entry_price: float,
        **kwargs
    ) -> RuleResult:
        """Check if take profit is hit."""
        tp_price = entry_price * (1 + self.take_profit_pct)
        triggered = current_price >= tp_price
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        return RuleResult(
            triggered=triggered,
            rule_name=self.name,
            reason=f"Current {current_price:.2f} {'≥' if triggered else '<'} target {tp_price:.2f}",
            value=pnl_pct
        )


class TrailingStopRule(TradingRule):
    """Trailing stop exit rule."""
    
    def __init__(self, trail_pct: float):
        """
        Initialize trailing stop rule.
        
        Args:
            trail_pct: Trailing stop percentage from high
        """
        self.trail_pct = trail_pct
    
    @property
    def name(self) -> str:
        return f"trailing_stop_{self.trail_pct:.1%}"
    
    def evaluate(
        self,
        current_price: float,
        highest_price: float,
        **kwargs
    ) -> RuleResult:
        """Check if trailing stop is hit."""
        trail_price = highest_price * (1 - self.trail_pct)
        triggered = current_price <= trail_price
        
        drop_pct = (highest_price - current_price) / highest_price
        
        return RuleResult(
            triggered=triggered,
            rule_name=self.name,
            reason=f"Dropped {drop_pct:.1%} from high {highest_price:.2f}",
            value=drop_pct
        )


class TimeBasedExitRule(TradingRule):
    """Exit after maximum bars held."""
    
    def __init__(self, max_bars: int):
        """
        Initialize time-based exit rule.
        
        Args:
            max_bars: Maximum bars to hold position
        """
        self.max_bars = max_bars
    
    @property
    def name(self) -> str:
        return f"time_exit_{self.max_bars}_bars"
    
    def evaluate(self, bars_held: int, **kwargs) -> RuleResult:
        """Check if max hold time exceeded."""
        triggered = bars_held >= self.max_bars
        
        return RuleResult(
            triggered=triggered,
            rule_name=self.name,
            reason=f"Held {bars_held} bars {'≥' if triggered else '<'} max {self.max_bars}",
            value=bars_held
        )


class RuleEngine:
    """
    Evaluates multiple trading rules.
    
    Combines entry and exit rules into a cohesive system.
    """
    
    def __init__(
        self,
        entry_rules: Optional[List[TradingRule]] = None,
        exit_rules: Optional[List[TradingRule]] = None,
        require_all_entry: bool = True,
        require_any_exit: bool = True,
    ):
        """
        Initialize rule engine.
        
        Args:
            entry_rules: List of entry rules
            exit_rules: List of exit rules
            require_all_entry: If True, ALL entry rules must pass
            require_any_exit: If True, ANY exit rule triggers exit
        """
        self.entry_rules = entry_rules or []
        self.exit_rules = exit_rules or []
        self.require_all_entry = require_all_entry
        self.require_any_exit = require_any_exit
    
    def check_entry(self, **kwargs) -> tuple:
        """
        Check if entry conditions are met.
        
        Returns:
            Tuple of (should_enter, list of rule results)
        """
        results = []
        for rule in self.entry_rules:
            try:
                result = rule.evaluate(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {rule.name}: {e}")
        
        if not results:
            return False, results
        
        if self.require_all_entry:
            should_enter = all(r.triggered for r in results)
        else:
            should_enter = any(r.triggered for r in results)
        
        return should_enter, results
    
    def check_exit(self, **kwargs) -> tuple:
        """
        Check if exit conditions are met.
        
        Returns:
            Tuple of (should_exit, triggered rule result or None)
        """
        for rule in self.exit_rules:
            try:
                result = rule.evaluate(**kwargs)
                if result.triggered:
                    return True, result
            except Exception as e:
                logger.error(f"Error evaluating {rule.name}: {e}")
        
        return False, None


def create_default_rules(
    entry_threshold: float = 0.65,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    max_hold_bars: int = 20,
) -> RuleEngine:
    """
    Create a default rule engine.
    
    Args:
        entry_threshold: Probability threshold for entry
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        max_hold_bars: Maximum bars to hold
        
    Returns:
        Configured RuleEngine
    """
    entry_rules = [
        ProbabilityThresholdRule(entry_threshold, "above"),
    ]
    
    exit_rules = [
        StopLossRule(stop_loss_pct),
        TakeProfitRule(take_profit_pct),
        TimeBasedExitRule(max_hold_bars),
    ]
    
    return RuleEngine(
        entry_rules=entry_rules,
        exit_rules=exit_rules,
        require_all_entry=True,
        require_any_exit=True,
    )
