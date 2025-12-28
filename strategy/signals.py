"""
Signal Generation Module.

Converts model probabilities into trading signals.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class SignalResult:
    """Result of signal generation."""
    signal: Signal
    probability: float
    confidence: float
    reason: str
    
    def to_dict(self) -> dict:
        return {
            'signal': self.signal.value,
            'probability': self.probability,
            'confidence': self.confidence,
            'reason': self.reason,
        }


class SignalGenerator:
    """
    Generates trading signals from model probabilities.
    
    Applies thresholds and additional filters to convert
    raw probabilities into actionable signals.
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.65,
        exit_threshold: float = 0.35,
        min_confidence_gap: float = 0.1,
    ):
        """
        Initialize signal generator.
        
        Args:
            entry_threshold: Min probability for BUY signal
            exit_threshold: Max probability for SELL signal
            min_confidence_gap: Minimum gap from threshold for high confidence
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_confidence_gap = min_confidence_gap
    
    def generate(
        self,
        probability: float,
        current_position: Optional[str] = None,
    ) -> SignalResult:
        """
        Generate a trading signal from probability.
        
        Args:
            probability: Model output probability [0, 1]
            current_position: 'long', 'short', or None
            
        Returns:
            SignalResult with signal and reasoning
        """
        # Validate probability
        probability = np.clip(probability, 0, 1)
        
        # Calculate confidence (distance from neutral 0.5)
        confidence = abs(probability - 0.5) * 2
        
        # No position: look for entry
        if current_position is None:
            if probability >= self.entry_threshold:
                gap = probability - self.entry_threshold
                return SignalResult(
                    signal=Signal.BUY,
                    probability=probability,
                    confidence=gap / (1 - self.entry_threshold),
                    reason=f"Probability {probability:.3f} >= entry threshold {self.entry_threshold}"
                )
            else:
                return SignalResult(
                    signal=Signal.HOLD,
                    probability=probability,
                    confidence=confidence,
                    reason=f"Probability {probability:.3f} < entry threshold {self.entry_threshold}"
                )
        
        # Has long position: look for exit
        elif current_position == 'long':
            if probability <= self.exit_threshold:
                return SignalResult(
                    signal=Signal.SELL,
                    probability=probability,
                    confidence=(self.exit_threshold - probability) / self.exit_threshold,
                    reason=f"Probability {probability:.3f} <= exit threshold {self.exit_threshold}"
                )
            else:
                return SignalResult(
                    signal=Signal.HOLD,
                    probability=probability,
                    confidence=confidence,
                    reason=f"Probability {probability:.3f} > exit threshold, holding"
                )
        
        return SignalResult(
            signal=Signal.HOLD,
            probability=probability,
            confidence=confidence,
            reason="No action for current state"
        )


class EnsembleSignalGenerator:
    """
    Generates signals from multiple model predictions.
    
    Requires agreement between multiple models for higher confidence.
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.65,
        exit_threshold: float = 0.35,
        min_agreement: float = 0.6,
    ):
        """
        Initialize ensemble signal generator.
        
        Args:
            entry_threshold: Entry probability threshold
            exit_threshold: Exit probability threshold
            min_agreement: Minimum fraction of models that must agree
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_agreement = min_agreement
    
    def generate(
        self,
        probabilities: list,
        current_position: Optional[str] = None,
    ) -> SignalResult:
        """
        Generate signal from multiple model predictions.
        
        Args:
            probabilities: List of probabilities from different models
            current_position: Current position state
            
        Returns:
            SignalResult based on consensus
        """
        if not probabilities:
            return SignalResult(
                signal=Signal.HOLD,
                probability=0.5,
                confidence=0,
                reason="No predictions provided"
            )
        
        # Calculate consensus
        avg_prob = np.mean(probabilities)
        
        # Count agreements
        buy_votes = sum(1 for p in probabilities if p >= self.entry_threshold)
        sell_votes = sum(1 for p in probabilities if p <= self.exit_threshold)
        
        n_models = len(probabilities)
        buy_agreement = buy_votes / n_models
        sell_agreement = sell_votes / n_models
        
        # No position: look for entry
        if current_position is None:
            if buy_agreement >= self.min_agreement:
                return SignalResult(
                    signal=Signal.BUY,
                    probability=avg_prob,
                    confidence=buy_agreement,
                    reason=f"Ensemble BUY: {buy_votes}/{n_models} models agree, avg prob {avg_prob:.3f}"
                )
            else:
                return SignalResult(
                    signal=Signal.HOLD,
                    probability=avg_prob,
                    confidence=1 - buy_agreement,
                    reason=f"Ensemble HOLD: Only {buy_votes}/{n_models} models for BUY"
                )
        
        # Has position: look for exit
        elif current_position == 'long':
            if sell_agreement >= self.min_agreement:
                return SignalResult(
                    signal=Signal.SELL,
                    probability=avg_prob,
                    confidence=sell_agreement,
                    reason=f"Ensemble SELL: {sell_votes}/{n_models} models agree, avg prob {avg_prob:.3f}"
                )
            else:
                return SignalResult(
                    signal=Signal.HOLD,
                    probability=avg_prob,
                    confidence=1 - sell_agreement,
                    reason=f"Ensemble HOLD: Only {sell_votes}/{n_models} models for SELL"
                )
        
        return SignalResult(
            signal=Signal.HOLD,
            probability=avg_prob,
            confidence=0.5,
            reason="No action for current state"
        )
