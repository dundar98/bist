"""
Portfolio Optimization.

Implements allocation strategies:
1. Equal Weight (Baseline)
2. Mean-Variance Optimization (Markowitz)
3. Risk Parity
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Optimizes portfolio weights.
    """
    
    def __init__(self, method: str = 'equal_weight'):
        """
        Args:
            method: 'equal_weight', 'max_sharpe', 'min_volatility', 'risk_parity'
        """
        self.method = method
    
    def optimize(
        self,
        returns: pd.DataFrame,
        probabilities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate optimal weights.
        
        Args:
            returns: DataFrame of historical returns (columns=symbols)
            probabilities: Optional dictionary of model probabilities (for Kelly/Confidence weighting)
            
        Returns:
            Dictionary of symbol -> weight (sum=1.0)
        """
        symbols = returns.columns.tolist()
        n = len(symbols)
        
        if n == 0:
            return {}
            
        if self.method == 'equal_weight':
            return {s: 1.0 / n for s in symbols}
            
        elif self.method == 'confidence_weighted':
            if probabilities is None:
                logger.warning("Probabilities required for confidence weighting. Defaulting to equal.")
                return {s: 1.0 / n for s in symbols}
            
            # Weight = Probability - 0.5 (scaled)
            # Only allocate to > 0.5
            scores = {s: max(0, probabilities.get(s, 0.5) - 0.5) for s in symbols}
            total_score = sum(scores.values())
            
            if total_score == 0:
                return {s: 0.0 for s in symbols}
                
            return {s: score / total_score for s in symbols}
            
        elif self.method == 'max_sharpe':
            return self._optimize_mean_variance(returns, objective='sharpe')
            
        elif self.method == 'min_volatility':
            return self._optimize_mean_variance(returns, objective='volatility')
            
        elif self.method == 'risk_parity':
            return self._risk_parity(returns)
            
        else:
            logger.warning(f"Unknown method {self.method}. Defaulting to equal.")
            return {s: 1.0 / n for s in symbols}
            
    def _optimize_mean_variance(
        self,
        returns: pd.DataFrame,
        objective: str = 'sharpe',
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """Mean-Variance Optimization."""
        mu = returns.mean() * 252  # Annualized returns
        cov = returns.cov() * 252  # Annualized covariance
        n = len(mu)
        
        def neg_sharpe(w):
            r = np.sum(mu * w)
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return -(r - risk_free_rate) / vol
            
        def volatility(w):
            return np.sqrt(np.dot(w.T, np.dot(cov, w)))
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(n))
        initial_w = np.array([1/n] * n)
        
        target_fun = neg_sharpe if objective == 'sharpe' else volatility
        
        try:
            result = minimize(
                target_fun,
                initial_w,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            weights = result.x
            # Clean up small weights
            weights[weights < 0.01] = 0.0
            weights = weights / np.sum(weights)
            
            return {k: float(v) for k, v in zip(returns.columns, weights)}
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {s: 1.0 / n for s in returns.columns}
    
    def _risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Risk Parity (Equal Risk Contribution).
        
        Allocates weights such that each asset contributes equally to portfolio risk.
        Simplified: Weight ~ 1 / volatility
        """
        vol = returns.std()
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        
        return {k: float(v) for k, v in zip(returns.columns, weights)}
