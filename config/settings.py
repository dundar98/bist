"""
Configuration settings for the BIST100 Deep Learning Trading System.

All configurable parameters are centralized here. No magic numbers in code.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date
import torch


@dataclass
class DataConfig:
    """Data loading and validation settings."""
    
    # Default symbols for training (subset of BIST100)
    default_symbols: List[str] = field(default_factory=lambda: [
        "THYAO", "GARAN", "AKBNK", "EREGL", "KCHOL",
        "SISE", "TUPRS", "SAHOL", "TCELL", "BIMAS"
    ])
    
    # Date range
    start_date: date = date(2018, 1, 1)
    end_date: date = date(2024, 1, 1)
    
    # Data source
    data_source: str = "yfinance"  # "yfinance" or "csv"
    csv_data_dir: Optional[str] = None
    
    # Yahoo Finance suffix for BIST stocks
    yfinance_suffix: str = ".IS"


@dataclass
class FeatureConfig:
    """Feature engineering settings."""
    
    # Lookback windows
    lookback_window: int = 60  # Sequence length for LSTM
    
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    volatility_window: int = 20
    sma_short: int = 10
    sma_long: int = 50
    
    # Normalization
    normalization_method: str = "rolling"  # "rolling" or "training"
    rolling_window: int = 60


@dataclass
class LabelConfig:
    """Labeling configuration for supervised learning."""
    
    # Target definition: Label = 1 if price increases by at least X% within N bars
    threshold_pct: float = 0.02  # 2% minimum return
    horizon_bars: int = 5  # Look ahead N bars
    
    # For multi-class (optional)
    use_multiclass: bool = False
    sell_threshold_pct: float = -0.02  # For 3-class: buy/hold/sell


@dataclass
class ModelConfig:
    """Neural network architecture settings."""
    
    # Model type
    model_type: str = "multitask"  # "lstm", "gru", "cnn", "transformer", "multitask"
    
    # Architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False  # Must be False for causal prediction
    
    # Input/output
    # input_size is computed dynamically based on features
    output_size: int = 1  # Binary classification


@dataclass
class TrainingConfig:
    """Training loop settings."""
    
    # Optimization
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    
    # Early stopping
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Class balance
    use_class_weights: bool = True
    
    # Walk-forward settings
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    n_splits: int = 1  # Number of walk-forward folds
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    random_seed: int = 42


@dataclass
class BacktestConfig:
    """Backtesting engine settings."""
    
    # Entry/exit rules
    entry_threshold: float = 0.60  # Probability threshold to enter
    exit_threshold: float = 0.35  # Probability threshold to exit (optional)
    
    # Stop loss / take profit
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    
    # Position sizing
    position_size_method: str = "fixed_fraction"  # "fixed_fraction" or "kelly"
    fixed_fraction: float = 0.1  # 10% of capital per trade
    max_positions: int = 5  # Maximum concurrent positions
    
    # Initial capital
    initial_capital: float = 100000.0
    
    # Transaction costs
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.001  # 0.1% slippage


@dataclass
class RiskConfig:
    """Risk management settings."""
    
    # Per-trade limits
    max_risk_per_trade: float = 0.02  # 2% of capital
    
    # Daily limits
    max_daily_drawdown: float = 0.05  # 5% daily drawdown triggers halt
    max_daily_loss: float = 0.03  # 3% daily loss triggers halt
    
    # Portfolio limits
    max_portfolio_drawdown: float = 0.15  # 15% total drawdown triggers halt
    max_correlation: float = 0.7  # Max correlation between positions
    
    # Circuit breaker
    enable_circuit_breaker: bool = True
    cooldown_periods: int = 10  # Bars to wait after circuit breaker


@dataclass
class PaperTradingConfig:
    """Paper trading simulation settings."""
    
    # Simulation mode
    use_realistic_fills: bool = True
    fill_delay_bars: int = 1  # 1-bar delay for fills
    
    # Logging
    log_all_decisions: bool = True
    log_file: str = "paper_trading_log.json"


@dataclass
class SystemConfig:
    """Master configuration container."""
    
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)


# Global default configuration
DEFAULT_CONFIG = SystemConfig()


def get_config() -> SystemConfig:
    """Get the default system configuration."""
    return SystemConfig()
