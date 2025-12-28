# Deep Learning Trading System for BIST100

A production-quality end-to-end trading system that trains deep learning models on BIST100 stock data, generates probabilistic trading signals, and includes backtesting with risk management.

## Features

- **BIST100 Constraint**: Strictly validates all data against BIST100 universe
- **Feature Engineering**: RSI, MACD, ATR, volatility, trend indicators
- **Deep Learning**: LSTM/GRU with probability output
- **Walk-Forward Training**: Time-series aware splits, no future leakage
- **Backtesting**: Complete simulation with position sizing, stop loss, take profit
- **Risk Management**: Per-trade limits, daily drawdown, circuit breaker
- **Paper Trading**: Bar-by-bar simulation with decision logging

## Installation

```bash
cd bist
pip install -r requirements.txt
```

## Quick Start

```bash
# Run full pipeline with demo data
python scripts/run_pipeline.py --mode demo

# Run with real data
python scripts/run_pipeline.py --symbols THYAO GARAN AKBNK --start 2020-01-01 --end 2024-01-01
```

## Project Structure

```
bist/
├── config/          # Configuration settings
├── data/            # Data loading and validation
├── models/          # Neural network architectures  
├── training/        # Training pipeline
├── backtest/        # Backtesting engine
├── strategy/        # Trading strategy logic
├── risk/            # Risk management
├── paper_trading/   # Paper trading simulation
├── utils/           # Utilities
├── tests/           # Unit tests
└── scripts/         # Execution scripts
```

## Configuration

Edit `config/settings.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LABEL_THRESHOLD` | 0.02 | Target return (2%) for positive label |
| `LABEL_HORIZON` | 5 | Bars to look ahead for labeling |
| `ENTRY_THRESHOLD` | 0.65 | Probability threshold to enter trade |
| `STOP_LOSS` | 0.03 | Stop loss percentage (3%) |
| `TAKE_PROFIT` | 0.06 | Take profit percentage (6%) |
| `MAX_RISK_PER_TRADE` | 0.02 | Maximum 2% risk per trade |

## Disclaimer

This system is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss.
