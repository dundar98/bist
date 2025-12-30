#!/usr/bin/env python3
"""
Daily Signal Scanner.

Scans all BIST100 stocks, generates predictions, and identifies
trading opportunities based on model signals.
"""

import logging
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import BIST100Validator, get_data_loader, prepare_features
from models import BaseModel
from strategy.signals import SignalGenerator, SignalType, SignalResult
from analysis import NewsSentimentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class StockSignal:
    """Trading signal for a single stock."""
    symbol: str
    probability: float
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    current_price: float
    change_1d: float  # 1-day price change %
    rsi: float
    volatility: float
    rsi: float
    volatility: float
    sentiment_score: float = 0.0 # New
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def priority(self) -> int:
        """Higher priority = more actionable signal."""
        if self.signal == SignalType.BUY.value and self.probability > 0.75:
            return 3
        elif self.signal == SignalType.BUY.value:
            return 2
        elif self.signal == SignalType.SELL.value:
            return 1
        return 0
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'probability': self.probability,
            'signal': self.signal,
            'confidence': self.confidence,
            'current_price': self.current_price,
            'sentiment_score': self.sentiment_score, # New
            'change_1d': self.change_1d,
            'rsi': self.rsi,
            'volatility': self.volatility,
            'reason': self.reason,
            'timestamp': str(self.timestamp),
        }


@dataclass
class DailyScanResult:
    """Result of daily market scan."""
    scan_date: date
    buy_signals: List[StockSignal]
    sell_signals: List[StockSignal]
    hold_signals: List[StockSignal]
    errors: List[str]
    scan_duration: float
    
    @property
    def total_scanned(self) -> int:
        return len(self.buy_signals) + len(self.sell_signals) + len(self.hold_signals)
    
    def get_top_signals(self, n: int = 5) -> List[StockSignal]:
        """Get top N buy signals by probability."""
        return sorted(self.buy_signals, key=lambda x: x.probability, reverse=True)[:n]
    
    def to_summary_dict(self) -> dict:
        return {
            'scan_date': str(self.scan_date),
            'total_scanned': self.total_scanned,
            'buy_count': len(self.buy_signals),
            'sell_count': len(self.sell_signals),
            'hold_count': len(self.hold_signals),
            'error_count': len(self.errors),
            'scan_duration_seconds': self.scan_duration,
        }


class DailyScanner:
    """
    Scans BIST100 stocks daily and generates trading signals.
    """
    
    def __init__(
        self,
        model: BaseModel,
        feature_columns: List[str],
        lookback: int = 60,
        entry_threshold: float = 0.65,
        exit_threshold: float = 0.35,
        data_source: str = "yfinance",
        device: str = "cpu",
    ):
        """
        Initialize daily scanner.
        
        Args:
            model: Trained model for predictions
            feature_columns: Feature column names
            lookback: Sequence length for model
            entry_threshold: Buy signal threshold
            exit_threshold: Sell signal threshold
            data_source: Data source ('yfinance' or 'synthetic')
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_columns = feature_columns
        self.lookback = lookback
        self.device = device
        
        self.signal_generator = SignalGenerator(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
        
        self.sentiment_analyzer = NewsSentimentAnalyzer() # New
        
        self.validator = BIST100Validator()
        self.data_loader = get_data_loader(data_source)
    
    def scan_all(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 180,  # Increased for more robust data
        limit: Optional[int] = 30,
    ) -> DailyScanResult:
        """
        Scan all BIST100 stocks.
        
        Args:
            symbols: Optional list of symbols (default: all BIST100)
            lookback_days: Days of history to load
            limit: Max number of stocks to scan (None for all)
            
        Returns:
            DailyScanResult with all signals
        """
        import time
        start_time = time.time()
        
        # Get symbols
        if symbols is None:
            # Get all valid BIST100 symbols
            symbols = self.validator.get_all_symbols()
            
            # Apply limit if specified
            if limit is not None:
                symbols = symbols[:limit]
        
        # Validate symbols
        symbols = self.validator.filter_valid_symbols(symbols)
        
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Log weekend warning
        if end_date.weekday() >= 5:  # Saturday or Sunday
            logger.warning(
                f"⚠️ Bugün hafta sonu ({end_date.strftime('%A')}). "
                "Son işlem günü verileri kullanılacak."
            )
        
        buy_signals = []
        sell_signals = []
        hold_signals = []
        errors = []
        
        logger.info(f"Starting daily scan for {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                signal = self._scan_symbol(symbol, start_date, end_date)
                
                if signal.signal == SignalType.BUY.value:
                    buy_signals.append(signal)
                elif signal.signal == SignalType.SELL.value:
                    sell_signals.append(signal)
                else:
                    hold_signals.append(signal)
                    
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Error scanning {symbol}: {e}")
        
        duration = time.time() - start_time
        
        result = DailyScanResult(
            scan_date=end_date,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            errors=errors,
            scan_duration=duration,
        )
        
        logger.info(
            f"Scan complete: {result.total_scanned} stocks, "
            f"{len(buy_signals)} BUY, {len(sell_signals)} SELL, "
            f"{len(errors)} errors, {duration:.1f}s"
        )
        
        return result
    
    def _scan_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> StockSignal:
        """Scan a single symbol and generate signal."""
        # Load data
        df = self.data_loader.load(symbol, start_date, end_date)
        
        if len(df) < self.lookback + 10:
            raise ValueError(f"Insufficient data: {len(df)} bars")
        
        # Generate features
        df_features, _ = prepare_features(df, normalize=True)
        df_features = df_features.dropna()
        
        if len(df_features) < self.lookback:
            raise ValueError(f"Insufficient features after NaN drop")
        
        # Get latest features for prediction
        latest_features = df_features[self.feature_columns].values[-self.lookback:]
        latest_features = np.nan_to_num(latest_features, nan=0.0)
        
        # Predict
        x = torch.from_numpy(latest_features.astype(np.float32)).unsqueeze(0)
        x = x.to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            # Handle Multi-Task output (direction, volatility)
            if isinstance(output, tuple):
                prob = output[0].item()
            else:
                prob = output.item()
        
        # Generate signal
        signal_result = self.signal_generator.generate(prob, current_position=None)
        
        # Get additional info
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        change_1d = (current_price - prev_price) / prev_price * 100
        
        # Get RSI and volatility from features
        rsi = df_features['rsi'].iloc[-1] if 'rsi' in df_features.columns else 50
        volatility = df_features['volatility'].iloc[-1] if 'volatility' in df_features.columns else 0
        
        # Get Sentiment
        try:
            sentiment = self.sentiment_analyzer.get_stock_sentiment(symbol)
        except Exception:
            sentiment = 0.0
            
        # Adjust signal based on sentiment? (Optional logic)
        # For now just recording it
        
        return StockSignal(
            symbol=symbol,
            probability=prob,
            signal=signal_result.signal.value.upper(),
            confidence=signal_result.confidence,
            current_price=current_price,
            change_1d=change_1d,
            rsi=rsi,
            volatility=volatility,
            sentiment_score=sentiment,
            reason=signal_result.reason,
        )


def generate_signal_report(result: DailyScanResult) -> str:
    """
    Generate a formatted text report of scan results.
    
    Args:
        result: DailyScanResult from scanner
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"📊 BIST100 GÜNLÜK SİNYAL RAPORU - {result.scan_date}")
    lines.append("=" * 60)
    lines.append("")
    
    # Summary
    lines.append(f"📈 Taranan Hisse: {result.total_scanned}")
    lines.append(f"✅ AL Sinyali: {len(result.buy_signals)}")
    lines.append(f"❌ SAT Sinyali: {len(result.sell_signals)}")
    lines.append(f"⏸️ BEKLE: {len(result.hold_signals)}")
    lines.append("")
    
    # Top buy signals
    if result.buy_signals:
        lines.append("-" * 60)
        lines.append("🟢 EN GÜÇLÜ AL SİNYALLERİ:")
        lines.append("-" * 60)
        
        for signal in result.get_top_signals(10):
            if signal.probability > 0.70:
                emoji = "🔥" 
                desc = "GÜÇLÜ AL"
            elif signal.probability > 0.60:
                emoji = "✅"
                desc = "AL"
            else:
                emoji = "⚠️"
                desc = "SPEKÜLATİF"
                
            lines.append(
                f"{emoji} {signal.symbol:6} | "
                f"Sinyal: {desc} | "
                f"Olasılık: {signal.probability:.1%} | "
                f"Fiyat: {signal.current_price:.2f} TL"
            )
            # lines.append(f"   └─ {signal.reason}") # Optional to save space
        lines.append("")
    
    # Sell signals
    if result.sell_signals:
        lines.append("-" * 60)
        lines.append("🔴 SAT SİNYALLERİ:")
        lines.append("-" * 60)
        
        for signal in sorted(result.sell_signals, key=lambda x: x.probability)[:5]:
            lines.append(
                f"❌ {signal.symbol:6} | "
                f"Olasılık: {signal.probability:.1%} | "
                f"Fiyat: {signal.current_price:.2f} TL"
            )
        lines.append("")
    
    # Errors
    if result.errors:
        lines.append("-" * 60)
        lines.append(f"⚠️ {len(result.errors)} hisse taranamadı")
        lines.append("-" * 60)
    
    lines.append("")
    lines.append("=" * 60)
    lines.append("⚠️ DİKKAT: Bu sinyaller tavsiye niteliğinde değildir.")
    lines.append("Yatırım kararlarınızı kendi araştırmanıza dayandırın.")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def generate_dashboard_json(result: DailyScanResult, output_path: str = "docs/dashboard_data.json"):
    """
    Generate JSON data for the web dashboard.
    """
    import json
    
    # Calculate average volatility
    all_signals = result.buy_signals + result.sell_signals + result.hold_signals
    avg_vol = np.mean([s.volatility for s in all_signals]) if all_signals else 0
    
    vol_status = "Normal"
    if avg_vol > 0.03: vol_status = "Yüksek ⚠️"
    elif avg_vol < 0.01: vol_status = "Düşük 💤"
    
    # Mock Portfolio Data (Since we don't have a live broker connection yet)
    # In a real scenario, this would come from a broker API or a local database
    portfolio = {
        "total_equity": 100000.0,
        "cash": 45000.0,
        "daily_pnl": 1250.0,
        "daily_pnl_pct": 1.25,
        "holdings": [
            {"symbol": "THYAO", "quantity": 100, "avg_price": 275.0, "current_price": 285.5, "pnl": 1050.0, "pnl_pct": 3.8},
            {"symbol": "AKBNK", "quantity": 500, "avg_price": 40.0, "current_price": 42.1, "pnl": 1050.0, "pnl_pct": 5.25},
            {"symbol": "ASELS", "quantity": 200, "avg_price": 55.0, "current_price": 54.5, "pnl": -100.0, "pnl_pct": -0.9},
        ]
    }
    
    # System Config
    from config import get_config
    conf = get_config()
    config_data = {
        "model_type": conf.model.model_type,
        "entry_threshold": conf.backtest.entry_threshold,
        "exit_threshold": conf.backtest.exit_threshold,
        "stop_loss": conf.backtest.stop_loss_pct,
        "take_profit": conf.backtest.take_profit_pct,
        "max_risk_per_trade": conf.risk.max_risk_per_trade,
        "max_drawdown": conf.risk.max_daily_drawdown,
    }
    
    data = {
        "scan_date": str(result.scan_date),
        "total_scanned": result.total_scanned,
        "buy_count": len(result.buy_signals),
        "sell_count": len(result.sell_signals),
        "hold_count": len(result.hold_signals),
        "error_count": len(result.errors),
        "market_volatility": vol_status,
        "buy_signals": [s.to_dict() for s in result.buy_signals],
        "sell_signals": [s.to_dict() for s in result.sell_signals],
        "hold_signals": [], 
        "portfolio": portfolio,
        "config": config_data
    }
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dashboard data saved to {output_path}")
