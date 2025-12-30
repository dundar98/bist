#!/usr/bin/env python3
"""
Aggressive Model Training Script.

Trains a Transformer-based model with aggressive settings:
1. Lower Profit Threshold (1% instead of 2%)
2. Class Weighting (Penalize missing buy signals)
3. Transformer Architecture
4. Real Data (Top 20 Liquid Stocks)
"""

import sys
import logging
import time
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import torch
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from data import get_data_loader, prepare_features
from models import create_model
from training import (
    TradingDataset, 
    ChronologicalSplitter, 
    create_data_loaders, 
    train_model
)
from utils import setup_logging

logger = logging.getLogger(__name__)

# Top 20 Liquid BIST Stocks for robust training
TRAINING_SYMBOLS = [
    "THYAO", "GARAN", "AKBNK", "EREGL", "KCHOL",
    "SISE", "TUPRS", "SAHOL", "TCELL", "BIMAS",
    "ASELS", "YKBNK", "HALKB", "PGSUS", "TAVHL",
    "FROTO", "TOASO", "ARCLK", "PETKM", "SASA"
]

import argparse

def load_real_data(interval="1d", lookback_days=730):
    """Load real data for training."""
    loader = get_data_loader("yfinance")
    start_date = date.today() - timedelta(days=lookback_days)
    end_date = date.today()
    
    all_dfs = []
    
    logger.info(f"Loading {interval} data for {len(TRAINING_SYMBOLS)} stocks...")
    for symbol in TRAINING_SYMBOLS:
        try:
            df = loader.load(symbol, start_date, end_date, interval=interval)
            if len(df) > 100:
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")
            
    if not all_dfs:
        raise ValueError("No data loaded!")
    return all_dfs

def train_vade(mode="UZUN", epochs=40, batch_size=64):
    setup_logging()
    config = get_config()
    
    # 1. Mode Mapping (Adjusted for easier convergence)
    MODES = {
        "KISA": {"interval": "15m", "threshold": 0.008, "horizon": 12, "model_name": "transformer_kisa.pt"},
        "ORTA": {"interval": "1h", "threshold": 0.03, "horizon": 24, "model_name": "transformer_orta.pt"},
        "UZUN": {"interval": "1d", "threshold": 0.07, "horizon": 20, "model_name": "transformer_uzun.pt"}
    }
    
    m_cfg = MODES.get(mode, MODES["UZUN"])
    INTERVAL = m_cfg["interval"]
    LABEL_THRESHOLD = m_cfg["threshold"]
    HORIZON = m_cfg["horizon"]
    SAVE_FILE = m_cfg["model_name"]
    
    logger.info(f"🚀 Training mode: {mode} (Interval: {INTERVAL}, Target: {LABEL_THRESHOLD:.1%}, Horizon: {HORIZON})")
    
    # Fetch more data for intraday if needed
    lookback = 730 if INTERVAL == "1d" else 59 # yfinance limit for intraday
    dfs = load_real_data(interval=INTERVAL, lookback_days=lookback)
    
    # Get official feature list from a sample
    sample_feat, feature_columns = prepare_features(dfs[0], normalize=True)
    logger.info(f"Using {len(feature_columns)} standardized features.")
    
    train_dfs, val_dfs, test_dfs = [], [], []
    splitter = ChronologicalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    for df in dfs:
        df_feat, _ = prepare_features(df, normalize=True)
        df_feat = df_feat.dropna()
        if len(df_feat) < 100: continue
        
        t, v, te = splitter.split_dataframe(df_feat)
        train_dfs.append(t); val_dfs.append(v); test_dfs.append(te)
        
    full_train = pd.concat(train_dfs); full_val = pd.concat(val_dfs); full_test = pd.concat(test_dfs)
    
    # 3. Create Datasets
    train_ds = TradingDataset(full_train, feature_columns, lookback=60, label_threshold=LABEL_THRESHOLD, label_horizon=HORIZON)
    val_ds = TradingDataset(full_val, feature_columns, lookback=60, label_threshold=LABEL_THRESHOLD, label_horizon=HORIZON)
    test_ds = TradingDataset(full_test, feature_columns, lookback=60, label_threshold=LABEL_THRESHOLD, label_horizon=HORIZON)
    
    logger.info(f"Positive Label Rate (Train): {train_ds.labels.mean():.2%}")
    
    train_loader, val_loader, test_loader = create_data_loaders(train_ds, val_ds, test_ds, batch_size=batch_size)
    
    # 4. Train Transformer
    model = create_model("transformer", input_size=len(feature_columns), hidden_size=128, num_layers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    trained_model, _, metrics = train_model(
        model, train_loader, val_loader, test_loader,
        epochs=epochs, patience=15, learning_rate=0.0007, device=device
    )
    
    # Save Model (Ensuring files exist for pipeline)
    if metrics['auc'] >= 0.40:
        save_path = PROJECT_ROOT / "models" / SAVE_FILE
        # Attach metadata for Scanner
        trained_model.feature_columns = feature_columns
        trained_model.lookback = 60
        torch.save(trained_model, save_path)
        print(f"✅ {mode} Model saved to {save_path} (AUC: {metrics['auc']:.4f})")
    else:
        print(f"⚠️ {mode} Model AUC too low ({metrics['auc']:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="UZUN", choices=["KISA", "ORTA", "UZUN"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    
    train_vade(args.mode, epochs=args.epochs, batch_size=args.batch_size)
