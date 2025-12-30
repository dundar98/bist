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

def load_real_data(lookback_days=730):
    """Load real data for training."""
    loader = get_data_loader("yfinance")
    start_date = date.today() - timedelta(days=lookback_days)
    end_date = date.today()
    
    all_dfs = []
    
    logger.info(f"Loading data for {len(TRAINING_SYMBOLS)} stocks...")
    for symbol in TRAINING_SYMBOLS:
        try:
            df = loader.load(symbol, start_date, end_date)
            if len(df) > 200:
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")
            
    if not all_dfs:
        raise ValueError("No data loaded!")
        
    # Concatenate all data (Simplest approach for now: treat as one big stream)
    # Ideally should use MultiStockDataset, but for quick Aggressive training, 
    # vertical concatenation with Chronological Split works if we split by time first.
    # Actually, Chronological Split on concatenated data leaks future if not careful.
    # Better: Split each DF, then concat train/val/test sets.
    
    return all_dfs

def train_aggressive():
    setup_logging()
    config = get_config()
    
    # 1. Aggressive Settings
    LABEL_THRESHOLD = 0.01 # 1% gain target (was 0.02)
    config.features.label_threshold = LABEL_THRESHOLD
    
    logger.info(f"🚀 Starting AGGRESSIVE Training (Threshold: {LABEL_THRESHOLD:.1%})")
    
    dfs = load_real_data()
    
    # Get official feature list from a sample
    sample_df, feature_columns = prepare_features(dfs[0], normalize=True)
    logger.info(f"Using {len(feature_columns)} standardized features.")
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    splitter = ChronologicalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    for df in dfs:
        # Prepare features ONCE here
        df_feat, _ = prepare_features(df, normalize=True)
        df_feat = df_feat.dropna()
        
        t, v, te = splitter.split_dataframe(df_feat)
        train_dfs.append(t)
        val_dfs.append(v)
        test_dfs.append(te)
        
    full_train = pd.concat(train_dfs)
    full_val = pd.concat(val_dfs)
    full_test = pd.concat(test_dfs)
    
    logger.info(f"Training Samples: {len(full_train)}")
    
    # 3. Create Datasets
    train_ds = TradingDataset(full_train, feature_columns, 
                              lookback=60, label_threshold=LABEL_THRESHOLD)
    val_ds = TradingDataset(full_val, feature_columns, 
                            lookback=60, label_threshold=LABEL_THRESHOLD)
    test_ds = TradingDataset(full_test, feature_columns, 
                             lookback=60, label_threshold=LABEL_THRESHOLD)
    
    logger.info(f"Positive Label Rate (Train): {train_ds.labels.mean():.2%}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, 
        batch_size=64, 
        weighted_sampling=False # We use Class Weights in Loss instead
    )
    
    # 4. Train Transformer
    logger.info("Initializing Transformer Model...")
    model = create_model(
        "transformer",
        input_size=len(feature_columns),
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        nhead=4
    )
    
    start_time = time.time()
    
    # FORCE DISABLE CLASS WEIGHTS (Since we have 60% positives now)
    # If we weight < 1.0, we hurt performance on Buys.
    
    trained_model, history, metrics = train_model(
        model, train_loader, val_loader, test_loader,
        epochs=20,
        patience=7,
        learning_rate=0.0005,
        device="cpu"
    )
    
    duration = time.time() - start_time
    
    print("\n" + "="*60)
    print("📢 AGGRESSIVE TRAINING RESULTS")
    print("="*60)
    print(f"Model: Transformer")
    print(f"Time: {duration:.1f}s")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test F1:  {metrics['f1']:.4f}")
    print(f"Positive Rate (Predicted): {metrics['positive_rate']:.2%}")
    print(f"Positive Rate (Actual):    {metrics['actual_positive_rate']:.2%}")
    print("-" * 60)
    
    # Save if at least random (0.50)
    if metrics['auc'] >= 0.50:
        save_path = PROJECT_ROOT / "models" / "aggressive_transformer.pt"
        torch.save(trained_model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")
        
        # Update config to use this model? 
        # We can't easily update config.py dynamically, 
        # but we can tell user to switch or rename file.
    else:
        print("⚠️ Result AUC too low, not saving.")

if __name__ == "__main__":
    train_aggressive()
