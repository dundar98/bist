#!/usr/bin/env python3
"""
Model Comparison Script.

Trains and evaluates multiple models side-by-side:
1. LSTM (Baseline)
2. Transformer (New)
3. Multi-Task (New)

Metrics compared:
- Test AUC
- Test F1 Score
- Training Time
- Loss Convergence
"""

import sys
from pathlib import Path
import logging
import time
from datetime import date
import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from data import BIST100Validator, SyntheticDataLoader, prepare_features
from models import create_model, LSTMModel, TransformerModel, MultiTaskModel
from training import (
    TradingDataset,
    ChronologicalSplitter,
    create_data_loaders,
    train_model,
)
from training.multitask_trainer import MultiTaskTrainer
from utils import setup_logging

logger = logging.getLogger(__name__)


def train_baseline_lstm(config, train_df, val_df, test_df, feature_columns, device):
    """Train Baseline LSTM."""
    logger.info("\n" + "-" * 40)
    logger.info("Training BASELINE LSTM")
    logger.info("-" * 40)
    
    # Datasets
    train_ds = TradingDataset(train_df, feature_columns, config.features.lookback_window)
    val_ds = TradingDataset(val_df, feature_columns, config.features.lookback_window)
    test_ds = TradingDataset(test_df, feature_columns, config.features.lookback_window)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, batch_size=config.training.batch_size
    )
    
    # Model
    model = create_model(
        "lstm",
        input_size=len(feature_columns),
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    
    # Train
    start_time = time.time()
    model, history, metrics = train_model(
        model, train_loader, val_loader, test_loader,
        epochs=15, patience=3, device=device
    )
    duration = time.time() - start_time
    
    return {
        "model": "LSTM (Baseline)",
        "auc": metrics['auc'],
        "f1": metrics['f1'],
        "accuracy": metrics['accuracy'],
        "loss": history.best_val_loss,
        "duration": duration,
        "best_epoch": history.best_epoch
    }


def train_transformer(config, train_df, val_df, test_df, feature_columns, device):
    """Train Transformer Model."""
    logger.info("\n" + "-" * 40)
    logger.info("Training TRANSFORMER")
    logger.info("-" * 40)
    
    # Datasets (Same as LSTM)
    train_ds = TradingDataset(train_df, feature_columns, config.features.lookback_window)
    val_ds = TradingDataset(val_df, feature_columns, config.features.lookback_window)
    test_ds = TradingDataset(test_df, feature_columns, config.features.lookback_window)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, batch_size=config.training.batch_size
    )
    
    # Model
    model = create_model(
        "transformer",
        input_size=len(feature_columns),
        hidden_size=64, # Transformer usually needs smaller embedding dim
        num_layers=2,
        dropout=0.2,
        nhead=4
    )
    
    # Train
    start_time = time.time()
    model, history, metrics = train_model(
        model, train_loader, val_loader, test_loader,
        epochs=15, patience=3, device=device
    )
    duration = time.time() - start_time
    
    return {
        "model": "Transformer (New)",
        "auc": metrics['auc'],
        "f1": metrics['f1'],
        "accuracy": metrics['accuracy'],
        "loss": history.best_val_loss,
        "duration": duration,
        "best_epoch": history.best_epoch
    }


def train_multitask(config, train_df, val_df, test_df, feature_columns, device):
    """Train Multi-Task Model."""
    logger.info("\n" + "-" * 40)
    logger.info("Training MULTI-TASK LEARNING")
    logger.info("-" * 40)
    
    # Datasets with Volatility Target
    train_ds = TradingDataset(train_df, feature_columns, config.features.lookback_window, return_volatility=True)
    val_ds = TradingDataset(val_df, feature_columns, config.features.lookback_window, return_volatility=True)
    test_ds = TradingDataset(test_df, feature_columns, config.features.lookback_window, return_volatility=True)
    
    # Custom collate might be needed if standard loader fails, but standard tensor stacking should work
    # We rely on Trainer handling the tuple of targets if needed, 
    # but MultiTaskTrainer handles robust unpacking
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, batch_size=config.training.batch_size
    )
    
    # Model
    model = create_model(
        "multitask",
        input_size=len(feature_columns),
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        backbone_type="lstm"
    )
    
    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        volatility_weight=0.5
    )
    
    # Train
    start_time = time.time()
    history = trainer.train(
        train_loader, val_loader, 
        epochs=15, patience=3
    )
    
    # Test Evaluation
    metrics = trainer.evaluate(test_loader)
    
    duration = time.time() - start_time
    
    return {
        "model": "Multi-Task (New)",
        "auc": metrics['auc'],
        "f1": metrics['f1'],
        "accuracy": metrics['accuracy'],
        "loss": history.best_val_loss,
        "duration": duration,
        "best_epoch": history.best_epoch
    }


def main():
    setup_logging()
    config = get_config()
    device = "cpu" # Force CPU for reliable demo run
    
    logger.info("Generating synthetic data for benchmarking...")
    validator = BIST100Validator()
    loader = SyntheticDataLoader(validator=validator)
    
    # Use single symbol for faster comparison
    symbol = "THYAO"
    df = loader.load(symbol, date(2020, 1, 1), date(2024, 1, 1))
    
    # Features
    df_features, feature_columns = prepare_features(df, normalize=True)
    df_features = df_features.dropna()
    
    # Splits
    splitter = ChronologicalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = splitter.split_dataframe(df_features)
    
    logger.info(f"Data ready. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    results = []
    
    # 1. Baseline
    res_lstm = train_baseline_lstm(config, train_df, val_df, test_df, feature_columns, device)
    results.append(res_lstm)
    
    # 2. Transformer
    res_trans = train_transformer(config, train_df, val_df, test_df, feature_columns, device)
    results.append(res_trans)
    
    # 3. Multi-Task
    res_multi = train_multitask(config, train_df, val_df, test_df, feature_columns, device)
    results.append(res_multi)
    
    # Report
    print("\n" + "=" * 80)
    print(f"{'MODEL COMPARISON RESULTS':^80}")
    print("=" * 80)
    print(f"{'Model':<20} | {'AUC':<10} | {'F1 Score':<10} | {'Acc':<10} | {'Time (s)':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['model']:<20} | {r['auc']:.4f}     | {r['f1']:.4f}     | {r['accuracy']:.4f}     | {r['duration']:.1f}s")
    
    print("-" * 80)
    
    # Check if improved
    baseline_auc = res_lstm['auc']
    best_new_auc = max(res_trans['auc'], res_multi['auc'])
    
    if best_new_auc > baseline_auc:
        print(f"\n✅ SUCCESS: New models outperform Baseline by {(best_new_auc - baseline_auc)*100:.2f}%")
        print(f"Recommended model: {res_trans['model'] if res_trans['auc'] > res_multi['auc'] else res_multi['model']}")
    else:
        print("\n⚠️ NOTE: Baseline performs comparably. Try hyperparameter tuning or more data.")

if __name__ == "__main__":
    main()
