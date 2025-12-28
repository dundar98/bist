#!/usr/bin/env python3
"""
BIST100 Deep Learning Trading System - Main Pipeline Script.

This script demonstrates the complete workflow:
1. Load and validate BIST100 data
2. Generate features
3. Create labeled dataset
4. Train LSTM model
5. Run backtest
6. Generate performance report

Usage:
    python scripts/run_pipeline.py --mode demo
    python scripts/run_pipeline.py --symbols THYAO GARAN --start 2020-01-01 --end 2024-01-01
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

# Import project modules
from config import get_config, SystemConfig
from data import (
    BIST100Validator,
    SyntheticDataLoader,
    YFinanceLoader,
    get_data_loader,
    prepare_features,
)
from models import create_model
from training import (
    TradingDataset,
    ChronologicalSplitter,
    create_data_loaders,
    train_model,
)
from backtest import BacktestEngine, generate_report
from utils import setup_logging


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BIST100 Deep Learning Trading System"
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "train", "backtest", "paper"],
        default="demo",
        help="Execution mode"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Stock symbols to trade (default: use config)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2024-01-01",
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--data-source",
        choices=["yfinance", "synthetic", "csv"],
        default="synthetic",
        help="Data source"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def run_demo_pipeline(config: SystemConfig, output_dir: Path):
    """
    Run a complete demo pipeline with synthetic data.
    
    This demonstrates all system components without requiring real data.
    """
    logger.info("=" * 60)
    logger.info("BIST100 DEEP LEARNING TRADING SYSTEM - DEMO MODE")
    logger.info("=" * 60)
    
    # Step 1: Validate symbols
    logger.info("\n[STEP 1] Validating BIST100 symbols...")
    validator = BIST100Validator()
    symbols = ["THYAO", "GARAN", "AKBNK", "EREGL", "TUPRS"]
    validated = validator.validate_symbols(symbols)
    logger.info(f"Validated {len(validated)} symbols: {validated}")
    
    # Step 2: Load synthetic data
    logger.info("\n[STEP 2] Loading synthetic OHLCV data...")
    loader = SyntheticDataLoader(validator=validator)
    
    start_date = date(2020, 1, 1)
    end_date = date(2024, 1, 1)
    
    all_data = {}
    for symbol in validated:
        df = loader.load(symbol, start_date, end_date)
        logger.info(f"  {symbol}: {len(df)} bars")
        all_data[symbol] = df
    
    # Step 3: Generate features
    logger.info("\n[STEP 3] Generating features...")
    feature_dfs = {}
    feature_columns = None
    
    for symbol, df in all_data.items():
        df_features, feat_names = prepare_features(
            df,
            normalize=True,
            normalization_method="rolling",
        )
        feature_dfs[symbol] = df_features
        
        if feature_columns is None:
            feature_columns = feat_names
        
        logger.info(f"  {symbol}: {len(feat_names)} features generated")
    
    logger.info(f"Feature columns: {feature_columns[:5]}... (total {len(feature_columns)})")
    
    # Step 4: Create datasets
    logger.info("\n[STEP 4] Creating training datasets...")
    
    # Use first symbol for demonstration
    demo_symbol = validated[0]
    demo_df = feature_dfs[demo_symbol].dropna()
    
    # Split data
    splitter = ChronologicalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = splitter.split_dataframe(demo_df)
    
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val:   {len(val_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    
    # Create PyTorch datasets
    is_multitask = config.model.model_type == "multitask"
    
    train_dataset = TradingDataset(
        data=train_df,
        feature_columns=feature_columns,
        lookback=config.features.lookback_window,
        label_threshold=config.labels.threshold_pct,
        label_horizon=config.labels.horizon_bars,
        return_volatility=is_multitask,
    )
    
    val_dataset = TradingDataset(
        data=val_df,
        feature_columns=feature_columns,
        lookback=config.features.lookback_window,
        label_threshold=config.labels.threshold_pct,
        label_horizon=config.labels.horizon_bars,
        return_volatility=is_multitask,
    )
    
    test_dataset = TradingDataset(
        data=test_df,
        feature_columns=feature_columns,
        lookback=config.features.lookback_window,
        label_threshold=config.labels.threshold_pct,
        label_horizon=config.labels.horizon_bars,
        return_volatility=is_multitask,
    )
    
    logger.info(f"  Train dataset: {len(train_dataset)} sequences")
    logger.info(f"  Label balance: {train_dataset.labels.mean():.2%} positive")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config.training.batch_size,
    )
    
    # Step 5: Create model
    logger.info("\n[STEP 5] Creating LSTM model...")
    input_size = len(feature_columns)
    
    model = create_model(
        model_type=config.model.model_type,
        input_size=input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        backbone_type="lstm" if config.model.model_type == "multitask" else None,
    )
    
    logger.info(f"  Model type: {model.model_type}")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    logger.info(f"  Device: {config.training.device}")
    
    # Step 6: Train model
    logger.info("\n[STEP 6] Training model...")
    
    trained_model, history, test_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=20,  # Reduced for demo
        learning_rate=config.training.learning_rate,
        patience=5,
        device=config.training.device,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    
    logger.info(f"  Best epoch: {history.best_epoch}")
    logger.info(f"  Best val loss: {history.best_val_loss:.4f}")
    
    if test_metrics:
        logger.info(f"  Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"  Test F1: {test_metrics['f1']:.4f}")
    
    # Step 7: Generate predictions for backtest
    logger.info("\n[STEP 7] Generating predictions for backtest...")
    
    trained_model.eval()
    all_preds = []
    
    # Predict on test data
    test_features = test_dataset.features
    for i in range(len(test_features)):
        x = torch.from_numpy(test_features[i:i+1]).to(config.training.device)
        with torch.no_grad():
            pred = trained_model(x)
            
        if isinstance(pred, tuple):
            # Multitask returns (direction, volatility)
            # We use direction probability for backtest
            pred = pred[0]
            
        all_preds.append(pred.item())
    
    predictions = np.array(all_preds)
    logger.info(f"  Generated {len(predictions)} predictions")
    logger.info(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    logger.info(f"  Mean prediction: {predictions.mean():.3f}")
    
    # Step 8: Run backtest
    logger.info("\n[STEP 8] Running backtest...")
    
    # Need OHLCV data aligned with predictions
    # The test dataset starts from lookback position
    test_start_idx = len(train_df) + len(val_df) + config.features.lookback_window - 1
    test_end_idx = test_start_idx + len(predictions)
    
    # Slice original data for backtest
    backtest_data = demo_df.iloc[test_start_idx:test_end_idx].reset_index(drop=True)
    
    # Ensure we have same length
    min_len = min(len(backtest_data), len(predictions))
    backtest_data = backtest_data.iloc[:min_len]
    predictions = predictions[:min_len]
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        entry_threshold=config.backtest.entry_threshold,
        exit_threshold=config.backtest.exit_threshold,
        stop_loss_pct=config.backtest.stop_loss_pct,
        take_profit_pct=config.backtest.take_profit_pct,
        position_size_pct=config.backtest.fixed_fraction,
        commission_pct=config.backtest.commission_pct,
    )
    
    state = engine.run(
        data=backtest_data,
        predictions=predictions,
        symbol=demo_symbol,
    )
    
    # Step 9: Generate report
    logger.info("\n[STEP 9] Generating performance report...")
    
    reporter = generate_report(
        state=state,
        initial_capital=config.backtest.initial_capital,
        output_dir=str(output_dir),
        print_summary=True,
        save_files=True,
        plot=False,  # Disable for headless systems
    )
    
    # Step 10: Save model
    logger.info("\n[STEP 10] Saving trained model...")
    
    model_path = output_dir / "model.pt"
    trained_model.save(str(model_path))
    logger.info(f"  Saved model to {model_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Model saved: {model_path}")
    logger.info(f"  Total trades: {len(state.trades)}")
    logger.info(f"  Final equity: {state.equity:.2f}")
    
    return trained_model, reporter


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(
        level=log_level,
        log_file=f"{args.output_dir}/trading_system.log",
    )
    
    # Load configuration
    config = get_config()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == "demo":
            run_demo_pipeline(config, output_dir)
        else:
            logger.info(f"Mode '{args.mode}' not yet implemented in this script")
            logger.info("Use --mode demo for a complete demonstration")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
