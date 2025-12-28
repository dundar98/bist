#!/usr/bin/env python3
"""
Daily Scan Runner.

Executes the daily market scan and notifications.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from training import Trainer
from models import create_model
from notifications import DailyScanner, generate_signal_report
from notifications.scanner import generate_dashboard_json # Added import
from notifications.email_service import EmailNotifier

logger = logging.getLogger(__name__)


def main():
    # Setup
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    
    # Paths
    output_dir = PROJECT_ROOT / "output"
    model_path = output_dir / "model.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Please train first.")
        sys.exit(1)
        
    # Load Model
    logger.info("Loading model...")
    # input_size will be adjusted by load_state_dict but we need initial
    model = create_model(
        config.model.model_type,
        input_size=1, # Dummy
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers
    )
    
    import torch
    model.load(str(model_path))
    
    # Init Scanner
    scanner = DailyScanner(
        model=model,
        feature_columns=["open", "high", "low", "close", "volume"], # dummy, will be loaded from data
        # Actually scanner re-prepares features so columns will be correct
        lookback=config.features.lookback_window,
        entry_threshold=config.backtest.entry_threshold
    )
    
    # Correct feature columns override
    # The scanner uses prep functionality which generates specific columns
    # We should trust the scanner's internal data loader + prep features logic
    
    # RUN SCAN
    try:
        result = scanner.scan_all(lookback_days=200)
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        sys.exit(1)
        
    # Generate Reports
    text_report = generate_signal_report(result)
    print(text_report)
    
    # Generate Web Dashboard Data
    docs_path = PROJECT_ROOT / "docs" / "dashboard_data.json"
    generate_dashboard_json(result, str(docs_path))
    
    # Send Email if configured
    import os
    if os.getenv("EMAIL_PASSWORD"):
        notifier = EmailNotifier()
        notifier.send_daily_report(
            recipient=os.getenv("EMAIL_RECIPIENT", "user@example.com"),
            result=result,
            dashboard_link=f"https://{os.getenv('GITHUB_REPOSITORY_OWNER', 'user')}.github.io/bist"
        )
    else:
        logger.info("Email credentials not found, skipping email.")

if __name__ == "__main__":
    main()
