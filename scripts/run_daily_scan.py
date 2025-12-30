#!/usr/bin/env python3
"""
Daily Scan Runner.

Executes the daily market scan and notifications.
"""

import sys
import logging
import argparse
import os
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from training import Trainer
from models import create_model
from notifications import DailyScanner, generate_signal_report
from notifications.scanner import generate_dashboard_json
from notifications.email_service import EmailNotifier, EmailConfig

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run daily BIST100 scan")
    
    # Email arguments
    parser.add_argument("--email", action="store_true", help="Send email report")
    parser.add_argument("--sender-email", type=str, help="Email sender address")
    parser.add_argument("--sender-password", type=str, help="Email sender password")
    parser.add_argument("--email-to", type=str, help="Email recipient address")
    
    # Data arguments
    parser.add_argument("--data-source", type=str, default="yfinance", help="Data source (yfinance/synthetic)")
    parser.add_argument("--output-dir", type=str, default="output/scans", help="Output directory")
    parser.add_argument("--lookback", type=int, default=200, help="Lookback days for scan")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    
    # Paths
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = PROJECT_ROOT / "output" / "model.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Please train first.")
        sys.exit(1)
        
    import pandas as pd
    from data import prepare_features
    
    # Dynamically determine feature columns and input size
    # We create a dummy dataframe to see what prepare_features returns
    dummy_data = pd.DataFrame({
        'open': [10.0] * 100,
        'high': [11.0] * 100,
        'low': [9.0] * 100,
        'close': [10.5] * 100,
        'volume': [1000] * 100
    })
    dummy_features, _ = prepare_features(dummy_data, normalize=True)
    feature_columns = dummy_features.columns.tolist() # Use all generated columns
    input_size = len(feature_columns)
    logger.info(f"Computed input size: {input_size} (Features: {len(feature_columns)})")
    
    # Load Model
    logger.info("Loading model...")
    model = create_model(
        config.model.model_type,
        input_size=input_size, 
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers
    )
    
    import torch
    model.load(str(model_path))
    
    # Init Scanner
    scanner = DailyScanner(
        model=model,
        feature_columns=feature_columns,
        lookback=config.features.lookback_window,
        entry_threshold=config.backtest.entry_threshold,
        data_source=args.data_source
    )
    
    # RUN SCAN
    try:
        # Scan ALL available symbols (no limit)
        result = scanner.scan_all(lookback_days=args.lookback, limit=None)
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        # Only exit if critical, but we might want to try reporting error
        sys.exit(1)
        
    # Generate Reports
    text_report = generate_signal_report(result)
    print(text_report)
    
    # Save Report to file
    report_file = output_dir / f"scan_{result.scan_date}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(text_report)
    
    # Generate Web Dashboard Data
    docs_path = PROJECT_ROOT / "docs" / "dashboard_data.json"
    generate_dashboard_json(result, str(docs_path))
    
    # Send Email
    should_send_email = args.email
    
    # Fallback to env vars if args not provided but env vars exist
    sender_email = args.sender_email or os.getenv("BIST_EMAIL_SENDER")
    sender_password = args.sender_password or os.getenv("BIST_EMAIL_PASSWORD") or os.getenv("EMAIL_PASSWORD")
    recipient = args.email_to or os.getenv("BIST_EMAIL_RECIPIENTS") or os.getenv("EMAIL_RECIPIENT")
    
    # Debug logging
    logger.info(f"Email Request - Flag: {should_send_email}")
    logger.info(f"Credentials Present - Sender: {bool(sender_email)}, Pass: {bool(sender_password)}, To: {bool(recipient)}")
    
    if should_send_email or (sender_email and sender_password and recipient):
        if sender_email and sender_password and recipient:
            logger.info(f"Sending email to {recipient}...")
            try:
                # Create config object
                email_config = EmailConfig(
                    sender_email=sender_email,
                    sender_password=sender_password,
                    recipient_emails=[r.strip() for r in recipient.split(',')]
                )
                
                notifier = EmailNotifier(email_config)
                
                dashboard_link = f"https://{os.getenv('GITHUB_REPOSITORY_OWNER', 'dundar98')}.github.io/bist"
                
                # Append dashboard link to report
                email_body = text_report + f"\n\n📊 Web Dashboard: {dashboard_link}"
                
                notifier.send_signal_report(
                    report_text=email_body,
                    scan_date=str(result.scan_date),
                    buy_count=len(result.buy_signals),
                    sell_count=len(result.sell_signals)
                )
                logger.info("Email sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
        else:
            logger.warning("Email requested but credentials missing.")
            logger.warning(f"Sender: {bool(sender_email)}, Pass: {bool(sender_password)}, To: {bool(recipient)}")
    else:
        logger.info("Email sending skipped (not requested or credentials missing).")

if __name__ == "__main__":
    main()
