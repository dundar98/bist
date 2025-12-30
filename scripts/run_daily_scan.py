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
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="UZUN", choices=["KISA", "ORTA", "UZUN"], help="Scanning mode (KISA/ORTA/UZUN)")
    
    # Email arguments
    parser.add_argument("--email", type=str, default="false", help="Send email report (true/false)")
    parser.add_argument("--sender-email", type=str, help="Email sender address")
    parser.add_argument("--sender-password", type=str, help="Email sender password")
    parser.add_argument("--email-to", type=str, help="Email recipient address")
    
    # Data arguments
    parser.add_argument("--data-source", type=str, default="yfinance", help="Data source (yfinance/synthetic)")
    parser.add_argument("--output-dir", type=str, default="output/scans", help="Output directory")
    parser.add_argument("--lookback", type=int, default=200, help="Lookback days for scan")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols to scan")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of symbols to scan")
    
    return parser.parse_args()

def main():
    # Fix Windows console encoding for emojis (Skipped to avoid hang)
    pass
        
    import pandas as pd
    
    args = parse_args()
    mode = args.mode.upper()
    config = get_config()
    
    # Paths
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path mapping based on mode
    model_files = {
        "KISA": "transformer_kisa.pt",
        "ORTA": "transformer_orta.pt",
        "UZUN": "transformer_uzun.pt"
    }
    model_name = model_files.get(mode, "transformer_uzun.pt")
    model_path = PROJECT_ROOT / "models" / model_name
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}. Please train the {mode} model first.")
        sys.exit(1)
        
    logger.info(f"Using {mode} model: {model_path}")
    
    # Custom symbols list
    symbols_to_scan = None
    if args.symbols:
        symbols_to_scan = [s.strip() for s in args.symbols.split(",")]
        logger.info(f"Custom symbols to scan: {symbols_to_scan}")
    
    try:
        # Initialize Scanner
        scanner = DailyScanner(
            model_path=str(model_path),
            config=config,
            device=config.training.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize scanner: {e}")
        sys.exit(1)
        
    # Run Scan
    logger.info(f"Running {mode} scan...")
    try:
        # Pass the mode to scan_all
        result = scanner.scan_all(symbols=symbols_to_scan, lookback_days=args.lookback, limit=args.limit, mode=mode)
    except Exception as e:
        logger.error(f"Scan failed: {e}")
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
    should_send_email = args.email.lower() == "true"
    
    # Fallback to env vars if args not provided but env vars exist
    sender_email = args.sender_email or os.getenv("BIST_EMAIL_SENDER")
    sender_password = args.sender_password or os.getenv("BIST_EMAIL_PASSWORD") or os.getenv("EMAIL_PASSWORD")
    recipient = args.email_to or os.getenv("BIST_EMAIL_RECIPIENTS") or os.getenv("EMAIL_RECIPIENT")
    
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
                
                # Append dashboard link and timeframe to report
                email_body = f"⏳ Tarama Modu: {mode}\n" + text_report + f"\n\n📊 Web Dashboard: {dashboard_link}"
                
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
