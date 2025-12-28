#!/usr/bin/env python3
"""
Daily Signal Runner.

Runs the daily scan and sends email notifications.
Can be scheduled with Windows Task Scheduler or cron.

Usage:
    python scripts/run_daily_scan.py --email
    python scripts/run_daily_scan.py --symbols THYAO GARAN AKBNK
    
To schedule (Windows):
    schtasks /create /tn "BIST100_Daily_Scan" /tr "python C:\path\to\run_daily_scan.py --email" /sc daily /st 09:00

To schedule (Linux/Mac cron):
    0 9 * * 1-5 /usr/bin/python3 /path/to/run_daily_scan.py --email
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import date

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="BIST100 Daily Signal Scanner")
    
    parser.add_argument(
        "--model",
        type=str,
        default="output/model.pt",
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Specific symbols to scan (default: top 30 BIST100)"
    )
    
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send results via email"
    )
    
    parser.add_argument(
        "--email-to",
        type=str,
        default=None,
        help="Override recipient email (comma-separated)"
    )
    
    parser.add_argument(
        "--smtp-server",
        type=str,
        default="smtp.gmail.com",
        help="SMTP server"
    )
    
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=587,
        help="SMTP port"
    )
    
    parser.add_argument(
        "--sender-email",
        type=str,
        default=os.getenv("BIST_EMAIL_SENDER"),
        help="Sender email (or set BIST_EMAIL_SENDER env var)"
    )
    
    parser.add_argument(
        "--sender-password",
        type=str,
        default=os.getenv("BIST_EMAIL_PASSWORD"),
        help="Sender password (or set BIST_EMAIL_PASSWORD env var)"
    )
    
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print report, don't save"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/scans",
        help="Directory to save scan results"
    )
    
    parser.add_argument(
        "--data-source",
        choices=["yfinance", "synthetic"],
        default="yfinance",
        help="Data source"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser.parse_args()


def load_model_and_config(model_path: str):
    """Load trained model and feature columns."""
    import torch
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run 'python scripts/run_pipeline.py --mode demo' first to train a model."
        )
    
    # Load model
    from models import LSTMModel
    model = LSTMModel.load(model_path)
    
    # Get feature columns from config
    from config import get_config
    config = get_config()
    
    # These are the normalized feature columns
    # (matching what was used in training)
    feature_columns = [
        'log_return_norm', 'return_5d_norm', 'return_10d_norm', 'return_20d_norm',
        'volatility_norm', 'volatility_ratio_norm', 'intraday_range_norm',
        'rsi_norm', 'rsi_normalized_norm', 'macd_norm', 'macd_signal_norm',
        'macd_histogram_norm', 'macd_normalized_norm', 'atr_norm', 'atr_pct_norm',
        'sma_short_norm', 'sma_long_norm', 'price_to_sma_short_norm',
        'price_to_sma_long_norm', 'sma_crossover_norm', 'trend_strength_norm',
        'bars_since_crossover_norm', 'bb_upper_norm', 'bb_lower_norm',
        'bb_middle_norm', 'bb_position_norm', 'bb_bandwidth_norm',
        'volume_sma_norm', 'volume_ratio_norm', 'obv_change_norm',
        'obv_norm', 'obv_normalized_norm', 'vpt_norm', 'higher_high_norm',
        'lower_low_norm', 'consecutive_up_norm', 'consecutive_down_norm',
        'gap_up_norm', 'gap_down_norm', 'dist_from_10d_high_norm',
        'dist_from_10d_low_norm', 'dist_from_20d_high_norm',
        'dist_from_20d_low_norm', 'dist_from_50d_high_norm', 'dist_from_50d_low_norm',
    ]
    
    return model, feature_columns, config


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("=" * 60)
    logger.info("BIST100 GÜNLÜK SİNYAL TARAMASI")
    logger.info("=" * 60)
    logger.info(f"Tarih: {date.today()}")
    
    try:
        # Load model
        logger.info(f"Model yükleniyor: {args.model}")
        model, feature_columns, config = load_model_and_config(args.model)
        logger.info(f"Model yüklendi. Özellik sayısı: {len(feature_columns)}")
        
        # Create scanner
        from notifications import DailyScanner, generate_signal_report
        
        scanner = DailyScanner(
            model=model,
            feature_columns=feature_columns,
            lookback=config.features.lookback_window,
            entry_threshold=config.backtest.entry_threshold,
            exit_threshold=config.backtest.exit_threshold,
            data_source=args.data_source,
            device=config.training.device,
        )
        
        # Run scan
        logger.info("Tarama başlıyor...")
        result = scanner.scan_all(symbols=args.symbols)
        
        # Generate report
        report = generate_signal_report(result)
        
        # Print report
        print("\n" + report)
        
        # Save report
        if not args.print_only:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = output_dir / f"scan_{date.today().isoformat()}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Rapor kaydedildi: {report_file}")
            
            # Save as JSON too
            import json
            json_file = output_dir / f"scan_{date.today().isoformat()}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': result.to_summary_dict(),
                    'buy_signals': [s.to_dict() for s in result.buy_signals],
                    'sell_signals': [s.to_dict() for s in result.sell_signals],
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON kaydedildi: {json_file}")
        
        # Send email if requested
        if args.email:
            from notifications import EmailNotifier, EmailConfig
            
            # Get recipients
            recipients = None
            if args.email_to:
                recipients = [r.strip() for r in args.email_to.split(",")]
            
            if not args.sender_email or not args.sender_password:
                logger.error(
                    "E-posta göndermek için gönderici bilgileri gerekli!\n"
                    "Kullanım:\n"
                    "  --sender-email your@gmail.com --sender-password app_password\n"
                    "  veya BIST_EMAIL_SENDER ve BIST_EMAIL_PASSWORD env değişkenleri"
                )
                return 1
            
            email_config = EmailConfig(
                smtp_server=args.smtp_server,
                smtp_port=args.smtp_port,
                sender_email=args.sender_email,
                sender_password=args.sender_password,
                recipient_emails=recipients or [args.sender_email],
            )
            
            notifier = EmailNotifier(email_config)
            
            success = notifier.send_signal_report(
                report_text=report,
                scan_date=str(date.today()),
                buy_count=len(result.buy_signals),
                sell_count=len(result.sell_signals),
            )
            
            if success:
                logger.info("✅ E-posta başarıyla gönderildi!")
            else:
                logger.error("❌ E-posta gönderilemedi!")
                return 1
        
        logger.info("=" * 60)
        logger.info("TARAMA TAMAMLANDI")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Hata oluştu: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
