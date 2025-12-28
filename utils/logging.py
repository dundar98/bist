"""
Logging Utilities.

Configures logging for the trading system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the trading system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        format_string: Optional custom format string
        
    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class TradeLogger:
    """
    Specialized logger for trade events.
    
    Logs trades with structured format for easy parsing.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize trade logger.
        
        Args:
            log_file: Optional dedicated trade log file
        """
        self.logger = logging.getLogger("trades")
        
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(message)s"
            ))
            self.logger.addHandler(handler)
    
    def log_entry(
        self,
        symbol: str,
        price: float,
        size: float,
        signal: float,
        reason: str,
    ) -> None:
        """Log trade entry."""
        self.logger.info(
            f"ENTRY | {symbol} | price={price:.2f} | size={size:.2f} | "
            f"signal={signal:.3f} | {reason}"
        )
    
    def log_exit(
        self,
        symbol: str,
        price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Log trade exit."""
        self.logger.info(
            f"EXIT  | {symbol} | price={price:.2f} | pnl={pnl:.2f} ({pnl_pct:.2%}) | "
            f"{reason}"
        )
    
    def log_signal(
        self,
        symbol: str,
        probability: float,
        signal: str,
        action: str,
    ) -> None:
        """Log signal generation."""
        self.logger.debug(
            f"SIGNAL | {symbol} | prob={probability:.3f} | signal={signal} | action={action}"
        )
