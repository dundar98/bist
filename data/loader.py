"""
Data Loader Module.

Provides abstract interface and concrete implementations for loading
OHLCV data from various sources. All loaders enforce BIST100 validation.
"""

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from .bist100_validator import BIST100Validator, get_validator, BIST100ValidationError

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Base exception for data loading errors."""
    pass


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    All data loaders must:
    1. Validate symbols against BIST100 before loading
    2. Return standardized DataFrame format
    3. Handle missing data appropriately
    """
    
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, validator: BIST100Validator = None):
        """
        Initialize the data loader.
        
        Args:
            validator: BIST100Validator instance. Uses default if None.
        """
        self.validator = validator or get_validator()
    
    @abstractmethod
    def _load_raw(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Load raw data for a single symbol.
        
        Must be implemented by subclasses.
        
        Args:
            symbol: Stock symbol (already validated)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    def load(
        self,
        symbol: str,
        start_date: Union[date, str],
        end_date: Union[date, str]
    ) -> pd.DataFrame:
        """
        Load data for a single symbol with validation.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (date object or 'YYYY-MM-DD' string)
            end_date: End date
            
        Returns:
            Standardized DataFrame with columns:
            ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Validate BIST100 membership
        validated_symbol = self.validator.validate_symbol(symbol)
        
        # Parse dates
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        
        logger.info(f"Loading data for {validated_symbol} from {start} to {end}")
        
        # Load raw data
        df = self._load_raw(validated_symbol, start, end)
        
        # Standardize format
        df = self._standardize(df, validated_symbol)
        
        # Validate output
        self._validate_output(df)
        
        logger.info(f"Loaded {len(df)} bars for {validated_symbol}")
        return df
    
    def load_multiple(
        self,
        symbols: List[str],
        start_date: Union[date, str],
        end_date: Union[date, str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        # Validate all symbols first
        validated_symbols = self.validator.validate_symbols(symbols)
        
        results = {}
        for symbol in validated_symbols:
            try:
                results[symbol] = self.load(symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                # Continue with other symbols
        
        logger.info(f"Successfully loaded data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def _standardize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Standardize DataFrame to required format.
        
        Args:
            df: Raw DataFrame
            symbol: Symbol name
            
        Returns:
            Standardized DataFrame
        """
        df = df.copy()
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
                df = df.reset_index(drop=True)
        
        # Rename common column variations
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
            'Date': 'timestamp',
            'Datetime': 'timestamp',
        }
        df = df.rename(columns=column_map)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Use adjusted close if available
        if 'adj_close' in df.columns:
            df['close'] = df['adj_close']
        
        return df
    
    def _validate_output(self, df: pd.DataFrame) -> None:
        """
        Validate that output DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            DataLoaderError: If required columns are missing
        """
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise DataLoaderError(f"Missing required columns: {missing}")
        
        # Check for NaN values
        nan_counts = df[self.REQUIRED_COLUMNS].isna().sum()
        if nan_counts.any():
            logger.warning(f"Data contains NaN values:\n{nan_counts[nan_counts > 0]}")
    
    @staticmethod
    def _parse_date(d: Union[date, str]) -> date:
        """Parse date from string or date object."""
        if isinstance(d, str):
            return datetime.strptime(d, '%Y-%m-%d').date()
        return d


class YFinanceLoader(DataLoader):
    """
    Data loader using Yahoo Finance API.
    
    Uses the yfinance library to fetch BIST stock data.
    BIST stocks have the '.IS' suffix on Yahoo Finance.
    """
    
    def __init__(
        self,
        validator: BIST100Validator = None,
        suffix: str = ".IS"
    ):
        """
        Initialize Yahoo Finance loader.
        
        Args:
            validator: BIST100Validator instance
            suffix: Exchange suffix for BIST stocks (default: '.IS')
        """
        super().__init__(validator)
        self.suffix = suffix
    
    def _load_raw(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise DataLoaderError(
                "yfinance not installed. Run: pip install yfinance"
            )
        
        # Add suffix for BIST
        ticker_symbol = f"{symbol}{self.suffix}"
        
        logger.debug(f"Fetching {ticker_symbol} from Yahoo Finance")
        
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=True
        )
        
        if df.empty:
            raise DataLoaderError(
                f"No data returned for {ticker_symbol}. "
                "Check if the symbol is valid on Yahoo Finance."
            )
        
        return df


class CSVLoader(DataLoader):
    """
    Data loader for CSV files.
    
    Expects CSV files with standard OHLCV columns.
    Files should be named as '{symbol}.csv' in the data directory.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        validator: BIST100Validator = None
    ):
        """
        Initialize CSV loader.
        
        Args:
            data_dir: Directory containing CSV files
            validator: BIST100Validator instance
        """
        super().__init__(validator)
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise DataLoaderError(f"Data directory not found: {data_dir}")
    
    def _load_raw(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        # Try common file naming patterns
        patterns = [
            f"{symbol}.csv",
            f"{symbol.lower()}.csv",
            f"{symbol}_daily.csv",
        ]
        
        file_path = None
        for pattern in patterns:
            candidate = self.data_dir / pattern
            if candidate.exists():
                file_path = candidate
                break
        
        if file_path is None:
            raise DataLoaderError(
                f"No CSV file found for {symbol} in {self.data_dir}. "
                f"Tried patterns: {patterns}"
            )
        
        logger.debug(f"Loading {symbol} from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Filter date range
        df['Date'] = pd.to_datetime(df['Date'])
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        df = df[mask]
        
        return df


class SyntheticDataLoader(DataLoader):
    """
    Generates synthetic OHLCV data for testing and demos.
    
    Creates realistic-looking price data using geometric Brownian motion.
    """
    
    def __init__(
        self,
        validator: BIST100Validator = None,
        seed: int = 42
    ):
        """
        Initialize synthetic data loader.
        
        Args:
            validator: BIST100Validator instance
            seed: Random seed for reproducibility
        """
        super().__init__(validator)
        self.seed = seed
    
    def _load_raw(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data."""
        # Create reproducible seed based on symbol
        symbol_seed = self.seed + sum(ord(c) for c in symbol)
        np.random.seed(symbol_seed)
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        if n_days == 0:
            raise DataLoaderError(f"No trading days in range {start_date} to {end_date}")
        
        # Parameters for price simulation
        initial_price = np.random.uniform(20, 500)
        mu = 0.0002  # Daily drift
        sigma = 0.02  # Daily volatility
        
        # Generate returns using geometric Brownian motion
        returns = np.random.normal(mu, sigma, n_days)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        close = prices
        
        # High: close + random up move
        high_spread = np.abs(np.random.normal(0, 0.01, n_days)) * prices
        high = close + high_spread
        
        # Low: close - random down move  
        low_spread = np.abs(np.random.normal(0, 0.01, n_days)) * prices
        low = close - low_spread
        
        # Open: between low and high
        open_price = low + np.random.uniform(0.3, 0.7, n_days) * (high - low)
        
        # Volume: random with some autocorrelation
        base_volume = np.random.uniform(100000, 10000000)
        volume = base_volume * np.exp(np.cumsum(np.random.normal(0, 0.1, n_days)))
        volume = volume.astype(int)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        
        logger.debug(f"Generated {len(df)} synthetic bars for {symbol}")
        return df


def get_data_loader(
    source: str = "yfinance",
    **kwargs
) -> DataLoader:
    """
    Factory function to get appropriate data loader.
    
    Args:
        source: Data source type ('yfinance', 'csv', 'synthetic')
        **kwargs: Additional arguments for the loader
        
    Returns:
        DataLoader instance
    """
    loaders = {
        "yfinance": YFinanceLoader,
        "csv": CSVLoader,
        "synthetic": SyntheticDataLoader,
    }
    
    if source not in loaders:
        raise ValueError(f"Unknown data source: {source}. Available: {list(loaders.keys())}")
    
    return loaders[source](**kwargs)
