"""
Feature Engineering Module.

Computes technical indicators and transforms for trading signals.
All features are computed in a way that prevents future data leakage.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Computes technical indicators and features from OHLCV data.
    
    Key design principles:
    1. No future data leakage - all indicators use only past data
    2. Configurable parameters for all indicators
    3. Proper handling of NaN values from lookback periods
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        volatility_window: int = 20,
        sma_short: int = 10,
        sma_long: int = 50,
        bollinger_window: int = 20,
        bollinger_std: float = 2.0,
    ):
        """
        Initialize feature engine with indicator parameters.
        
        Args:
            rsi_period: Period for RSI calculation
            macd_fast: Fast EMA period for MACD
            macd_slow: Slow EMA period for MACD
            macd_signal: Signal line period for MACD
            atr_period: Period for ATR calculation
            volatility_window: Window for volatility calculation
            sma_short: Short-term SMA period
            sma_long: Long-term SMA period
            bollinger_window: Window for Bollinger Bands
            bollinger_std: Standard deviation multiplier for Bollinger Bands
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.volatility_window = volatility_window
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std
        
        # Track which features were computed
        self.feature_names: List[str] = []
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with original columns plus computed features
        """
        df = df.copy()
        
        # Log returns
        df = self._add_returns(df)
        
        # Volatility
        df = self._add_volatility(df)
        
        # RSI
        df = self._add_rsi(df)
        
        # MACD
        df = self._add_macd(df)
        
        # ATR
        df = self._add_atr(df)
        
        # Moving averages and trend
        df = self._add_trend_features(df)
        
        # Bollinger Bands
        df = self._add_bollinger_bands(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Price patterns
        df = self._add_pattern_features(df)
        
        # Store feature names (exclude original columns)
        original_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol'}
        self.feature_names = [c for c in df.columns if c not in original_cols]
        
        logger.info(f"Computed {len(self.feature_names)} features")
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        # Log returns (safer than simple returns for training)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Multi-period returns
        for period in [5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # Rolling volatility
        df['volatility'] = df['log_return'].rolling(
            window=self.volatility_window
        ).std() * np.sqrt(252)  # Annualized
        
        # Volatility ratio (current vs long-term)
        long_vol = df['log_return'].rolling(window=60).std()
        df['volatility_ratio'] = df['volatility'] / (long_vol * np.sqrt(252) + 1e-8)
        
        # Intraday volatility (high-low range)
        df['intraday_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Relative Strength Index."""
        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI normalized to [-1, 1]
        df['rsi_normalized'] = (df['rsi'] - 50) / 50
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Normalized MACD (relative to price)
        df['macd_normalized'] = df['macd'] / df['close']
        
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range."""
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Simple moving averages
        df['sma_short'] = df['close'].rolling(window=self.sma_short).mean()
        df['sma_long'] = df['close'].rolling(window=self.sma_long).mean()
        
        # Price relative to SMAs
        df['price_to_sma_short'] = df['close'] / df['sma_short'] - 1
        df['price_to_sma_long'] = df['close'] / df['sma_long'] - 1
        
        # SMA crossover signal
        df['sma_crossover'] = (df['sma_short'] > df['sma_long']).astype(int)
        
        # Trend strength (distance between SMAs)
        df['trend_strength'] = (df['sma_short'] - df['sma_long']) / df['sma_long']
        
        # Days since last trend change
        crossover_change = df['sma_crossover'].diff().abs()
        df['bars_since_crossover'] = crossover_change.groupby(
            (crossover_change != 0).cumsum()
        ).cumcount()
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands features."""
        sma = df['close'].rolling(window=self.bollinger_window).mean()
        std = df['close'].rolling(window=self.bollinger_window).std()
        
        df['bb_upper'] = sma + self.bollinger_std * std
        df['bb_lower'] = sma - self.bollinger_std * std
        df['bb_middle'] = sma
        
        # Position within bands (0 = lower, 1 = upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (
            df['bb_upper'] - df['bb_lower'] + 1e-8
        )
        
        # Bandwidth (volatility indicator)
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving average
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1)
        
        # On-balance volume (simplified)
        df['obv_change'] = np.where(
            df['close'] > df['close'].shift(1),
            df['volume'],
            np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)
        )
        df['obv'] = df['obv_change'].cumsum()
        df['obv_normalized'] = df['obv'] / (df['obv'].rolling(window=20).std() + 1e-8)
        
        # Volume price trend
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern-based features."""
        # Higher highs and lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Consecutive up/down days
        up_day = (df['close'] > df['close'].shift(1)).astype(int)
        df['consecutive_up'] = up_day.groupby((up_day != up_day.shift()).cumsum()).cumsum()
        df['consecutive_up'] = df['consecutive_up'] * up_day
        
        down_day = (df['close'] < df['close'].shift(1)).astype(int)
        df['consecutive_down'] = down_day.groupby((down_day != down_day.shift()).cumsum()).cumsum()
        df['consecutive_down'] = df['consecutive_down'] * down_day
        
        # Gap indicators
        df['gap_up'] = ((df['open'] > df['high'].shift(1)) & 
                        (df['open'] > df['close'].shift(1))).astype(int)
        df['gap_down'] = ((df['open'] < df['low'].shift(1)) & 
                          (df['open'] < df['close'].shift(1))).astype(int)
        
        # Distance from N-day high/low
        for period in [10, 20, 50]:
            df[f'dist_from_{period}d_high'] = (
                df['close'] / df['high'].rolling(window=period).max() - 1
            )
            df[f'dist_from_{period}d_low'] = (
                df['close'] / df['low'].rolling(window=period).min() - 1
            )
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of computed feature names."""
        return self.feature_names.copy()


class FeatureNormalizer:
    """
    Normalizes features for neural network input.
    
    Supports two modes:
    1. Rolling normalization: Uses rolling statistics (no future leakage)
    2. Training-fit normalization: Fits on training data only
    """
    
    def __init__(self, method: str = "rolling", window: int = 60):
        """
        Initialize normalizer.
        
        Args:
            method: "rolling" or "fit" normalization
            window: Window size for rolling normalization
        """
        self.method = method
        self.window = window
        self._means: Optional[pd.Series] = None
        self._stds: Optional[pd.Series] = None
        self._feature_columns: Optional[List[str]] = None
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str]) -> 'FeatureNormalizer':
        """
        Fit normalizer on training data.
        
        Only used when method='fit'.
        
        Args:
            df: Training DataFrame
            feature_columns: Columns to normalize
            
        Returns:
            self
        """
        self._feature_columns = feature_columns
        self._means = df[feature_columns].mean()
        self._stds = df[feature_columns].std()
        
        # Avoid division by zero
        self._stds = self._stds.replace(0, 1)
        
        logger.info(f"Fitted normalizer on {len(feature_columns)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using normalization.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        if self.method == "rolling":
            return self._rolling_normalize(df)
        elif self.method == "fit":
            return self._fit_normalize(df)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _rolling_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling z-score normalization."""
        if self._feature_columns is None:
            # Use all numeric columns except standard ones
            exclude = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol'}
            self._feature_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns 
                if c not in exclude
            ]
        
        for col in self._feature_columns:
            if col in df.columns:
                rolling_mean = df[col].rolling(window=self.window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=self.window, min_periods=1).std()
                
                # Z-score
                df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
                
                # Clip extreme values
                df[f'{col}_norm'] = df[f'{col}_norm'].clip(-5, 5)
        
        return df
    
    def _fit_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization using fitted statistics."""
        if self._means is None or self._stds is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        for col in self._feature_columns:
            if col in df.columns:
                df[f'{col}_norm'] = (df[col] - self._means[col]) / (self._stds[col] + 1e-8)
                df[f'{col}_norm'] = df[f'{col}_norm'].clip(-5, 5)
        
        return df
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, feature_columns)
        return self.transform(df)


def prepare_features(
    df: pd.DataFrame,
    feature_config: Optional[dict] = None,
    normalize: bool = True,
    normalization_method: str = "rolling",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to prepare all features.
    
    Args:
        df: Raw OHLCV DataFrame
        feature_config: Optional dict of feature parameters
        normalize: Whether to normalize features
        normalization_method: 'rolling' or 'fit'
        
    Returns:
        Tuple of (DataFrame with features, list of feature column names)
    """
    # Initialize feature engine
    config = feature_config or {}
    engine = FeatureEngine(**config)
    
    # Compute features
    df = engine.compute_all_features(df)
    feature_names = engine.get_feature_names()
    
    # Normalize if requested
    if normalize:
        normalizer = FeatureNormalizer(method=normalization_method)
        df = normalizer.transform(df)
        # Update feature names to use normalized versions
        feature_names = [f'{name}_norm' for name in feature_names]
    
    return df, feature_names
