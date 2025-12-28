"""
Unit tests for the data module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.bist100_validator import (
    BIST100Validator,
    BIST100ValidationError,
    validate_bist100,
)
from data.loader import SyntheticDataLoader
from data.features import FeatureEngine


class TestBIST100Validator:
    """Tests for BIST100 validation."""
    
    def test_valid_symbol(self):
        """Test that valid BIST100 symbols pass validation."""
        validator = BIST100Validator()
        
        assert validator.is_valid_symbol("THYAO")
        assert validator.is_valid_symbol("GARAN")
        assert validator.is_valid_symbol("AKBNK")
    
    def test_invalid_symbol_raises(self):
        """Test that invalid symbols raise exception."""
        validator = BIST100Validator()
        
        with pytest.raises(BIST100ValidationError):
            validator.validate_symbol("AAPL")  # US stock
        
        with pytest.raises(BIST100ValidationError):
            validator.validate_symbol("FAKE123")
    
    def test_normalize_symbol(self):
        """Test symbol normalization."""
        validator = BIST100Validator()
        
        # Should handle .IS suffix
        assert validator.is_valid_symbol("THYAO.IS")
        assert validator.is_valid_symbol("thyao")  # lowercase
    
    def test_batch_validation(self):
        """Test batch symbol validation."""
        validator = BIST100Validator()
        
        valid_symbols = ["THYAO", "GARAN", "AKBNK"]
        result = validator.validate_symbols(valid_symbols)
        assert len(result) == 3
    
    def test_batch_validation_with_invalid(self):
        """Test that batch validation fails with any invalid symbol."""
        validator = BIST100Validator()
        
        mixed_symbols = ["THYAO", "AAPL", "GARAN"]
        
        with pytest.raises(BIST100ValidationError):
            validator.validate_symbols(mixed_symbols)
    
    def test_filter_valid_symbols(self):
        """Test filtering returns only valid symbols."""
        validator = BIST100Validator()
        
        mixed = ["THYAO", "AAPL", "GARAN", "MSFT"]
        filtered = validator.filter_valid_symbols(mixed)
        
        assert filtered == ["THYAO", "GARAN"]


class TestSyntheticDataLoader:
    """Tests for synthetic data loader."""
    
    def test_load_creates_valid_data(self):
        """Test that synthetic data has correct format."""
        validator = BIST100Validator()
        loader = SyntheticDataLoader(validator=validator)
        
        df = loader.load("THYAO", date(2020, 1, 1), date(2021, 1, 1))
        
        # Check required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            assert col in df.columns
        
        # Check data validity
        assert len(df) > 0
        assert (df['high'] >= df['low']).all()
        assert (df['close'] > 0).all()
    
    def test_load_rejects_non_bist100(self):
        """Test that loader rejects non-BIST100 symbols."""
        validator = BIST100Validator()
        loader = SyntheticDataLoader(validator=validator)
        
        with pytest.raises(BIST100ValidationError):
            loader.load("AAPL", date(2020, 1, 1), date(2021, 1, 1))


class TestFeatureEngine:
    """Tests for feature engineering."""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2
        open_price = (close + high) / 2
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n, freq='D'),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(100000, 1000000, n),
        })
    
    def test_compute_all_features(self, sample_ohlcv):
        """Test that all features are computed."""
        engine = FeatureEngine()
        df = engine.compute_all_features(sample_ohlcv)
        
        # Check that features were added
        assert len(df.columns) > len(sample_ohlcv.columns)
        
        # Check specific features exist
        assert 'log_return' in df.columns
        assert 'rsi' in df.columns
        assert 'macd' in df.columns
        assert 'atr' in df.columns
    
    def test_feature_names_returned(self, sample_ohlcv):
        """Test that feature names are tracked."""
        engine = FeatureEngine()
        engine.compute_all_features(sample_ohlcv)
        
        names = engine.get_feature_names()
        assert len(names) > 0
        assert 'log_return' in names
    
    def test_rsi_bounds(self, sample_ohlcv):
        """Test that RSI is within valid bounds."""
        engine = FeatureEngine()
        df = engine.compute_all_features(sample_ohlcv)
        
        # RSI should be between 0 and 100
        rsi = df['rsi'].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
