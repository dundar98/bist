"""
Unit tests for models module.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model, LSTMModel, GRUModel, get_available_models


class TestModelCreation:
    """Tests for model factory."""
    
    def test_create_lstm_model(self):
        """Test LSTM model creation."""
        model = create_model(
            model_type="lstm",
            input_size=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
        )
        
        assert isinstance(model, LSTMModel)
        assert model.model_type == "lstm"
    
    def test_create_gru_model(self):
        """Test GRU model creation."""
        model = create_model(
            model_type="gru",
            input_size=20,
            hidden_size=64,
        )
        
        assert isinstance(model, GRUModel)
        assert model.model_type == "gru"
    
    def test_invalid_model_type_raises(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            create_model(model_type="invalid_model", input_size=20)
    
    def test_get_available_models(self):
        """Test getting available model types."""
        available = get_available_models()
        
        assert "lstm" in available
        assert "gru" in available
        assert isinstance(available, list)


class TestLSTMModel:
    """Tests for LSTM model."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return LSTMModel(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
        )
    
    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        batch_size = 8
        seq_len = 60
        input_size = 20
        
        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)
        
        assert output.shape == (batch_size, 1)
    
    def test_output_probability_range(self, model):
        """Test that output is a valid probability."""
        x = torch.randn(8, 60, 20)
        output = model(x)
        
        # Should be between 0 and 1 (sigmoid output)
        assert (output >= 0).all()
        assert (output <= 1).all()
    
    def test_predict_proba(self, model):
        """Test predict_proba method."""
        x = torch.randn(4, 60, 20)
        probs = model.predict_proba(x)
        
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (4, 1)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_predict_binary(self, model):
        """Test binary prediction."""
        x = torch.randn(4, 60, 20)
        preds = model.predict(x, threshold=0.5)
        
        assert isinstance(preds, np.ndarray)
        assert set(np.unique(preds)).issubset({0, 1})
    
    def test_count_parameters(self, model):
        """Test parameter counting."""
        params = model.count_parameters()
        
        assert params > 0
        assert isinstance(params, int)
    
    def test_get_config(self, model):
        """Test config export."""
        config = model.get_config()
        
        assert config['model_type'] == 'lstm'
        assert config['input_size'] == 20
        assert config['hidden_size'] == 64
        assert config['num_layers'] == 2


class TestModelSaveLoad:
    """Tests for model serialization."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading model."""
        # Create and train briefly
        model = LSTMModel(input_size=10, hidden_size=32, num_layers=1)
        
        # Save
        save_path = tmp_path / "test_model.pt"
        model.save(str(save_path))
        
        assert save_path.exists()
        
        # Load
        loaded = LSTMModel.load(str(save_path))
        
        assert loaded.input_size == model.input_size
        assert loaded.hidden_size == model.hidden_size
    
    def test_loaded_model_produces_same_output(self, tmp_path):
        """Test that loaded model produces same predictions."""
        model = LSTMModel(input_size=10, hidden_size=32, num_layers=1)
        model.eval()
        
        # Get prediction before saving
        x = torch.randn(2, 30, 10)
        with torch.no_grad():
            pred_before = model(x).numpy()
        
        # Save and load
        save_path = tmp_path / "test_model.pt"
        model.save(str(save_path))
        loaded = LSTMModel.load(str(save_path))
        loaded.eval()
        
        # Get prediction after loading
        with torch.no_grad():
            pred_after = loaded(x).numpy()
        
        np.testing.assert_array_almost_equal(pred_before, pred_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
