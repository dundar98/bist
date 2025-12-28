"""
Base Model Interface.

Defines the abstract interface that all trading models must implement.
This enables easy swapping between LSTM, GRU, CNN, Transformer, etc.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all trading models.
    
    All models must:
    1. Accept sequences of features as input
    2. Output probabilities in [0, 1]
    3. Implement save/load functionality
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        """
        Initialize base model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of recurrent/hidden layers
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Will be set by subclasses
        self.model_type: str = "base"
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor of probabilities with shape (batch_size, 1)
        """
        pass
    
    def predict_proba(
        self,
        x: torch.Tensor,
        return_numpy: bool = True
    ) -> np.ndarray:
        """
        Predict probabilities for input sequences.
        
        Args:
            x: Input tensor
            return_numpy: If True, return numpy array; else torch tensor
            
        Returns:
            Probabilities as numpy array or tensor
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        
        if return_numpy:
            return probs.cpu().numpy()
        return probs
    
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            x: Input tensor
            threshold: Probability threshold for positive class
            
        Returns:
            Binary predictions as numpy array
        """
        probs = self.predict_proba(x, return_numpy=True)
        return (probs >= threshold).astype(int)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.get_config(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            device: Device to load model to
            
        Returns:
            Loaded model instance
        """
        # Use weights_only=False for compatibility with model config dict
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        
        # Import here to avoid circular dependency
        from .factory import create_model
        
        # Extract required params and pass rest as kwargs
        model_type = config.pop("model_type")
        input_size = config.pop("input_size")
        hidden_size = config.pop("hidden_size", 128)
        num_layers = config.pop("num_layers", 2)
        dropout = float(config.pop("dropout", 0.3))
        
        model = create_model(
            model_type=model_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            **config  # Pass remaining config as kwargs
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        
        return model
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models for robust predictions.
    
    Combines predictions from multiple base models using
    averaging or weighted averaging.
    """
    
    def __init__(
        self,
        models: list,
        weights: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of BaseModel instances
            weights: Optional weights for each model (must sum to 1)
        """
        # Use first model's config for base init
        first = models[0]
        super().__init__(
            input_size=first.input_size,
            hidden_size=first.hidden_size,
            num_layers=first.num_layers,
            dropout=first.dropout,
        )
        
        self.models = nn.ModuleList(models)
        self.model_type = "ensemble"
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack and weight
        stacked = torch.stack(predictions, dim=0)  # (n_models, batch, 1)
        weights = self.weights.view(-1, 1, 1)
        
        return (stacked * weights).sum(dim=0)
