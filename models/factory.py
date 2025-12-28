"""
Model Factory.

Provides factory function for creating models by type.
Enables easy swapping between different architectures.
"""

from typing import Dict, Type, Any

from .base import BaseModel
from .lstm_model import LSTMModel, GRUModel, CNNLSTMModel
from .transformer_model import TransformerModel
from .multitask_model import MultiTaskModel


# Registry of available model types
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "cnn_lstm": CNNLSTMModel,
    "transformer": TransformerModel,
    "multitask": MultiTaskModel,
}


def create_model(
    model_type: str,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    **kwargs
) -> BaseModel:
    """
    Factory function to create a model by type.
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'cnn_lstm')
        input_size: Number of input features
        hidden_size: Size of hidden layers
        num_layers: Number of layers
        dropout: Dropout rate
        **kwargs: Additional model-specific arguments
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_type is not registered
    """
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. Available: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_type]
    
    model = model_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs
    )
    
    return model


def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    Register a new model type.
    
    Args:
        name: Name to register the model under
        model_class: Model class (must inherit from BaseModel)
    """
    if not issubclass(model_class, BaseModel):
        raise TypeError(f"{model_class} must inherit from BaseModel")
    
    MODEL_REGISTRY[name] = model_class


def get_available_models() -> list:
    """Return list of available model types."""
    return list(MODEL_REGISTRY.keys())
