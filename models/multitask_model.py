"""
Multi-Task Learning Model.

Predicts both:
1. Market Direction (Classification): Up/Down
2. Market Volatility (Regression): Future volatility magnitude

Benefits:
- Feature extraction becomes more robust by solving two related tasks
- Volatility prediction helps in risk sizing and dynamic thresholds
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from .base import BaseModel
from .lstm_model import LSTMModel


class MultiTaskModel(BaseModel):
    """
    Multi-Task Learning Model.
    
    Architecture:
        Input -> Shared Backbone (LSTM/GRU) -> Features
                                            |-> Direction Head (Sigmoid)
                                            |-> Volatility Head (Softplus)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        backbone_type: str = 'lstm',
        **kwargs
    ):
        """
        Initialize Multi-Task model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden size of backbone
            num_layers: Number of layers in backbone
            dropout: Dropout rate
            backbone_type: One of 'lstm', 'gru'
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.model_type = "multitask"
        self.backbone_type = backbone_type
        
        # Shared Backbone
        if backbone_type == 'lstm':
            self.backbone = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif backbone_type == 'gru':
            self.backbone = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
            
        self.dropout_layer = nn.Dropout(dropout)
        
        # Head 1: Direction (Binary Classification)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Volatility (Regression)
        # Volatility is always positive, so we use Softplus or ReLU
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize heads
        for m in self.direction_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.volatility_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (direction_prob, volatility_est)
            - direction_prob: (batch, 1) probability of up move
            - volatility_est: (batch, 1) estimated volatility
        """
        # Backbone
        out, _ = self.backbone(x)
        
        # Use last hidden state
        features = out[:, -1, :]
        features = self.dropout_layer(features)
        
        # Heads
        direction_prob = self.direction_head(features)
        volatility_est = self.volatility_head(features)
        
        return direction_prob, volatility_est

    def get_config(self) -> Dict[str, Any]:
        """Return config."""
        config = super().get_config()
        config.update({
            "backbone_type": self.backbone_type
        })
        return config
