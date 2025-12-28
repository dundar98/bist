"""
Transformer Model for Time Series Trading.

Implements a Transformer-based architecture for trading signal prediction.
Uses Multi-Head Attention to capture long-range dependencies in market data.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .base import BaseModel


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the
    embeddings, so that the two can be summed.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]


class TransformerModel(BaseModel):
    """
    Transformer-Encoder based model for time series classification.
    
    Architecture:
        Input -> Linear Projection -> Positional Encoding ->
        Transformer Encoder Layers -> Global Average Pooling ->
        MLP Head -> Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        nhead: int = 4,
        dim_feedforward: int = 256,
        **kwargs
    ):
        """
        Initialize Transformer model.

        Args:
            input_size: Number of input features
            hidden_size: Dimension of the model (d_model)
            num_layers: Number of Transformer encoder layers
            dropout: Dropout rate
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network model
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.model_type = "transformer"
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # Feature projection to d_model size
        self.feature_projection = nn.Linear(input_size, hidden_size)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output Head
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters."""
        initrange = 0.1
        self.feature_projection.bias.data.zero_()
        self.feature_projection.weight.data.uniform_(-initrange, initrange)
        
        for p in self.fc.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_length, input_size)

        Returns:
            Probability tensor (batch_size, 1)
        """
        # Project features to hidden_size
        x = self.feature_projection(x) # (batch, seq, hidden)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer Encoder
        # Output: (batch, seq, hidden)
        x = self.transformer_encoder(x)
        
        # Global Average Pooling over sequence dimension
        # We process the entire sequence and aggregate information
        x = x.mean(dim=1) # (batch, hidden)
        
        # Output head
        x = self.dropout_layer(x)
        output = self.fc(x)
        
        return output

    def get_config(self) -> dict:
        """Return model configuration."""
        config = super().get_config()
        config.update({
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
        })
        return config
