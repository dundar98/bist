"""
LSTM Model for Trading Signal Prediction.

Implements an LSTM-based architecture for binary classification
of trading signals. Outputs probability in [0, 1].
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .base import BaseModel


class LSTMModel(BaseModel):
    """
    LSTM-based model for trading signal prediction.
    
    Architecture:
        Input(features) → LSTM(hidden, layers) → Dropout → FC → Sigmoid
    
    Key design choices:
    1. Unidirectional LSTM to maintain causality
    2. Dropout for regularization
    3. Single sigmoid output for probability
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = False,
        **kwargs
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: If True, use bidirectional LSTM (NOT recommended for trading)
            use_attention: If True, add attention mechanism
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.model_type = "lstm"
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Direction multiplier
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Optional attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
        
        # Output layers
        lstm_output_size = hidden_size * self.num_directions
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            Probability tensor (batch_size, 1)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        # lstm_out: (batch, seq_len, hidden * directions)
        # h_n: (layers * directions, batch, hidden)
        
        if self.use_attention:
            # Attention mechanism
            attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                context = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                context = h_n[-1]  # (batch, hidden)
        
        # Apply dropout and fully connected layers
        out = self.dropout_layer(context)
        out = self.fc(out)
        
        return out
    
    def get_config(self) -> dict:
        """Return model configuration."""
        config = super().get_config()
        config.update({
            "bidirectional": self.bidirectional,
            "use_attention": self.use_attention,
        })
        return config


class GRUModel(BaseModel):
    """
    GRU-based model for trading signal prediction.
    
    Similar to LSTM but with fewer parameters (often trains faster).
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        **kwargs
    ):
        """
        Initialize GRU model.
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: If True, use bidirectional GRU
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.model_type = "gru"
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Output layers
        gru_output_size = hidden_size * self.num_directions
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            Probability tensor (batch_size, 1)
        """
        # GRU forward
        gru_out, h_n = self.gru(x, hidden)
        
        # Use last hidden state
        if self.bidirectional:
            context = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            context = h_n[-1]
        
        # Apply dropout and FC
        out = self.dropout_layer(context)
        out = self.fc(out)
        
        return out


class CNNLSTMModel(BaseModel):
    """
    Hybrid CNN-LSTM model.
    
    Uses 1D CNN to extract local patterns, then LSTM for sequential modeling.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        cnn_filters: int = 64,
        kernel_size: int = 3,
        **kwargs
    ):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            cnn_filters: Number of CNN filters
            kernel_size: CNN kernel size
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.model_type = "cnn_lstm"
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        
        # CNN layers (1D convolution along sequence)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
        )
        
        # LSTM after CNN
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Probability tensor (batch_size, 1)
        """
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN
        cnn_out = self.cnn(x)
        
        # Back to (batch, seq_len, features) for LSTM
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # Use last hidden state
        out = self.dropout_layer(h_n[-1])
        out = self.fc(out)
        
        return out
