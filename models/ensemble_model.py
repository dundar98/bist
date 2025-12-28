"""
Ensemble Model.

Combines predictions from multiple models using:
1. Simple Averaging
2. Weighted Averaging (Learnable or Manual)
3. Stacking (Meta-learner)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union

from .base import BaseModel


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models.
    """
    
    def __init__(
        self,
        models: List[BaseModel],
        mode: str = 'average',  # 'average', 'weighted', 'voting'
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of loaded models
            mode: Ensemble mode
            weights: Optional manual weights for 'weighted' mode
        """
        # Initialize with config from first model
        super().__init__(
            input_size=models[0].config['input_size'],
            hidden_size=models[0].config['hidden_size']
        )
        
        self.models = nn.ModuleList(models)
        self.mode = mode
        self.loss_fn = nn.BCELoss()
        
        if mode == 'weighted':
            if weights is not None:
                self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
            else:
                # Learnable weights
                self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Aggregated probability
        """
        predictions = []
        
        # Get predictions from all models
        for model in self.models:
            model.eval() # Base models usually frozen in ensemble
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)
            
        # Stack predictions: (batch, n_models)
        preds_stack = torch.cat(predictions, dim=1)
        
        if self.mode == 'average':
            return preds_stack.mean(dim=1, keepdim=True)
            
        elif self.mode == 'weighted':
            # Normalize weights to sum to 1 using Softmax
            w = torch.softmax(self.weights, dim=0)
            # Weighted sum
            weighted_preds = (preds_stack * w).sum(dim=1, keepdim=True)
            return weighted_preds
            
        elif self.mode == 'voting':
            # Hard voting (>0.5 counts as 1)
            votes = (preds_stack > 0.5).float().mean(dim=1, keepdim=True)
            return votes
            
        else:
            return preds_stack.mean(dim=1, keepdim=True)
    
    def train_weights(
        self,
        loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 0.01
    ):
        """
        Train ensemble weights (if mode='weighted' and weights learnable).
        """
        if self.mode != 'weighted' or not self.weights.requires_grad:
            return
            
        optimizer = torch.optim.Adam([self.weights], lr=lr)
        
        print(f"Training ensemble weights for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for features, labels in loader:
                optimizer.zero_grad()
                
                output = self.forward(features)
                loss = self.loss_fn(output, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch}: Loss {total_loss:.4f}")
