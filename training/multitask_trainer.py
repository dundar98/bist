"""
Multi-Task Trainer.

Specialized trainer for Multi-Task Learning model.
Handles dual outputs (classification + regression) and combined loss.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .trainer import Trainer, TrainingMetrics
from models.base import BaseModel


class MultiTaskTrainer(Trainer):
    """
    Trainer for Multi-Task Model.
    
    Optimizes for both direction (BCE) and volatility (MSE).
    """
    
    def __init__(
        self,
        model: BaseModel,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        volatility_weight: float = 0.5,
        **kwargs
    ):
        """
        Initialize Multi-Task trainer.
        
        Args:
            model: MultiTaskModel
            volatility_weight: Weight for volatility loss term
        """
        super().__init__(
            model=model,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
        self.volatility_weight = volatility_weight
        
        # Loss functions
        self.direction_criterion = nn.BCELoss()
        self.volatility_criterion = nn.MSELoss()
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(loader, desc="Training", leave=False):
            # Unpack batch (features, labels, volatilities)
            features = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            vol_targets = batch[2].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            dir_pred, vol_pred = self.model(features)
            
            # Calculate losses
            # Direction loss
            if self._class_weights is not None:
                # Manual weighting for BCE
                raw_loss = self.direction_criterion(dir_pred, labels)
                weights = torch.where(
                    labels == 1,
                    self._class_weights,
                    torch.ones_like(labels)
                )
                dir_loss = (raw_loss * weights).mean()
            else:
                dir_loss = self.direction_criterion(dir_pred, labels)
            
            # Volatility loss (ensure shapes match)
            vol_loss = self.volatility_criterion(vol_pred.squeeze(), vol_targets.squeeze())
            
            # Combined loss
            loss = dir_loss + self.volatility_weight * vol_loss
            
            # Backward
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / n_batches
    
    def _validate(
        self,
        loader: DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Run validation.
        
        Returns:
            loss, direction_predictions, labels
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                features = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                vol_targets = batch[2].to(self.device)
                
                dir_pred, vol_pred = self.model(features)
                
                # Direction loss
                dir_loss = self.direction_criterion(dir_pred, labels)
                
                # Volatility loss
                vol_loss = self.volatility_criterion(vol_pred.squeeze(), vol_targets.squeeze())
                
                loss = dir_loss + self.volatility_weight * vol_loss
                
                total_loss += loss.item()
                n_batches += 1
                
                all_preds.append(dir_pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        
        return total_loss / n_batches, preds, labels
