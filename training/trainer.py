"""
Training Module.

Implements the training loop with early stopping, metrics tracking,
and proper handling of class imbalance.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm

from models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    
    def to_dict(self) -> Dict:
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc': self.auc,
        }


@dataclass
class TrainingHistory:
    """Training history tracker."""
    metrics: List[TrainingMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    
    def add(self, metrics: TrainingMetrics) -> bool:
        """
        Add metrics and check if this is the best epoch.
        
        Returns:
            True if this is the best epoch so far
        """
        self.metrics.append(metrics)
        
        if metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
            self.best_epoch = metrics.epoch
            return True
        return False
    
    def to_dataframe(self):
        """Convert history to DataFrame."""
        import pandas as pd
        return pd.DataFrame([m.to_dict() for m in self.metrics])


class EarlyStopping:
    """
    Early stopping handler.
    
    Stops training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Model trainer with early stopping and metrics tracking.
    """
    
    def __init__(
        self,
        model: BaseModel,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        use_class_weights: bool = True,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate
            weight_decay: L2 regularization
            max_grad_norm: Gradient clipping threshold
            use_class_weights: Whether to use class weights for imbalanced data
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.use_class_weights = use_class_weights
        
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Loss function (will be configured with class weights if needed)
        self.criterion = nn.BCELoss()
        
        # Training state
        self.history = TrainingHistory()
        self.current_epoch = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[TrainingMetrics], None]] = None,
    ) -> TrainingHistory:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            class_weights: Optional class weights for BCE loss
            callback: Optional callback called after each epoch
            
        Returns:
            Training history
        """
        # Set up loss function with class weights
        if class_weights is not None and self.use_class_weights:
            # BCEWithLogitsLoss doesn't support class weights directly
            # We'll use a weighted BCE loss
            pos_weight = class_weights.to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # Model already has sigmoid, so we need BCELoss with manual weighting
            # Let's keep BCELoss and weight in training loop instead
            self.criterion = nn.BCELoss(reduction='none')
            self._class_weights = class_weights.to(self.device)
        else:
            self.criterion = nn.BCELoss()
            self._class_weights = None
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Best model state
        best_state = None
        
        logger.info(f"Starting training for up to {epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_preds, val_labels = self._validate(val_loader)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                epoch, train_loss, val_loss, val_preds, val_labels
            )
            
            # Update history
            is_best = self.history.add(metrics)
            
            if is_best:
                best_state = self.model.state_dict().copy()
                if self.checkpoint_dir:
                    self._save_checkpoint(f'best_model.pt')
            
            # Logging
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"AUC: {metrics.auc:.4f} | "
                f"F1: {metrics.f1:.4f}"
                + (" *" if is_best else "")
            )
            
            # Callback
            if callback:
                callback(metrics)
            
            # Early stopping check
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        elapsed = time.time() - start_time
        logger.info(
            f"Training completed in {elapsed/60:.1f} minutes. "
            f"Best epoch: {self.history.best_epoch}"
        )
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for features, labels in tqdm(loader, desc="Training", leave=False):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Calculate loss with optional class weighting
            if self._class_weights is not None:
                loss = self.criterion(outputs, labels)
                weights = torch.where(
                    labels == 1,
                    self._class_weights,
                    torch.ones_like(labels)
                )
                loss = (loss * weights).mean()
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
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
        """Run validation and return loss, predictions, and labels."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                
                if self._class_weights is not None:
                    loss = self.criterion(outputs, labels).mean()
                else:
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                n_batches += 1
                
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        
        return total_loss / n_batches, preds, labels
    
    def _calculate_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> TrainingMetrics:
        """Calculate all metrics."""
        # Flatten arrays
        preds_flat = predictions.flatten()
        labels_flat = labels.flatten()
        
        # Binary predictions at 0.5 threshold
        binary_preds = (preds_flat >= 0.5).astype(int)
        binary_labels = labels_flat.astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(binary_labels, binary_preds)
        
        # Handle edge cases where one class is missing
        try:
            precision = precision_score(binary_labels, binary_preds, zero_division=0)
            recall = recall_score(binary_labels, binary_preds, zero_division=0)
            f1 = f1_score(binary_labels, binary_preds, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        try:
            auc = roc_auc_score(binary_labels, preds_flat)
        except:
            auc = 0.5  # Default for when only one class is present
        
        return TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
        )
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        logger.debug(f"Saved checkpoint to {path}")
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of metrics
        """
        val_loss, preds, labels = self._validate(test_loader)
        
        metrics = self._calculate_metrics(
            epoch=-1,
            train_loss=0,
            val_loss=val_loss,
            predictions=preds,
            labels=labels,
        )
        
        # Add confusion matrix analysis
        binary_preds = (preds.flatten() >= 0.5).astype(int)
        binary_labels = labels.flatten().astype(int)
        cm = confusion_matrix(binary_labels, binary_preds)
        
        results = metrics.to_dict()
        results['confusion_matrix'] = cm.tolist()
        
        # Calculate profit-related metrics
        results['positive_rate'] = binary_preds.mean()
        results['actual_positive_rate'] = binary_labels.mean()
        
        return results


def train_model(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: str = 'cpu',
    checkpoint_dir: Optional[str] = None,
) -> Tuple[BaseModel, TrainingHistory, Optional[Dict]]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data
        val_loader: Validation data
        test_loader: Optional test data
        epochs: Max epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: Device to use
        checkpoint_dir: Where to save checkpoints
        
    Returns:
        Tuple of (trained_model, history, test_metrics)
    """
    # Support for Multi-Task Model
    if model.model_type == 'multitask':
        from .multitask_trainer import MultiTaskTrainer
        trainer = MultiTaskTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )
    
    # Get class weights from training data
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights()
    else:
        class_weights = None
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        class_weights=class_weights,
    )
    
    test_metrics = None
    if test_loader:
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test metrics: {test_metrics}")
    
    return model, history, test_metrics
