"""
Dataset Module.

Provides PyTorch datasets for training with proper labeling.
Handles sequence creation and label generation.
"""

import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading signal prediction.
    
    Creates sequences of features with corresponding labels.
    Labels are generated based on future price movement.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        lookback: int = 60,
        label_threshold: float = 0.02,
        label_horizon: int = 5,
        transform=None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with features and price data
            feature_columns: List of feature column names to use
            lookback: Number of bars for each sequence
            label_threshold: Minimum % return for positive label
            label_horizon: Number of bars to look ahead for labeling
            transform: Optional transform to apply to sequences
        """
        self.lookback = lookback
        self.label_threshold = label_threshold
        self.label_horizon = label_horizon
        self.transform = transform
        self.feature_columns = feature_columns
        
        # Store original data info
        self.symbols = data['symbol'].unique().tolist() if 'symbol' in data.columns else ['UNKNOWN']
        
        # Generate labels
        data = self._generate_labels(data.copy())
        
        # Extract features and labels
        self.features, self.labels, self.timestamps, self.symbol_ids = self._prepare_data(data)
        
        logger.info(
            f"Created dataset with {len(self)} samples, "
            f"label distribution: {self.labels.mean():.2%} positive"
        )
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels based on future price movement.
        
        Label = 1 if max(close[t+1:t+horizon]) / close[t] - 1 >= threshold
        Label = 0 otherwise
        """
        # Calculate forward maximum return within horizon
        close = df['close'].values
        n = len(close)
        
        labels = np.zeros(n, dtype=np.float32)
        
        for i in range(n - self.label_horizon):
            # Look at next horizon bars
            future_prices = close[i + 1: i + 1 + self.label_horizon]
            if len(future_prices) > 0:
                max_future = np.max(future_prices)
                max_return = max_future / close[i] - 1
                labels[i] = 1.0 if max_return >= self.label_threshold else 0.0
        
        df['label'] = labels
        
        # Mark last horizon rows as invalid (no future data)
        df['valid_label'] = True
        df.loc[df.index[-self.label_horizon:], 'valid_label'] = False
        
        return df
    
    def _prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences from DataFrame.
        
        Returns:
            features: Array of shape (n_samples, lookback, n_features)
            labels: Array of shape (n_samples,)
            timestamps: Array of timestamps for each sample
            symbol_ids: Array of symbol IDs for each sample
        """
        # Filter valid rows (have labels and enough history)
        df = df.copy()
        
        # Drop rows with NaN in features
        df = df.dropna(subset=self.feature_columns)
        
        # Only use valid labels
        df = df[df['valid_label'] == True].copy()
        
        # Need at least lookback rows
        if len(df) < self.lookback:
            raise ValueError(
                f"Not enough data: {len(df)} rows, need at least {self.lookback}"
            )
        
        # Extract feature matrix
        feature_data = df[self.feature_columns].values.astype(np.float32)
        label_data = df['label'].values.astype(np.float32)
        
        # Handle timestamps
        if 'timestamp' in df.columns:
            timestamp_data = df['timestamp'].values
        else:
            timestamp_data = np.arange(len(df))
        
        # Handle symbols
        if 'symbol' in df.columns:
            symbol_map = {s: i for i, s in enumerate(self.symbols)}
            symbol_data = df['symbol'].map(symbol_map).values
        else:
            symbol_data = np.zeros(len(df))
        
        # Create sequences
        n_samples = len(feature_data) - self.lookback + 1
        n_features = len(self.feature_columns)
        
        features = np.zeros((n_samples, self.lookback, n_features), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.float32)
        timestamps = np.empty(n_samples, dtype=object)
        symbol_ids = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(n_samples):
            features[i] = feature_data[i:i + self.lookback]
            labels[i] = label_data[i + self.lookback - 1]
            timestamps[i] = timestamp_data[i + self.lookback - 1]
            symbol_ids[i] = symbol_data[i + self.lookback - 1]
        
        return features, labels, timestamps, symbol_ids
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label) as tensors
        """
        features = torch.from_numpy(self.features[idx])
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.
        
        Returns:
            Tensor with weight for positive class
        """
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return torch.tensor([1.0])
        
        # Weight positive class higher if under-represented
        weight = n_neg / n_pos
        return torch.tensor([weight], dtype=torch.float32)
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Get per-sample weights for weighted sampling.
        
        Returns:
            Array of sample weights
        """
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return np.ones(len(self.labels))
        
        weights = np.where(self.labels == 1, n_neg / n_pos, 1.0)
        return weights / weights.sum() * len(weights)


class MultiStockDataset(Dataset):
    """
    Dataset that combines multiple stocks.
    
    Properly handles data from multiple symbols while maintaining
    time ordering within each symbol.
    """
    
    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        feature_columns: List[str],
        lookback: int = 60,
        label_threshold: float = 0.02,
        label_horizon: int = 5,
    ):
        """
        Initialize multi-stock dataset.
        
        Args:
            stock_data: Dictionary mapping symbol to DataFrame
            feature_columns: Feature columns to use
            lookback: Sequence length
            label_threshold: Label threshold
            label_horizon: Label horizon
        """
        self.lookback = lookback
        self.label_threshold = label_threshold
        self.label_horizon = label_horizon
        self.feature_columns = feature_columns
        self.symbols = list(stock_data.keys())
        
        # Combine all data
        all_features = []
        all_labels = []
        all_symbols = []
        
        for symbol, df in stock_data.items():
            df = df.copy()
            df['symbol'] = symbol
            
            try:
                dataset = TradingDataset(
                    data=df,
                    feature_columns=feature_columns,
                    lookback=lookback,
                    label_threshold=label_threshold,
                    label_horizon=label_horizon,
                )
                
                all_features.append(dataset.features)
                all_labels.append(dataset.labels)
                all_symbols.extend([symbol] * len(dataset))
                
            except ValueError as e:
                logger.warning(f"Skipping {symbol}: {e}")
        
        if not all_features:
            raise ValueError("No valid data for any symbol")
        
        self.features = np.concatenate(all_features, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.symbol_list = all_symbols
        
        logger.info(
            f"Created multi-stock dataset with {len(self)} samples "
            f"from {len(self.symbols)} stocks"
        )
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.features[idx])
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return features, label
    
    def get_class_weights(self) -> torch.Tensor:
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor([1.0])
        return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    weighted_sampling: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        weighted_sampling: If True, use weighted sampling for imbalanced classes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training loader with optional weighted sampling
    if weighted_sampling and hasattr(train_dataset, 'get_sample_weights'):
        weights = train_dataset.get_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    # Validation loader (no shuffle)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Test loader
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader
