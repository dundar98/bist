"""Training module for the BIST100 trading system."""

from .dataset import (
    TradingDataset,
    MultiStockDataset,
    create_data_loaders,
)
from .splitter import (
    Split,
    ChronologicalSplitter,
    WalkForwardSplitter,
    PurgedCVSplitter,
    split_by_date,
)
from .trainer import (
    TrainingMetrics,
    TrainingHistory,
    EarlyStopping,
    Trainer,
    train_model,
)

__all__ = [
    # Dataset
    "TradingDataset",
    "MultiStockDataset",
    "create_data_loaders",
    # Splitter
    "Split",
    "ChronologicalSplitter",
    "WalkForwardSplitter",
    "PurgedCVSplitter",
    "split_by_date",
    # Trainer
    "TrainingMetrics",
    "TrainingHistory",
    "EarlyStopping",
    "Trainer",
    "train_model",
]
