"""
Data Splitter Module.

Provides time-series aware splitting strategies including
simple chronological splits and walk-forward validation.
"""

import logging
from dataclasses import dataclass
from typing import Generator, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Split:
    """Represents a single train/val/test split."""
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    fold: int = 0


class ChronologicalSplitter:
    """
    Simple chronological train/val/test split.
    
    Ensures no future data leakage by splitting purely by time.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(self, n_samples: int) -> Split:
        """
        Split indices into train/val/test.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            Split object with index ranges
        """
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        split = Split(
            train_start=0,
            train_end=train_end,
            val_start=train_end,
            val_end=val_end,
            test_start=val_end,
            test_end=n_samples,
            fold=0,
        )
        
        logger.info(
            f"Chronological split: train[0:{train_end}], "
            f"val[{train_end}:{val_end}], test[{val_end}:{n_samples}]"
        )
        
        return split
    
    def split_dataframe(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train/val/test.
        
        Args:
            df: DataFrame to split (assumed sorted by time)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        split = self.split(len(df))
        
        train_df = df.iloc[split.train_start:split.train_end].copy()
        val_df = df.iloc[split.val_start:split.val_end].copy()
        test_df = df.iloc[split.test_start:split.test_end].copy()
        
        return train_df, val_df, test_df


class WalkForwardSplitter:
    """
    Walk-forward validation splitter.
    
    Creates multiple splits with expanding training windows.
    This is the most realistic approach for time series.
    
    Example with 3 folds:
    Fold 1: Train[-------] Val[--] Test[--]
    Fold 2: Train[-----------] Val[--] Test[--]
    Fold 3: Train[---------------] Val[--] Test[--]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        expanding: bool = True,
        min_train_size: Optional[int] = None,
    ):
        """
        Initialize walk-forward splitter.
        
        Args:
            n_splits: Number of walk-forward folds
            train_ratio: Initial fraction for training (first fold)
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            expanding: If True, training window expands; if False, slides
            min_train_size: Minimum training samples (optional)
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.expanding = expanding
        self.min_train_size = min_train_size
    
    def split(self, n_samples: int) -> Generator[Split, None, None]:
        """
        Generate walk-forward splits.
        
        Args:
            n_samples: Total number of samples
            
        Yields:
            Split objects for each fold
        """
        # Calculate sizes
        val_size = int(n_samples * self.val_ratio)
        test_size = int(n_samples * self.test_ratio)
        fold_size = val_size + test_size
        
        # Initial training size
        initial_train_size = int(n_samples * self.train_ratio)
        
        # Remaining data for folds
        remaining = n_samples - initial_train_size
        
        # Adjust n_splits if necessary
        max_splits = remaining // fold_size
        actual_splits = min(self.n_splits, max_splits)
        
        if actual_splits < self.n_splits:
            logger.warning(
                f"Reduced splits from {self.n_splits} to {actual_splits} "
                f"due to data size"
            )
        
        for fold in range(actual_splits):
            if self.expanding:
                # Training window grows with each fold
                train_start = 0
                train_end = initial_train_size + fold * fold_size
            else:
                # Training window slides
                train_start = fold * fold_size
                train_end = initial_train_size + fold * fold_size
            
            val_start = train_end
            val_end = val_start + val_size
            test_start = val_end
            test_end = test_start + test_size
            
            # Ensure we don't exceed data bounds
            if test_end > n_samples:
                break
            
            # Check minimum training size
            if self.min_train_size and (train_end - train_start) < self.min_train_size:
                continue
            
            split = Split(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                fold=fold,
            )
            
            logger.info(
                f"Fold {fold}: train[{train_start}:{train_end}], "
                f"val[{val_start}:{val_end}], test[{test_start}:{test_end}]"
            )
            
            yield split
    
    def split_dataframe(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate walk-forward DataFrame splits.
        
        Args:
            df: DataFrame to split
            
        Yields:
            Tuples of (train_df, val_df, test_df)
        """
        for split in self.split(len(df)):
            train_df = df.iloc[split.train_start:split.train_end].copy()
            val_df = df.iloc[split.val_start:split.val_end].copy()
            test_df = df.iloc[split.test_start:split.test_end].copy()
            yield train_df, val_df, test_df


class PurgedCVSplitter:
    """
    Purged cross-validation splitter.
    
    Similar to walk-forward but with a gap between train and test
    to prevent information leakage from overlapping sequences.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 10,
    ):
        """
        Initialize purged CV splitter.
        
        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to skip between train and val/test
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, n_samples: int) -> Generator[Split, None, None]:
        """
        Generate purged CV splits.
        
        Args:
            n_samples: Total number of samples
            
        Yields:
            Split objects
        """
        fold_size = n_samples // (self.n_splits + 1)
        
        for fold in range(self.n_splits):
            test_start = (fold + 1) * fold_size
            test_end = test_start + fold_size
            
            # Validation is portion before test
            val_size = fold_size // 2
            val_start = test_start - val_size - self.purge_gap
            val_end = test_start - self.purge_gap
            
            # Training is everything before validation
            train_start = 0
            train_end = val_start - self.purge_gap
            
            if train_end <= 0:
                continue
            
            split = Split(
                train_start=train_start,
                train_end=train_end,
                val_start=max(0, val_start),
                val_end=max(0, val_end),
                test_start=test_start,
                test_end=min(test_end, n_samples),
                fold=fold,
            )
            
            yield split


def split_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
    date_column: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by specific dates.
    
    Args:
        df: DataFrame with date column
        train_end_date: End date for training (YYYY-MM-DD)
        val_end_date: End date for validation (YYYY-MM-DD)
        date_column: Name of date column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)
    
    train_df = df[df[date_column] <= train_end]
    val_df = df[(df[date_column] > train_end) & (df[date_column] <= val_end)]
    test_df = df[df[date_column] > val_end]
    
    logger.info(
        f"Date split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    return train_df, val_df, test_df
