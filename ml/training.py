"""Lightweight walk-forward style ML training utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import FeatureValue, ModelRun, Prediction, PriceBar, Timeframe

FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "macd_signal",
    "atr_pct",
    "volatility",
    "volume_ratio",
    "trend_score",
    "volume_score",
    "momentum_score",
]


@dataclass(frozen=True)
class TrainingResult:
    model_run: ModelRun
    metrics: dict[str, float | int | str]


def _price_frame(db: Session, timeframe: Timeframe) -> pd.DataFrame:
    rows = db.execute(
        select(PriceBar.symbol_id, PriceBar.timestamp, PriceBar.close).where(PriceBar.timeframe == timeframe)
    ).all()
    return pd.DataFrame(rows, columns=["symbol_id", "timestamp", "close"])


def _feature_frame(db: Session, timeframe: Timeframe, feature_set: str) -> pd.DataFrame:
    rows = db.execute(
        select(
            FeatureValue.symbol_id,
            FeatureValue.timestamp,
            *[getattr(FeatureValue, column) for column in FEATURE_COLUMNS],
        ).where(FeatureValue.timeframe == timeframe, FeatureValue.feature_set == feature_set)
    ).all()
    return pd.DataFrame(rows, columns=["symbol_id", "timestamp", *FEATURE_COLUMNS])


def build_training_frame(
    db: Session,
    *,
    timeframe: Timeframe,
    feature_set: str = "technical_v1",
    horizon_bars: int = 5,
    target_return: float = 0.03,
) -> pd.DataFrame:
    """Build a supervised frame from feature values and future returns."""

    prices = _price_frame(db, timeframe)
    features = _feature_frame(db, timeframe, feature_set)
    if prices.empty or features.empty:
        return pd.DataFrame()

    prices = prices.sort_values(["symbol_id", "timestamp"])
    prices["future_close"] = prices.groupby("symbol_id")["close"].shift(-horizon_bars)
    prices["future_return"] = (prices["future_close"] - prices["close"]) / prices["close"]
    labels = prices.dropna(subset=["future_return"])[["symbol_id", "timestamp", "future_return"]]
    labels["target"] = (labels["future_return"] >= target_return).astype(int)

    frame = features.merge(labels, on=["symbol_id", "timestamp"], how="inner")
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].fillna(frame[FEATURE_COLUMNS].median(numeric_only=True))
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["target", "future_return"])
    return frame


def train_baseline_models(
    db: Session,
    *,
    timeframe: Timeframe = Timeframe.DAILY,
    feature_set: str = "technical_v1",
    horizon_bars: int = 5,
    target_return: float = 0.03,
    artifact_dir: Path = Path("output/models"),
    min_rows: int = 80,
) -> TrainingResult:
    """Train cheap baseline classifiers and store the best run metadata."""

    frame = build_training_frame(
        db,
        timeframe=timeframe,
        feature_set=feature_set,
        horizon_bars=horizon_bars,
        target_return=target_return,
    )
    if len(frame) < min_rows or frame["target"].nunique() < 2:
        metrics = {
            "status": "skipped",
            "rows": int(len(frame)),
            "reason": "not enough labelled rows or only one target class",
        }
        model_run = ModelRun(
            name=f"baseline_{timeframe.value}_{horizon_bars}",
            model_type="skipped",
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            train_start=None,
            train_end=None,
            metrics_json=json.dumps(metrics, sort_keys=True),
            artifact_path=None,
        )
        db.add(model_run)
        db.flush()
        return TrainingResult(model_run=model_run, metrics=metrics)

    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=int)
    future_return = frame["future_return"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test, _, future_test = train_test_split(
        x,
        y,
        future_return,
        test_size=0.25,
        shuffle=False,
    )

    candidates = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
        ),
    }

    best_name = ""
    best_model = None
    best_metrics: dict[str, float | int | str] = {}
    best_precision = -1.0
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        rmse = float(np.sqrt(mean_squared_error(y_test, probability)))
        metrics = {
            "status": "trained",
            "model": name,
            "rows": int(len(frame)),
            "test_rows": int(len(y_test)),
            "positive_rate": float(y.mean()),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_test, probability)),
            "avg_return_top_decile": float(np.mean(future_test[probability >= np.quantile(probability, 0.9)]))
            if len(probability) >= 10
            else 0.0,
        }
        if metrics["precision"] > best_precision:
            best_precision = float(metrics["precision"])
            best_name = name
            best_model = model
            best_metrics = metrics

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{best_name}_{timeframe.value}_{horizon_bars}_{datetime.now(timezone.utc):%Y%m%d%H%M%S}.joblib"
    joblib.dump({"model": best_model, "feature_columns": FEATURE_COLUMNS, "metrics": best_metrics}, artifact_path)

    model_run = ModelRun(
        name=f"{best_name}_{timeframe.value}_{horizon_bars}",
        model_type=best_name,
        timeframe=timeframe,
        horizon_bars=horizon_bars,
        train_start=pd.to_datetime(frame["timestamp"].min()).to_pydatetime(),
        train_end=pd.to_datetime(frame["timestamp"].max()).to_pydatetime(),
        metrics_json=json.dumps(best_metrics, sort_keys=True),
        artifact_path=str(artifact_path),
    )
    db.add(model_run)
    db.flush()

    latest = frame.sort_values(["symbol_id", "timestamp"]).groupby("symbol_id").tail(1)
    probabilities = best_model.predict_proba(latest[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
    for row, probability in zip(latest.itertuples(index=False), probabilities):
        db.add(
            Prediction(
                model_run_id=model_run.id,
                symbol_id=int(row.symbol_id),
                prediction_time=pd.to_datetime(row.timestamp).to_pydatetime(),
                probability=float(probability),
                raw_score=float(probability),
            )
        )

    return TrainingResult(model_run=model_run, metrics=best_metrics)
