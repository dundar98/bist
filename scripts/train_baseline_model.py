#!/usr/bin/env python3
"""Train cheap baseline ML models and store metrics in model_runs."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.repositories.prices import timeframe_from_string
from database.session import SessionLocal
from ml.training import train_baseline_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline ML model")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "1h", "15m"])
    parser.add_argument("--feature-set", default="technical_v1")
    parser.add_argument("--horizon-bars", type=int, default=5)
    parser.add_argument("--target-return", type=float, default=0.03)
    parser.add_argument("--min-rows", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeframe = timeframe_from_string(args.timeframe)

    with SessionLocal() as db:
        result = train_baseline_models(
            db,
            timeframe=timeframe,
            feature_set=args.feature_set,
            horizon_bars=args.horizon_bars,
            target_return=args.target_return,
            min_rows=args.min_rows,
        )
        model_run_id = result.model_run.id
        model_type = result.model_run.model_type
        metrics = result.metrics
        db.commit()

    print(f"Model run stored. id={model_run_id} type={model_type}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
