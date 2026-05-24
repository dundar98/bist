#!/usr/bin/env python3
"""Build a selective portfolio snapshot from stored feature values."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.repositories.portfolio import create_portfolio_snapshot
from database.repositories.prices import timeframe_from_string
from database.session import SessionLocal
from signals.scoring import horizon_from_string
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a selective BIST portfolio from feature scores")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "1h", "15m"])
    parser.add_argument("--horizon", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--strategy", default="technical_selective_v1")
    parser.add_argument("--feature-set", default="technical_v1")
    parser.add_argument("--symbol-limit", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=55.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    timeframe = timeframe_from_string(args.timeframe)
    horizon = horizon_from_string(args.horizon)

    with SessionLocal() as db:
        snapshot = create_portfolio_snapshot(
            db,
            timeframe=timeframe,
            horizon=horizon,
            strategy=args.strategy,
            feature_set=args.feature_set,
            symbol_limit=args.symbol_limit,
            max_positions=args.max_positions,
            min_score=args.min_score,
        )
        db.commit()
        db.refresh(snapshot)

        print(
            f"Portfolio built. id={snapshot.id} timeframe={timeframe.value} "
            f"horizon={horizon.value} items={len(snapshot.items)} strategy={snapshot.strategy}"
        )
        for item in sorted(snapshot.items, key=lambda row: row.rank)[:20]:
            ticker = item.symbol.ticker if item.symbol else str(item.symbol_id)
            print(f"{item.rank:02d}. {ticker} score={item.score:.2f} weight={item.suggested_weight:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
