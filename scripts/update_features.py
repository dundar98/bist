#!/usr/bin/env python3
"""Compute and persist feature values from stored price bars."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.repositories.features import FeatureStoreError, compute_and_store_features
from database.repositories.prices import list_active_symbols, timeframe_from_string
from database.session import SessionLocal
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute technical features from stored price bars")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "1h", "15m"])
    parser.add_argument("--symbols", help="Comma-separated ticker list. Defaults to active BIST100 symbols.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lookback-bars", type=int, default=260)
    parser.add_argument("--feature-set", default="technical_v1")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    timeframe = timeframe_from_string(args.timeframe)
    total_changed = 0
    errors: list[str] = []

    with SessionLocal() as db:
        if args.symbols:
            requested = {ticker.strip().upper() for ticker in args.symbols.split(",") if ticker.strip()}
            symbols = [symbol for symbol in list_active_symbols(db, limit=None) if symbol.ticker in requested]
            missing = sorted(requested - {symbol.ticker for symbol in symbols})
            for ticker in missing:
                errors.append(f"{ticker}: symbol not found or inactive")
        else:
            symbols = list_active_symbols(db, limit=args.limit)

        if args.limit is not None and args.symbols:
            symbols = symbols[: args.limit]

        logger.info("Computing features for %s symbols timeframe=%s", len(symbols), timeframe.value)

        for symbol in symbols:
            try:
                changed = compute_and_store_features(
                    db,
                    symbol,
                    timeframe,
                    lookback_bars=args.lookback_bars,
                    feature_set=args.feature_set,
                )
                db.commit()
                total_changed += changed
                logger.info("%s: persisted %s feature rows", symbol.ticker, changed)
            except Exception as exc:
                db.rollback()
                message = f"{symbol.ticker}: {exc}"
                errors.append(message)
                logger.error(message)

    print(f"Feature update complete. symbols={len(symbols)} changed_rows={total_changed} errors={len(errors)}")
    if errors:
        for error in errors[:20]:
            print(f"ERROR {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
