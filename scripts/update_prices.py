#!/usr/bin/env python3
"""Incrementally update OHLCV price bars in the database."""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import get_data_loader
from database.repositories.prices import (
    PriceDataError,
    get_last_price_timestamp,
    list_active_symbols,
    timeframe_from_string,
    upsert_price_bars,
)
from database.session import SessionLocal
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update BIST price bars in the database")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "1h", "15m"])
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic"])
    parser.add_argument("--symbols", help="Comma-separated ticker list. Defaults to active BIST100 symbols.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols processed.")
    parser.add_argument("--start", default=None, help="Initial backfill start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--lookback-days", type=int, default=365 * 3, help="Backfill window when no prior bars exist.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _intraday_cap_days(timeframe: str, lookback_days: int) -> int:
    if timeframe == "15m":
        return min(lookback_days, 59)
    if timeframe == "1h":
        return min(lookback_days, 729)
    return lookback_days


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    timeframe = timeframe_from_string(args.timeframe)
    end_date = _parse_date(args.end) or date.today()
    lookback_days = _intraday_cap_days(args.timeframe, args.lookback_days)
    loader = get_data_loader(source=args.source)

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

        logger.info("Updating %s symbols for timeframe=%s source=%s", len(symbols), timeframe.value, args.source)

        for symbol in symbols:
            try:
                last_ts = get_last_price_timestamp(db, symbol.id, timeframe)
                if last_ts is not None:
                    start_date = last_ts.date() + timedelta(days=1)
                else:
                    start_date = _parse_date(args.start) or (end_date - timedelta(days=lookback_days))

                if start_date > end_date:
                    logger.info("%s already up to date", symbol.ticker)
                    continue

                df = loader.load(symbol.ticker, start_date=start_date, end_date=end_date, interval=timeframe.value)
                changed = upsert_price_bars(db, symbol, timeframe, df, source=args.source)
                db.commit()
                total_changed += changed
                logger.info("%s: persisted %s bars", symbol.ticker, changed)
            except Exception as exc:
                db.rollback()
                if last_ts is not None and "No data returned" in str(exc):
                    logger.info("%s: no new bars returned; keeping existing latest bar %s", symbol.ticker, last_ts.date())
                    continue
                message = f"{symbol.ticker}: {exc}"
                errors.append(message)
                logger.error(message)

    print(f"Price update complete. symbols={len(symbols)} changed_bars={total_changed} errors={len(errors)}")
    if errors:
        for error in errors[:20]:
            print(f"ERROR {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
