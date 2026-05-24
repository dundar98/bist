#!/usr/bin/env python3
"""Manual yfinance 1h data check.

This script is intentionally not a pytest test; run it directly when needed.
"""

from datetime import date, timedelta

import yfinance as yf


def main() -> int:
    symbol = "THYAO.IS"
    end = date.today()
    start = end - timedelta(days=30)

    print(f"Fetching 1h data for {symbol} from {start} to {end}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.isoformat(), end=end.isoformat(), interval="1h")

    if df.empty:
        print("No data found with start/end.")
        print("Trying without dates (last 30d period)...")
        df = ticker.history(period="1mo", interval="1h")

    print(f"Result: {len(df)} rows")
    if not df.empty:
        print(df.head())
        return 0

    print("Still no data.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
