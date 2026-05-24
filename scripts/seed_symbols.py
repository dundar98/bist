#!/usr/bin/env python3
"""Seed the database with the current in-repo BIST universe."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.bist100_validator import BIST100_SYMBOLS
from database.models import Symbol
from database.session import SessionLocal


def main() -> None:
    created = 0
    updated = 0

    with SessionLocal() as db:
        existing = {
            symbol.ticker: symbol
            for symbol in db.query(Symbol).filter(Symbol.ticker.in_(BIST100_SYMBOLS)).all()
        }

        for ticker in sorted(BIST100_SYMBOLS):
            symbol = existing.get(ticker)
            if symbol is None:
                db.add(Symbol(ticker=ticker, market="BIST", is_active=True, is_bist100=True))
                created += 1
            else:
                symbol.is_active = True
                symbol.is_bist100 = True
                updated += 1

        db.commit()

    print(f"Seeded symbols. created={created} updated={updated}")


if __name__ == "__main__":
    main()
