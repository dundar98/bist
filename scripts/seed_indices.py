#!/usr/bin/env python3
"""Seed Turkish market index symbols used by the market radar."""

import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.models import Symbol
from database.session import SessionLocal

INDICES = [
    ("XU100", "BIST 100", "Endeks"),
    ("XU030", "BIST 30", "Endeks"),
    ("XBANK", "BIST Banka", "Endeks"),
    ("XUSIN", "BIST Sanayi", "Endeks"),
]


def main() -> int:
    created = 0
    updated = 0
    with SessionLocal() as db:
        for ticker, name, sector in INDICES:
            symbol = db.scalar(select(Symbol).where(Symbol.ticker == ticker))
            if symbol is None:
                db.add(
                    Symbol(
                        ticker=ticker,
                        name=name,
                        sector=sector,
                        market="BIST_INDEX",
                        is_active=True,
                        is_bist100=False,
                    )
                )
                created += 1
            else:
                symbol.name = name
                symbol.sector = sector
                symbol.market = "BIST_INDEX"
                symbol.is_active = True
                symbol.is_bist100 = False
                updated += 1
        db.commit()

    print(f"Seeded indices. created={created} updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
