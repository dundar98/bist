#!/usr/bin/env python3
"""Update signal outcome metrics from stored future price bars."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.repositories.outcomes import update_signal_outcomes
from database.session import SessionLocal
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update signal outcome metrics")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--include-non-buy", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    with SessionLocal() as db:
        changed = update_signal_outcomes(
            db,
            limit=args.limit,
            only_buy=not args.include_non_buy,
        )
        db.commit()

    print(f"Outcome update complete. changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
