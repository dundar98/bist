#!/usr/bin/env python3
"""Create a timestamped PostgreSQL backup using pg_dump."""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    backup_dir = Path(os.getenv("BIST_BACKUP_DIR", "backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    output = backup_dir / f"bist_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.dump"

    database_url = os.getenv("BIST_DATABASE_URL")
    if not database_url:
        print("BIST_DATABASE_URL is required", file=sys.stderr)
        return 1

    command = ["pg_dump", "--format=custom", "--file", str(output), database_url]
    subprocess.run(command, check=True)
    print(f"Backup created: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

