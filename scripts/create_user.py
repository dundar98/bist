#!/usr/bin/env python3
"""Create or update a manually managed platform user."""

import argparse
import getpass
import sys
from pathlib import Path

from sqlalchemy import or_, select

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.security import hash_password
from database.models import User, UserRole
from database.session import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a BIST platform user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", help="Plain password. If omitted, prompt securely.")
    parser.add_argument("--full-name", default=None)
    parser.add_argument("--role", default=UserRole.VIEWER.value, choices=[role.value for role in UserRole])
    parser.add_argument("--inactive", action="store_true")
    parser.add_argument("--update", action="store_true", help="Update existing user if username/email already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    username = args.username.strip().lower()
    email = args.email.strip().lower()
    password = args.password or getpass.getpass("Password: ")

    if not password:
        print("Password cannot be empty", file=sys.stderr)
        return 1

    with SessionLocal() as db:
        existing = db.scalar(
            select(User).where(
                or_(
                    User.username == username,
                    User.email == email,
                )
            )
        )

        if existing is not None and not args.update:
            print("User already exists. Re-run with --update to modify.", file=sys.stderr)
            return 1

        if existing is None:
            user = User(
                username=username,
                email=email,
                password_hash=hash_password(password),
                full_name=args.full_name,
                role=UserRole(args.role),
                is_active=not args.inactive,
            )
            db.add(user)
            action = "created"
        else:
            user = existing
            user.username = username
            user.email = email
            user.password_hash = hash_password(password)
            user.full_name = args.full_name
            user.role = UserRole(args.role)
            user.is_active = not args.inactive
            action = "updated"

        db.commit()
        db.refresh(user)

    print(f"User {action}. id={user.id} username={user.username} role={user.role.value} active={user.is_active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
