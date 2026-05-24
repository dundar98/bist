#!/usr/bin/env python3
"""Send a small JSON notification to a webhook URL."""

import argparse
import json
import os
import sys
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send webhook notification")
    parser.add_argument("--title", required=True)
    parser.add_argument("--message", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    webhook_url = os.getenv("BIST_ALERT_WEBHOOK_URL")
    if not webhook_url:
        print("BIST_ALERT_WEBHOOK_URL not configured; skipping notification")
        return 0

    payload = json.dumps({"title": args.title, "message": args.message}).encode("utf-8")
    request = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        print(f"Notification sent. status={response.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

