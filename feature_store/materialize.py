"""One-time (or periodic) backfill: materializes the offline feature values
in data/raw/transactions.parquet into the Redis online store.

For each account_id, Feast writes the feature row with the latest
event_timestamp (i.e. the account's most recent transaction as of the end
of the given date range) into Redis. This is the "training-time" snapshot
that the online store starts from; from that point on, the Kafka consumer
(Session 3 wiring in consumer/consume.py) keeps it fresh via
`store.push()` as new transactions arrive.

Usage:
    python feature_store/materialize.py
    python feature_store/materialize.py --start 2025-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from feast import FeatureStore

REPO_PATH = Path(__file__).resolve().parent / "feature_repo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--start",
        default="2025-01-01",
        help="Start of the materialization window (inclusive), YYYY-MM-DD",
    )
    p.add_argument(
        "--end",
        default="2026-01-01",
        help="End of the materialization window (exclusive), YYYY-MM-DD. "
        "Defaults to just past the dataset's max timestamp (2025-12-31) so "
        "every transaction is covered.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    store = FeatureStore(repo_path=str(REPO_PATH))
    print(f"Materializing account_transaction_features from {start} to {end}...")
    store.materialize(start_date=start, end_date=end)
    print("Done.")


if __name__ == "__main__":
    main()
