"""Replays the synthetic transaction dataset onto a Kafka topic as a live
stream, simulating real-time transaction events arriving one at a time.

Only the *raw* transaction fields are sent (transaction_id, account_id,
timestamp, amount, merchant_category, location, device_id, is_fraud) — the
15 engineered behavioral features in the parquet file were computed offline
with full account history available, which a real streaming producer would
not have. Feature computation on the live stream is the feature store's job
(Session 3), not the producer's.

Schema and design decisions are documented in docs/kafka.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import KafkaError, Producer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [producer] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

RAW_FIELDS = [
    "transaction_id",
    "account_id",
    "timestamp",
    "amount",
    "merchant_category",
    "location",
    "device_id",
    "is_fraud",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/raw/transactions.parquet", help="Source parquet file")
    p.add_argument("--topic", default="transactions")
    p.add_argument("--bootstrap-servers", default="localhost:9092")
    p.add_argument("--rate", type=float, default=100.0, help="Transactions per second to emit")
    p.add_argument("--limit", type=int, default=5000, help="Max rows to replay (0 = all rows)")
    p.add_argument(
        "--inject-malformed-every",
        type=int,
        default=0,
        help="Every Nth message is deliberately corrupted, to exercise consumer error handling (0 = disabled)",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for malformed-message sampling")
    p.add_argument(
        "--amount-multiplier",
        type=float,
        default=1.0,
        help=(
            "Scales every replayed amount by this factor before sending -- "
            "a deliberate, documented synthetic distribution shift for "
            "exercising the Session 9 Evidently drift report (see "
            "docs/monitoring.md). 1.0 = no shift (default)."
        ),
    )
    return p.parse_args()


def delivery_report(err: KafkaError | None, msg) -> None:
    if err is not None:
        log.error("Delivery failed for key=%s: %s", msg.key(), err)


def make_malformed_payload(rng: random.Random) -> bytes:
    """Simulates the kinds of bad messages a real stream can produce:
    truncated JSON, wrong type for a field, or a missing required field."""
    choice = rng.randint(0, 2)
    if choice == 0:
        return b'{"transaction_id": "txn_broken", "account_id": "acct_0000'  # truncated
    if choice == 1:
        return json.dumps({"transaction_id": "txn_broken", "amount": "not-a-number"}).encode()
    return json.dumps({"transaction_id": "txn_broken"}).encode()  # missing required fields


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(
            "Input dataset not found at %s — generate it first with "
            "data_generation/generate_transactions.py",
            input_path,
        )
        sys.exit(1)

    log.info("Loading %s ...", input_path)
    df = pd.read_parquet(input_path, columns=RAW_FIELDS)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    if args.amount_multiplier != 1.0:
        df["amount"] = (df["amount"] * args.amount_multiplier).round(2)
        log.warning(
            "Injecting synthetic drift: scaling amount by %.2fx for this replay",
            args.amount_multiplier,
        )
    log.info("Replaying %d rows at %.1f tx/sec (topic=%s)", len(df), args.rate, args.topic)

    producer = Producer({"bootstrap.servers": args.bootstrap_servers})

    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    sent, malformed = 0, 0
    start = time.monotonic()

    for i, row in enumerate(df.itertuples(index=False), start=1):
        record = row._asdict()
        record["timestamp"] = record["timestamp"].isoformat()
        record["is_fraud"] = bool(record["is_fraud"])

        if args.inject_malformed_every and i % args.inject_malformed_every == 0:
            payload = make_malformed_payload(rng)
            malformed += 1
        else:
            payload = json.dumps(record).encode("utf-8")

        producer.produce(
            args.topic,
            key=record["account_id"].encode("utf-8"),
            value=payload,
            callback=delivery_report,
        )
        producer.poll(0)
        sent += 1

        if sent % 500 == 0:
            log.info("Sent %d/%d messages (%d malformed injected)", sent, len(df), malformed)

        if interval:
            time.sleep(interval)

    producer.flush(30)
    elapsed = time.monotonic() - start
    log.info(
        "Done: %d messages sent (%d deliberately malformed) in %.1fs (%.1f tx/sec actual)",
        sent,
        malformed,
        elapsed,
        sent / elapsed if elapsed else 0.0,
    )


if __name__ == "__main__":
    main()
