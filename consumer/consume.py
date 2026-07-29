"""Consumes transaction events from Kafka, validates them, and writes valid
raw records into a local "data lake" (JSONL files partitioned by date).

Error handling (documented in detail in docs/kafka.md):
  - Malformed messages (invalid JSON, missing required fields, wrong types)
    are written to a dead-letter file instead of crashing the consumer.
  - Out-of-order timestamps (a transaction arriving with an earlier
    timestamp than one already seen for the same account) are logged and
    counted, not dropped — the raw record is still landed in the lake;
    ordering is a concern for the feature-computation layer, not for raw
    ingestion.
  - Consumer restarts resume from the last *committed* offset (manual
    commit, only after a record is durably written), giving at-least-once
    delivery. A crash between write and commit can reprocess a record on
    restart; downstream consumers should dedupe by transaction_id.

Session 3 adds real-time feature computation: each valid record is run
through feature_store/online_features.py (the same 15 engineered features
as the offline generator, computed incrementally from Redis-backed
per-account state) and pushed into the Feast online store. A failure to
push a feature update is logged and counted but does not stop ingestion —
the data lake write is the durability guarantee; a missed feature push is
recoverable by re-running feature_store/materialize.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from confluent_kafka import Consumer, KafkaException
from feast import FeatureStore
from feast.data_source import PushMode

from feature_store.online_features import OnlineFeatureEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [consumer] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "transaction_id": str,
    "account_id": str,
    "timestamp": str,
    "amount": (int, float),
    "merchant_category": str,
    "location": str,
    "device_id": str,
    "is_fraud": bool,
}


class DeadLetterWriter:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "_dead_letter" / "dead_letter.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, raw_value: bytes, error: str, topic: str, partition: int, offset: int) -> None:
        entry = {
            "error": error,
            "topic": topic,
            "partition": partition,
            "offset": offset,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "raw_value": raw_value.decode("utf-8", errors="replace"),
        }
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class DataLakeWriter:
    """Appends valid records to date-partitioned JSONL files."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._handles: dict[str, "TextIOWrapper"] = {}

    def write(self, record: dict) -> None:
        date_str = record["timestamp"][:10]  # YYYY-MM-DD
        fh = self._handles.get(date_str)
        if fh is None:
            partition_dir = self.output_dir / f"dt={date_str}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            fh = open(partition_dir / "transactions.jsonl", "a", encoding="utf-8")
            self._handles[date_str] = fh
        fh.write(json.dumps(record) + "\n")
        fh.flush()

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()


def validate(payload: bytes) -> tuple[dict | None, str | None]:
    """Returns (record, None) if valid, or (None, error_message) if not."""
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"

    if not isinstance(record, dict):
        return None, "payload is not a JSON object"

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            return None, f"missing required field: {field}"
        if not isinstance(record[field], expected_type):
            return None, f"field {field!r} has wrong type: {type(record[field]).__name__}"

    try:
        datetime.fromisoformat(record["timestamp"])
    except ValueError:
        return None, f"unparseable timestamp: {record['timestamp']!r}"

    return record, None


def forward_to_feature_engineering(
    record: dict, engine: OnlineFeatureEngine, store: FeatureStore
) -> None:
    """Computes the 15 engineered features for this transaction from
    Redis-backed per-account state (see feature_store/online_features.py)
    and pushes them into the Feast online store, keeping it consistent
    with what training sees from the offline parquet (modulo the
    documented geo-jitter tolerance)."""
    features = engine.compute(record)
    row = pd.DataFrame([features])
    store.push("transactions_push_source", row, to=PushMode.ONLINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topic", default="transactions")
    p.add_argument("--bootstrap-servers", default="localhost:9092")
    p.add_argument("--group-id", default="fraud-lake-writer")
    p.add_argument("--output-dir", default="data/lake")
    p.add_argument(
        "--offset-reset",
        default="earliest",
        choices=["earliest", "latest"],
        help="Where to start if this group has no committed offset yet",
    )
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=15.0,
        help="Exit after this many seconds with no new messages (0 = run forever)",
    )
    p.add_argument(
        "--feature-repo",
        default="feature_store/feature_repo",
        help="Path to the Feast feature repo (see feature_store/feature_repo/feature_store.yaml)",
    )
    p.add_argument(
        "--redis-host",
        default="localhost",
        help="Host for the online-feature-engine's Redis state (db=1) -- see feature_store/online_features.py",
    )
    p.add_argument("--redis-port", type=int, default=6380)
    p.add_argument(
        "--accounts-path",
        default="data/raw/accounts.parquet",
        help="Account home-location reference table (see data_generation/generate_transactions.py)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    lake = DataLakeWriter(output_dir)
    dead_letter = DeadLetterWriter(output_dir)

    feature_store = FeatureStore(repo_path=str(REPO_ROOT / args.feature_repo))
    feature_engine = OnlineFeatureEngine(args.redis_host, args.redis_port, REPO_ROOT / args.accounts_path)

    consumer = Consumer(
        {
            "bootstrap.servers": args.bootstrap_servers,
            "group.id": args.group_id,
            "auto.offset.reset": args.offset_reset,
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([args.topic])

    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        log.info("Shutdown signal received, finishing current message and exiting...")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    last_seen_ts: dict[str, str] = {}
    valid_count = 0
    malformed_count = 0
    out_of_order_count = 0
    feature_push_failures = 0
    last_message_time = time.monotonic()

    log.info(
        "Consuming topic=%s group=%s offset-reset=%s -> %s",
        args.topic,
        args.group_id,
        args.offset_reset,
        output_dir,
    )

    try:
        while running:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                if args.idle_timeout and (time.monotonic() - last_message_time) > args.idle_timeout:
                    log.info("No messages for %.0fs, exiting.", args.idle_timeout)
                    break
                continue

            if msg.error():
                raise KafkaException(msg.error())

            last_message_time = time.monotonic()
            record, error = validate(msg.value())

            if error is not None:
                malformed_count += 1
                log.warning("Dead-lettering message at offset %d: %s", msg.offset(), error)
                dead_letter.write(msg.value(), error, msg.topic(), msg.partition(), msg.offset())
                consumer.commit(msg, asynchronous=False)
                continue

            account_id = record["account_id"]
            prev_ts = last_seen_ts.get(account_id)
            if prev_ts is not None and record["timestamp"] < prev_ts:
                out_of_order_count += 1
                log.warning(
                    "Out-of-order timestamp for %s: %s arrived after %s (landing anyway)",
                    account_id,
                    record["timestamp"],
                    prev_ts,
                )
            last_seen_ts[account_id] = max(prev_ts or record["timestamp"], record["timestamp"])

            lake.write(record)
            try:
                forward_to_feature_engineering(record, feature_engine, feature_store)
            except Exception:
                # A feature-store hiccup shouldn't block raw ingestion --
                # the lake write above already durably captured this
                # record, and a missed push is recoverable by re-running
                # feature_store/materialize.py. See module docstring.
                feature_push_failures += 1
                log.exception("Feature push failed for %s, continuing", record["transaction_id"])
            valid_count += 1
            consumer.commit(msg, asynchronous=False)

            if valid_count % 500 == 0:
                log.info(
                    "Landed %d valid records (%d malformed, %d out-of-order, %d feature-push failures)",
                    valid_count,
                    malformed_count,
                    out_of_order_count,
                    feature_push_failures,
                )
    finally:
        consumer.close()
        lake.close()
        dead_letter.close()
        log.info(
            "Final: %d valid records landed, %d dead-lettered, %d out-of-order timestamps, %d feature-push failures",
            valid_count,
            malformed_count,
            out_of_order_count,
            feature_push_failures,
        )


if __name__ == "__main__":
    main()
