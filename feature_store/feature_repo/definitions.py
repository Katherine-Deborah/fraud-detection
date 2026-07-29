"""Feast feature repo definitions for the fraud detection pipeline.

Entity, sources, and feature view live in one file per Feast convention
(`feast apply` scans every .py file in this directory). See
docs/feature_store.md for the full write-up of why this layout was chosen.

Dtypes below match the actual on-disk dtypes of
data/raw/transactions.parquet exactly (verified via `df.dtypes`), notably
`txn_count_last_1h`/`txn_count_last_24h`, which are float64 on disk because
they come from a time-based `.rolling().count()` in the generator, not a
plain integer count.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float64, Int32, Int64
from feast.value_type import ValueType

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSACTIONS_PATH = REPO_ROOT / "data" / "raw" / "transactions.parquet"

account = Entity(
    name="account",
    join_keys=["account_id"],
    value_type=ValueType.STRING,
    description="A synthetic account identifier (see docs/dataset.md).",
)

# Batch/offline source: the Session 1 dataset. Every row is one transaction
# event; feature values are per-transaction, so "the features for account X"
# means "the features as of X's most recent transaction."
transactions_source = FileSource(
    name="transactions_source",
    path=str(TRANSACTIONS_PATH),
    timestamp_field="timestamp",
)

# Streaming path: the Session 2 Kafka consumer computes the same 15 features
# per arriving transaction (feature_store/online_features.py) and pushes
# them here via `store.push()`. A PushSource must wrap a batch source so
# Feast can reconcile the schema between the offline and online paths — the
# batch_source is only used for schema/offline reads, not re-read on push.
transactions_push_source = PushSource(
    name="transactions_push_source",
    batch_source=transactions_source,
)

account_transaction_features = FeatureView(
    name="account_transaction_features",
    entities=[account],
    # Long TTL: the synthetic dataset spans all of 2025 and materialization
    # pulls the latest row per account as of "now" (2026), so a short TTL
    # relative to real transaction recency would make every feature look
    # stale. In production this would track real transaction cadence instead.
    ttl=timedelta(days=3650),
    schema=[
        Field(name="hour_of_day", dtype=Int32),
        Field(name="day_of_week", dtype=Int32),
        Field(name="is_weekend", dtype=Int64),
        Field(name="time_since_last_txn_sec", dtype=Float64),
        Field(name="txn_count_last_1h", dtype=Float64),
        Field(name="txn_count_last_24h", dtype=Float64),
        Field(name="avg_amount_last_5_txns", dtype=Float64),
        Field(name="amount_to_avg_ratio", dtype=Float64),
        Field(name="geo_distance_from_last_txn_km", dtype=Float64),
        Field(name="home_distance_km", dtype=Float64),
        Field(name="is_new_device", dtype=Int64),
        Field(name="is_new_merchant_category", dtype=Int64),
        Field(name="cumulative_distinct_devices", dtype=Int64),
        Field(name="cumulative_distinct_merchant_categories", dtype=Int64),
        Field(name="account_txn_seq_num", dtype=Int64),
    ],
    online=True,
    source=transactions_push_source,
)
