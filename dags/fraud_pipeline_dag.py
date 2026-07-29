"""Session 4: batch orchestration DAG.

ingest_batch -> validate_batch -> feature_engineering -> materialize_to_feast

This is a periodic *reconciliation* job, distinct from the Session 2/3
real-time path (Kafka consumer -> OnlineFeatureEngine -> store.push per
transaction). That streaming path is documented (docs/feature_store.md) as
not idempotent: replaying already-processed messages double-counts into the
rolling features. This DAG instead re-reads everything landed in the data
lake so far, deduplicates by transaction_id, recomputes all 15 engineered
features from scratch with the same vectorized function the offline dataset
was built with (data_generation.generate_transactions.
compute_engineered_features), and pushes the result into Feast -- a
correctness backstop that doesn't depend on the streaming path's state.

Great Expectations runs on the *raw* deduplicated batch, before feature
engineering, so a bad batch never reaches feature computation or Feast.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LAKE_DIR = PROJECT_ROOT / "data" / "lake"
ACCOUNTS_PATH = PROJECT_ROOT / "data" / "raw" / "accounts.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_REPO_PATH = PROJECT_ROOT / "feature_store" / "feature_repo"

REQUIRED_FIELDS = [
    "transaction_id", "account_id", "timestamp", "amount",
    "merchant_category", "location", "device_id", "is_fraud",
]


@dag(
    dag_id="fraud_pipeline",
    description="Ingest -> validate (Great Expectations) -> feature engineering -> materialize to Feast",
    schedule="@daily",
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=["fraud-detection", "session-4"],
)
def fraud_pipeline():

    @task()
    def ingest_batch() -> str:
        """Reads every JSONL record landed by the Kafka consumer
        (consumer/consume.py) under data/lake/dt=*/transactions.jsonl,
        excluding the dead-letter directory, dedupes by transaction_id
        (closing the known at-least-once double-landing gap -- see
        docs/kafka.md), and writes one batch parquet for this run."""
        files = sorted(LAKE_DIR.glob("dt=*/transactions.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"No lake files found under {LAKE_DIR} -- run the producer/"
                "consumer (Session 2) before triggering this DAG."
            )

        frames = [pd.read_json(f, lines=True) for f in files]
        df = pd.concat(frames, ignore_index=True)
        before = len(df)
        df = df.drop_duplicates(subset="transaction_id", keep="last")
        deduped = before - len(df)

        missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
        if missing:
            raise ValueError(f"Lake data is missing required columns: {missing}")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "batch_latest.parquet"
        df[REQUIRED_FIELDS].to_parquet(out_path, index=False)
        print(f"Ingested {before} records from {len(files)} file(s), "
              f"deduplicated {deduped}, wrote {len(df)} rows to {out_path}")
        return str(out_path)

    @task()
    def validate_batch(batch_path: str) -> str:
        """Runs the Great Expectations suite (validation/
        great_expectations_checks.py) against the raw deduplicated batch.
        Raises on failure, which fails this Airflow task -- a bad batch
        never reaches feature engineering or Feast."""
        from validation.great_expectations_checks import validate_batch as run_checks

        df = pd.read_parquet(batch_path)
        result = run_checks(df)  # raises BatchValidationError on failure
        print(f"Validation passed: {result['statistics']['successful_expectations']}"
              f"/{result['statistics']['evaluated_expectations']} expectations met.")
        return batch_path

    @task()
    def feature_engineering(batch_path: str) -> str:
        """Recomputes the same 15 engineered features as the offline
        dataset generator, vectorized over the whole validated batch.

        The raw lake records only carry a city name (`location`), not the
        offline dataset's per-transaction jittered lat/lon, so geo features
        computed here use city-centroid coordinates -- the same documented
        approximation OnlineFeatureEngine uses for the streaming path (see
        docs/feature_store.md's "Geo-feature tolerance" section)."""
        from data_generation.generate_transactions import compute_engineered_features
        from feature_store.online_features import CITY_COORDS

        df = pd.read_parquet(batch_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        coords = df["location"].map(lambda c: CITY_COORDS.get(c, (None, None)))
        df["lat"] = coords.map(lambda t: t[0])
        df["lon"] = coords.map(lambda t: t[1])

        accounts = pd.read_parquet(ACCOUNTS_PATH)
        features = compute_engineered_features(df, accounts)
        features = features.drop(columns=["lat", "lon"])

        out_path = PROCESSED_DIR / "features_latest.parquet"
        features.to_parquet(out_path, index=False)
        print(f"Computed features for {len(features)} rows, wrote {out_path}")
        return str(out_path)

    @task()
    def materialize_to_feast(features_path: str) -> None:
        """Pushes the recomputed features into the Feast online store
        (Redis), reachable here via the compose network's `redis` service
        name -- see feature_store/store_utils.py for why this differs from
        the host-venv scripts' localhost:6380."""
        from feast.data_source import PushMode

        from feature_store.store_utils import get_feature_store

        df = pd.read_parquet(features_path)
        store = get_feature_store(FEATURE_REPO_PATH)
        store.push("transactions_push_source", df, to=PushMode.ONLINE)
        print(f"Pushed {len(df)} feature rows to the Feast online store.")

    batch_path = ingest_batch()
    validated_path = validate_batch(batch_path)
    features_path = feature_engineering(validated_path)
    materialize_to_feast(features_path)


fraud_pipeline()
