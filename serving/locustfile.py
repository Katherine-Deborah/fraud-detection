"""Session 8 load test: hits /predict at a sustained rate and lets Locust
report real p50/p95/p99 latency (PROJECT.md target: p99 < 50ms @ 500 TPS
-- report the actual measured number even if it misses the target).

Account IDs are sampled from the real 50,000-account population
(data/raw/accounts.parquet) rather than random strings, so most requests
hit Feast's materialized online store (the realistic path) instead of
the cold-start fallback in serving/app.py.

Run headless, e.g.:
    .venv-serving/Scripts/locust -f serving/locustfile.py --headless \
        -u 200 -r 20 -t 60s --host http://localhost:8090 \
        --csv=docs/eda/load_test
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
from locust import HttpUser, between, task

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.generate_transactions import MERCHANT_CATEGORIES  # noqa: E402

ACCOUNTS_PATH = REPO_ROOT / "data" / "raw" / "accounts.parquet"
_accounts_df = pd.read_parquet(ACCOUNTS_PATH, columns=["account_id"])
ACCOUNT_IDS = _accounts_df["account_id"].tolist()


class FraudScoringUser(HttpUser):
    # No think time: this is a throughput/latency load test, not a
    # user-behavior simulation -- each simulated user fires requests back
    # to back, and sustained rate is controlled via -u/-r (user count /
    # spawn rate) at invocation time.
    wait_time = between(0, 0)

    @task
    def predict(self) -> None:
        account_id = random.choice(ACCOUNT_IDS)
        category = random.choice(MERCHANT_CATEGORIES)
        amount = round(random.uniform(5, 800), 2)
        self.client.post(
            "/predict",
            json={
                "account_id": account_id,
                "amount": amount,
                "merchant_category": category,
            },
        )
