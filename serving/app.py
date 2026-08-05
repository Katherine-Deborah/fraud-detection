"""Session 8: FastAPI gateway for real-time fraud scoring.

    POST /predict {account_id, amount, merchant_category, [timestamp]}
      -> fraud_score, is_fraud, threshold, ...

Design decision (confirmed with the user before building this): `/predict`
is **read-only** against the Feast/Redis online store. It calls
`store.get_online_features()` to fetch the account's already-materialized
behavioral features (the 12 of the 15 engineered features that genuinely
depend on transaction history -- see FEAST_FETCH_FEATURES below), and does
**not** call `feature_store.online_features.OnlineFeatureEngine.compute()`,
which would recompute features against the *current* transaction and
mutate the account's rolling Redis state as a side effect.

Two reasons this matters:
  1. `docs/feature_store.md` already documents that the online engine's
     Redis updates are not idempotent -- a load test (this session's other
     deliverable) hammering a mutating endpoint would inflate rolling
     counters and corrupt the same account history future requests read
     from. A read-only endpoint can be load-tested repeatedly with no
     side effects.
  2. A scoring request is not necessarily a confirmed real transaction
     (e.g. a pre-auth check) -- state advancement stays exclusively owned
     by the Kafka consumer -> OnlineFeatureEngine -> Feast push pipeline
     (Sessions 2/3), so there's a single source of truth for "what actually
     happened."

The real tradeoff this creates, documented rather than hidden: the fetched
features reflect the account's state as of its *last materialized/pushed*
transaction, not freshly recomputed against the transaction being scored
right now. Concretely, `txn_count_last_1h/24h` won't include the current
request, and `amount_to_avg_ratio` reflects the previous transaction's
ratio, not this one's. `hour_of_day`/`day_of_week`/`is_weekend` are the one
exception -- those three are pure functions of the transaction's own
timestamp (not history), so this gateway computes them fresh from the
request instead of pulling stale values out of Feast. See docs/serving.md
for the full writeup and what a mutating/hybrid design would look like.

Request schema is intentionally narrower than the raw Kafka message shape
(consumer/consume.py's REQUIRED_FIELDS): `device_id` and `location` are
dropped rather than accepted-and-silently-ignored, since neither currently
feeds into the read-only feature vector (is_new_device / geo features come
from Feast's stored state, not the request) -- an honest API surface only
accepts fields that actually affect the score.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_client import REGISTRY as DEFAULT_REGISTRY
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.generate_transactions import MERCHANT_CATEGORIES  # noqa: E402
from feature_store.store_utils import get_feature_store  # noqa: E402

FEATURE_REPO = REPO_ROOT / "feature_store" / "feature_repo"
METADATA_PATH = REPO_ROOT / "serving" / "model_metadata.json"
TRITON_HTTP_URL = "http://localhost:8000"

# --- Prometheus metrics (Session 9) -----------------------------------
#
# docs/serving.md documents running this gateway as `uvicorn --workers 4`
# (needed to hit any real throughput at all on this Windows/ProactorEventLoop
# setup). That means 4 separate OS processes, each with its own in-memory
# Counter/Histogram state -- a naive /metrics scrape would only ever see
# whichever one worker happened to handle that particular request, not the
# aggregate across all 4. prometheus_client's documented fix is multiprocess
# mode: set PROMETHEUS_MULTIPROC_DIR to an empty directory *before* any
# worker imports this module, and each worker then writes its counters to
# per-PID mmap files there instead of pure memory; /metrics aggregates them
# across files at scrape time via MultiProcessCollector. The directory must
# be created/emptied exactly once by the parent process before workers fork
# -- see serving/run_server.py, the launcher that does that. Running this
# module directly (e.g. `uvicorn serving.app:app` with no --workers, for
# quick local testing) works unmodified: MULTIPROC_DIR is simply unset and
# everything falls back to the default single-process registry.
MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

if MULTIPROC_DIR:
    # The default registry's process/platform/GC collectors report on only
    # the current process and don't aggregate meaningfully across workers in
    # multiprocess mode -- prometheus_client's own docs say to drop them.
    from prometheus_client import GC_COLLECTOR, PLATFORM_COLLECTOR, PROCESS_COLLECTOR

    for collector in (GC_COLLECTOR, PLATFORM_COLLECTOR, PROCESS_COLLECTOR):
        try:
            DEFAULT_REGISTRY.unregister(collector)
        except KeyError:
            pass

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the gateway",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10),
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Predictions made, by fraud/not-fraud outcome",
    ["is_fraud"],
)

# The 12 of the 15 engineered features that are genuinely history-dependent
# and therefore worth fetching from the online store -- hour_of_day/
# day_of_week/is_weekend are computed fresh from the request instead (see
# module docstring).
FEAST_FETCH_FEATURES = [
    "time_since_last_txn_sec",
    "txn_count_last_1h",
    "txn_count_last_24h",
    "avg_amount_last_5_txns",
    "amount_to_avg_ratio",
    "geo_distance_from_last_txn_km",
    "home_distance_km",
    "is_new_device",
    "is_new_merchant_category",
    "cumulative_distinct_devices",
    "cumulative_distinct_merchant_categories",
    "account_txn_seq_num",
]


def cold_start_features(amount: float) -> dict[str, float]:
    """Defaults for an account never materialized or streamed into Feast --
    mirrors OnlineFeatureEngine's own true-first-transaction defaults
    (feature_store/online_features.py) so a cold-start score is computed
    the same way training/serving would treat a real first transaction."""
    return {
        "time_since_last_txn_sec": -1.0,
        "txn_count_last_1h": 1.0,
        "txn_count_last_24h": 1.0,
        "avg_amount_last_5_txns": amount,
        "amount_to_avg_ratio": 1.0,
        "geo_distance_from_last_txn_km": -1.0,
        # Can't be computed without the account's home coordinates (only
        # OnlineFeatureEngine loads data/raw/accounts.parquet for that);
        # -1 is out-of-training-distribution for this field specifically
        # (offline data always has a real home_distance_km) -- a known,
        # narrow cold-start gap, see docs/serving.md.
        "home_distance_km": -1.0,
        "is_new_device": 1,
        "is_new_merchant_category": 1,
        "cumulative_distinct_devices": 1,
        "cumulative_distinct_merchant_categories": 1,
        "account_txn_seq_num": 1,
    }


class PredictRequest(BaseModel):
    account_id: str
    amount: float = Field(gt=0)
    merchant_category: str
    timestamp: datetime | None = None
    transaction_id: str | None = None

    @field_validator("merchant_category")
    @classmethod
    def _valid_category(cls, v: str) -> str:
        if v not in MERCHANT_CATEGORIES:
            raise ValueError(f"unknown merchant_category {v!r}, must be one of {MERCHANT_CATEGORIES}")
        return v


class PredictResponse(BaseModel):
    transaction_id: str
    account_id: str
    fraud_score: float
    is_fraud: bool
    threshold: float
    model_version: str
    algorithm: str
    cold_start: bool
    latency_ms: float


class AppState:
    metadata: dict[str, Any]
    feature_store: Any
    triton_session: requests.Session


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not METADATA_PATH.exists():
        raise RuntimeError(
            f"{METADATA_PATH} not found -- run training/fetch_production_model.py "
            "and training/export_to_onnx.py first (see docs/serving.md)."
        )
    import json

    state.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    state.feature_store = get_feature_store(FEATURE_REPO)
    # A plain requests.Session, not httpx.AsyncClient -- measured directly
    # on this machine (see docs/serving.md's load test section): 200
    # concurrent Triton calls via httpx.AsyncClient/asyncio topped out at
    # ~131 req/s with p99 ~435ms, while the *same* calls via requests +a
    # thread pool hit ~445 req/s with p99 ~145ms. Windows' asyncio
    # ProactorEventLoop has real overhead under concurrent sockets that a
    # plain threaded requests.Session doesn't -- so both the Feast call and
    # this one go through run_in_threadpool instead of native async I/O.
    state.triton_session = requests.Session()
    yield
    state.triton_session.close()


app = FastAPI(title="Fraud Detection Gateway", lifespan=lifespan)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    # request.url.path is safe to use as a label directly: this gateway has
    # exactly 3 fixed routes (/health, /predict, /metrics), no path
    # parameters that would blow up label cardinality.
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    if MULTIPROC_DIR:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        registry = DEFAULT_REGISTRY
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def build_feature_vector(req: PredictRequest, engineered: dict[str, float], metadata: dict) -> np.ndarray:
    ts = req.timestamp or datetime.now(timezone.utc)
    row: dict[str, float] = {
        "amount": req.amount,
        "hour_of_day": ts.hour,
        "day_of_week": ts.weekday(),  # Monday=0 -- matches pandas .dt.dayofweek
        "is_weekend": int(ts.weekday() >= 5),
        **engineered,
        "is_first_txn": int(engineered["account_txn_seq_num"] == 1),
    }

    for col, fill_value in metadata["sentinel_fill_values"].items():
        if row[col] == -1:
            row[col] = fill_value

    for cat_col in metadata["category_columns"]:
        row[cat_col] = 1 if cat_col == f"category_{req.merchant_category}" else 0

    vector = [row[c] for c in metadata["feature_columns"]]
    return np.array(vector, dtype=np.float32)


def _score_with_triton_sync(vector: np.ndarray, metadata: dict) -> float:
    payload = {
        "inputs": [
            {
                "name": metadata["triton_input_name"],
                "shape": [1, metadata["n_features"]],
                "datatype": "FP32",
                "data": vector.tolist(),
            }
        ],
        "outputs": [{"name": metadata["triton_output_name"]}],
    }
    resp = state.triton_session.post(
        f"{TRITON_HTTP_URL}/v2/models/{metadata['triton_model_name']}/infer",
        json=payload,
        timeout=5.0,
    )
    resp.raise_for_status()
    body = resp.json()
    probs = body["outputs"][0]["data"]  # [P(class=0), P(class=1)]
    return float(probs[1])


async def score_with_triton(vector: np.ndarray, metadata: dict) -> float:
    return await run_in_threadpool(_score_with_triton_sync, vector, metadata)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    metadata = state.metadata

    # get_online_features is a synchronous, blocking Redis call (Feast has
    # no async API) -- run it in FastAPI's threadpool rather than awaiting
    # it directly in the event loop, otherwise it serializes every
    # concurrent request on a given worker for the duration of the Redis
    # round-trip. Confirmed empirically: without this, a 50-user load test
    # capped at ~79 req/s with p99 ~2.8s; with it, throughput and tail
    # latency both recovered (see docs/serving.md's load test section).
    resp = await run_in_threadpool(
        lambda: state.feature_store.get_online_features(
            features=[f"account_transaction_features:{f}" for f in FEAST_FETCH_FEATURES],
            entity_rows=[{"account_id": req.account_id}],
        ).to_dict()
    )

    cold_start = resp["account_txn_seq_num"][0] is None
    if cold_start:
        engineered = cold_start_features(req.amount)
    else:
        engineered = {f: resp[f][0] for f in FEAST_FETCH_FEATURES}

    vector = build_feature_vector(req, engineered, metadata)

    try:
        fraud_score = await score_with_triton(vector, metadata)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Triton inference failed: {e}") from e

    threshold = metadata["threshold"]
    is_fraud = fraud_score >= threshold
    latency_ms = (time.perf_counter() - start) * 1000

    FRAUD_PREDICTIONS.labels("true" if is_fraud else "false").inc()

    return PredictResponse(
        transaction_id=req.transaction_id or str(uuid.uuid4()),
        account_id=req.account_id,
        fraud_score=fraud_score,
        is_fraud=is_fraud,
        threshold=threshold,
        model_version=str(metadata["model_version"]),
        algorithm=metadata["algorithm"],
        cold_start=cold_start,
        latency_ms=latency_ms,
    )
