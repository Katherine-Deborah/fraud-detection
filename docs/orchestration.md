# Orchestration (Airflow + Great Expectations)

Session 4 adds a scheduled batch job that ties ingestion, data validation,
feature engineering, and the feature store together — distinct from (and a
correctness backstop for) the Session 2/3 real-time Kafka path.

## Why a separate batch path, not just the streaming one

The Kafka consumer (`consumer/consume.py`) already computes features per
transaction in real time and pushes them into Feast as they arrive. But
`docs/feature_store.md` documents a known gap: that path is **not
idempotent** — replaying already-processed messages (e.g. after a crash, or
under a fresh consumer group for testing) double-counts into the rolling
features.

This DAG (`dags/fraud_pipeline_dag.py`, `fraud_pipeline`) is a periodic
*reconciliation* job that doesn't share that weakness: it re-reads
everything landed in the data lake so far, deduplicates by
`transaction_id`, and recomputes every feature from scratch with the same
vectorized function the offline training dataset was built with
(`data_generation.generate_transactions.compute_engineered_features`) — no
incremental Redis state to drift. Great Expectations validates the raw
deduplicated batch before any of that computation happens, so a corrupted
batch never reaches feature engineering or the online store.

## DAG stages

```
ingest_batch → validate_batch → feature_engineering → materialize_to_feast
```

1. **`ingest_batch`** — reads every `data/lake/dt=*/transactions.jsonl`
   file written by the consumer (excluding `_dead_letter/`, which the
   consumer already quarantines), concatenates them, and drops duplicate
   `transaction_id`s (`keep="last"`). Writes `data/processed/batch_latest.parquet`.
   Raises `FileNotFoundError` if the lake is empty — running the producer/
   consumer (Session 2) is a prerequisite, not something this task should
   silently paper over.
2. **`validate_batch`** — runs the Great Expectations suite in
   `validation/great_expectations_checks.py` against the raw batch. Raises
   `BatchValidationError` (which fails the Airflow task) on any violation.
   See "Great Expectations checks" below.
3. **`feature_engineering`** — resolves each record's `location` city name
   to city-centroid coordinates (`feature_store.online_features.CITY_COORDS`,
   the same table the real-time engine uses) and calls the offline
   generator's `compute_engineered_features` over the whole batch at once.
   Geo features (`geo_distance_from_last_txn_km`, `home_distance_km`)
   therefore use centroid coordinates rather than the offline dataset's
   per-transaction jitter — the same documented approximation as the
   streaming path (`docs/feature_store.md`'s "Geo-feature tolerance"
   section), for the same reason: the lake only carries a city name, not
   jittered lat/lon.
4. **`materialize_to_feast`** — pushes the computed feature rows into the
   Feast online store in one `store.push(..., to=PushMode.ONLINE)` call.

Schedule: `@daily`, `catchup=False`. Chosen to match the lake's
`dt=YYYY-MM-DD` partitioning; a production system would tune this to
actual data arrival rate, not a fixed calendar cadence.

## Great Expectations checks

Run via Great Expectations 1.19's Ephemeral Data Context (in-memory, no
on-disk GX project — nothing here needs to persist between runs beyond what
the DAG run's own pass/fail state already captures). Six checks, not just
schema boilerplate:

| Check | Catches |
|---|---|
| No nulls in any of the 8 required raw fields | Missing/corrupted fields from a bad producer run |
| `transaction_id` uniqueness | Exactly the at-least-once double-landing gap documented in `docs/kafka.md` — this is the check that would have caught it if `ingest_batch`'s dedup step were skipped |
| `amount` between $0.01 and $50,000 | Unit bugs, parsing errors, sign flips — not meant to flag genuinely large legitimate purchases |
| `is_fraud` in `{True, False}` | A corrupted/non-boolean label column |
| `account_id` matches `^acct_\d{6}$` | Malformed or truncated entity keys |
| `transaction_id` not blank/whitespace-only | Empty-string IDs that would otherwise pass a not-null check |

The fraud-rate-range check described in earlier planning was dropped after
testing at real batch sizes: a batch of a few thousand streamed
transactions legitimately lands at 0% fraud most of the time (the full
dataset's rate is ~0.2%), so a "fraud rate must be within range" check adds
no detection power at this scale without a much bigger sample — the
`is_fraud`-domain and uniqueness checks above already cover the realistic
failure modes (a flipped label column, a double-counted fraud row).

## Verified: a green run, and a visibly-failing one

Ran end-to-end against the 8,000-record lake already on disk from Session
2/3 testing:

- **Clean run**: `ingest_batch` → `validate_batch` → `feature_engineering` →
  `materialize_to_feast` all succeeded; confirmed via
  `get_online_features` that `materialize_to_feast` actually wrote fresh
  values (not a no-op) for an account touched by the batch.
- **Deliberately broken input**: appended one record with `amount: -999.0`
  to the lake file, triggered the DAG again. `validate_batch` failed with
  `BatchValidationError: ... expect_column_values_to_be_between(amount): 1
  unexpected of 2001`, and `feature_engineering`/`materialize_to_feast`
  were correctly marked `upstream_failed` rather than running on
  unvalidated data. Removed the injected record afterward and re-ran to
  leave the DAG's last run green.

## Running it locally

```bash
docker compose up -d postgres
docker compose up airflow-init                 # one-time: db migrate + admin user
docker compose up -d airflow-webserver airflow-scheduler redis kafka

# trigger manually (or wait for the daily schedule)
docker compose exec airflow-scheduler airflow dags trigger fraud_pipeline
```

UI at http://localhost:8080. Needs `.env` populated from `.env.example`
first (a Fernet key and a webserver secret key — commands to generate both
are in that file's comments).

## Two real problems hit and fixed

**Feast's own dependency set clashes with two *unused* Airflow providers,
harmlessly.** Building the custom Airflow image
(`Dockerfile.airflow` = `apache/airflow:2.10.4-python3.11` +
`requirements-airflow.txt`, adding pandas/pyarrow/feast[redis]/
great_expectations on top) produced pip resolver warnings about
`apache-airflow-providers-snowflake` and `apache-airflow-providers-google`
wanting an older pandas, and `snowflake-snowpark-python` wanting an older
cloudpickle. This project uses neither Snowflake nor Google Cloud
providers, and `docker run ... python -c "import airflow, feast,
pandas, great_expectations"` confirmed every package this project actually
uses imports and runs cleanly — Airflow core's own requirements (SQLAlchemy,
Flask, etc.) were untouched by the installs. Deliberately not switched to
an isolated-venv operator (`PythonVirtualenvOperator`/`ExternalPythonOperator`)
over this, since the actual conflict surface is zero for what this DAG
uses.

**Constructing `FeatureStore(config=...)` without also passing `repo_path`
silently breaks the registry lookup.** `feature_store/store_utils.py`
exists so the Airflow container can override Feast's Redis connection
string (`redis:6379`, the compose network's service name) without touching
`feature_store.yaml` (`localhost:6380`, correct for host-venv scripts — see
`docs/feature_store.md`'s WSL port-collision writeup). The first version
loaded a `RepoConfig` and passed only `config=` to `FeatureStore()`; Feast's
`__init__` falls back to `os.getcwd()` for `repo_path` whenever it isn't
given explicitly, even though `config` was provided — and it uses
`repo_path` to resolve the registry's relative `path: registry.db`. Inside
the container that resolved to the wrong (empty) location, so
`store.push()` failed with `PushSourceNotFoundException: Unable to find
push source 'transactions_push_source'` despite `feast apply` having been
run correctly on the host. Fixed by passing both `repo_path=repo_path` and
`config=config` together — confirmed by `materialize_to_feast` succeeding
and a subsequent `get_online_features` call returning the freshly pushed
values.

## What's stubbed for later sessions

- `feature_engineering`'s rolling-window features (`avg_amount_last_5_txns`,
  `txn_count_last_1h`/`24h`) are computed over *only* the current batch's
  history for each account, not that account's full lifetime history — fine
  for this session's demo-scale lake, but a production version would need
  to seed each run with prior state (or simply widen `ingest_batch`'s glob
  to the full lake every time, which is what it already does today; this
  will need revisiting once the lake is large enough that full-history
  reprocessing every run stops being cheap).
- No retries/alerting configured on task failure — default Airflow
  behavior (no retry, task goes red) was left as-is for this session.
