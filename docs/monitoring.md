# Monitoring & Drift Detection — Session 9

Prometheus + Grafana for live serving-layer observability, plus an
Evidently-based drift report wired into the existing Airflow batch
reconciliation DAG (Session 4), with a real, live-verified demo of an
injected distribution shift actually getting flagged.

## Architecture

```
FastAPI gateway (serving/app.py, local process, port 8090)
    |-- /metrics (Prometheus multiprocess registry) --> scraped by prometheus:9090
Triton (docker, port 8002)
    |-- /metrics (built-in) --------------------------> scraped by prometheus:9090
prometheus:9090 --> grafana:3000 (auto-provisioned datasource + dashboard)

Airflow DAG (dags/fraud_pipeline_dag.py), after feature_engineering:
    training/export_reference_distribution.py (run once, offline)
        --> data/processed/reference_features.parquet + reference_fill_values.json
    drift_report task: build_feature_matrix(features_latest.parquet) vs. reference
        --> Evidently DataDriftPreset --> monitoring/drift_reports/drift_<ts>.html
```

## Two scope decisions confirmed with the user before building

1. **"Live" data for drift comparison** = the existing Airflow batch
   reconciliation task's output (`features_latest.parquet`), not a new log
   of `/predict` request features. `/predict` is deliberately read-only
   against Feast (Session 8) and its fetched features are documented as
   stale/partial relative to the transaction actually being scored; the
   DAG's `feature_engineering` task, by contrast, recomputes the complete,
   correct 15-feature vector from raw ingested data on a schedule --
   already the right shape for a distributional comparison, no new data
   path needed.
2. **Synthetic drift injection** = a new `--amount-multiplier` flag on
   `producer/produce.py` that scales `amount` before sending, replayed
   through the real Kafka -> consumer -> lake -> DAG path end-to-end,
   rather than a script that writes directly into `features_latest.parquet`
   and skips ingestion entirely.

## Prometheus + Grafana

`docker-compose.yml` adds `prometheus` (`prom/prometheus:v3.0.1`) and
`grafana` (`grafana/grafana:11.4.0`). Grafana's datasource
(`monitoring/grafana/provisioning/datasources/datasource.yml`, fixed
`uid: prometheus` so dashboard JSON can reference it) and dashboard
(`monitoring/grafana/provisioning/dashboards/fraud_detection_dashboard.json`)
are both provisioned from files mounted read-only into the container --
`docker compose up` alone reproduces the exact same dashboard, no
click-through setup. Verified via Grafana's own API after a fresh
`docker compose up -d`:

```
GET /api/datasources        -> Prometheus datasource present, uid=prometheus
GET /api/search?type=dash-db -> "Fraud Detection - Serving Layer" present
```

Dashboard has 5 panels: request rate (by path/status), `/predict` latency
p50/p95/p99 (`histogram_quantile` over the new latency histogram),
fraud-flag rate, predictions-by-outcome, and a bonus Triton
inference-rate panel (`nv_inference_count`, Triton's own built-in metric --
`docker-compose.yml`'s port 8002 comment had flagged "wired up properly in
Session 9" since it was exposed but not scraped until now).

`monitoring/prometheus/prometheus.yml` scrapes two targets: the FastAPI
gateway via `host.docker.internal:8090` (it runs as a local host process,
not a compose service -- Session 8's decision, unchanged; containerizing
it is Session 10's job) and `triton:8002` on the compose network directly.
Both confirmed `health: up` via `GET /api/v1/targets` after starting the
gateway.

## FastAPI gateway instrumentation

`serving/app.py` adds three metrics via `prometheus_client`:
`http_requests_total` (Counter, labels method/path/status_code, via a
middleware wrapping every request), `http_request_duration_seconds`
(Histogram, same labels minus status_code), and `fraud_predictions_total`
(Counter, label `is_fraud`, incremented inside `/predict`). `/metrics`
exposes them.

### Gotcha #1: multiprocess metrics

`docs/serving.md` documents running this gateway as `uvicorn --workers 4`
(a real throughput requirement on this Windows/`ProactorEventLoop` setup --
see that doc's load-test section). That's 4 separate OS processes;
`prometheus_client`'s default in-memory registry is per-process, so a naive
`/metrics` scrape would only ever reflect whichever one worker happened to
handle that specific request. Fixed with `prometheus_client`'s documented
multiprocess mode (`PROMETHEUS_MULTIPROC_DIR`) and a new launcher,
`serving/run_server.py`, replacing the old `python -m uvicorn ...` command
in `docs/serving.md` -- it wipes/recreates the multiprocess directory
*once*, in the parent, before workers fork (each worker doing it
independently would race and delete each other's files).

Getting this actually working took two more fixes, both found only by
running it, not by reading the code:

- **`sys.path` vs. `PYTHONPATH`**: `uvicorn --workers` on Windows uses
  `multiprocessing`'s "spawn" start method, which pickles the *parent
  process's current* `sys.path` into each worker's prep-data and the
  worker overwrites its own `sys.path` with that copy verbatim -- it does
  **not** rebuild `sys.path` from `PYTHONPATH` at worker startup. Setting
  `os.environ["PYTHONPATH"]` in the launcher had no effect; only mutating
  `sys.path` directly in the parent, before calling `uvicorn.run()`, fixed
  the `ModuleNotFoundError: No module named 'serving'` every worker hit.
  (`python -m uvicorn serving.app:app ...`, the original Session 8 command,
  happened to work because `-m` puts the invoking cwd on `sys.path` for the
  process itself; `python serving/run_server.py` instead puts `serving/`
  -- this script's own directory -- on `sys.path[0]`.)
- **Stale processes left running from Session 8**: something was already
  listening on port 8090 (an old, un-instrumented `serving/app.py`
  instance, confirmed via `curl -i /metrics` returning 404 from a build
  with no `/metrics` route) plus two more orphaned `uvicorn --workers 4`
  process trees. All killed before starting the new launcher.

Verified end-to-end: fired 8 `/health` and 6 `/predict` requests spread
across the 4 workers (`http_requests_total{path="/health"}` read back as
exactly 9 -- 8 plus one earlier probe -- and `fraud_predictions_total`
split 4 false / 2 true, matching the 6 `/predict` calls exactly), proving
the aggregation is correct across processes, not just non-crashing.

## Evidently drift report

`training/export_reference_distribution.py` (run once, regenerate only if
the dataset or feature schema changes) builds `X_train` via the existing
chronological split (`training/data_prep.py`, unchanged from Session 5),
samples 100,000 of the 3,507,193 training rows with a fixed seed, and
writes `data/processed/reference_features.parquet` +
`reference_fill_values.json`. Deliberately **not** reusing
`serving/model_metadata.json` (Session 8's export) -- that's a *serving*
concern about the currently-deployed model; this is a *monitoring* concern
about the training distribution the model was built from, and keeping them
separate means neither has to know about the other.

`dags/fraud_pipeline_dag.py` adds a `drift_report` task after
`feature_engineering`, running in parallel with `materialize_to_feast`
(both depend only on the features path, not on each other). It applies the
same `build_feature_matrix` transform training uses (one-hot categories +
sentinel fill, using the persisted `reference_fill_values.json`, not a
value re-derived from this batch) to get the freshly computed batch onto
the identical 29-column schema as the reference sample, then runs
Evidently's `Report(metrics=[DataDriftPreset()], include_tests=True)`.
This is a **report, not a gate**: unlike `validate_batch` (which fails the
DAG on malformed data, by design), drift is informational -- the data is
still well-formed, its distribution just moved -- so `drift_report` never
fails the DAG; it logs a clear `DRIFT DETECTED` / `No dataset drift
detected` line with the drifted-column share and writes an HTML+the
underlying JSON snapshot to `monitoring/drift_reports/`.

`training/data_prep.py` had its module-level `from sklearn.metrics import
...` moved into the individual functions that use it (lazy import), so
this module is importable in the Airflow image without a scikit-learn
dependency this project would otherwise have to declare there (evidently
happens to pull scikit-learn in transitively for its own internal drift
methods, but that's incidental to evidently, not something this project's
own code should rely on).

### KNOWN ISSUE (pre-dates Session 9): `feature_engineering` computes history-dependent features from only the streamed batch, not each account's true history

The ~48% baseline drifted-column share described below is **not** a benign
"narrow window vs. long window" statistical nuance -- that was this doc's
first-draft explanation, and it understated a real, pre-existing
correctness bug in Session 4's DAG that this session's drift-comparison
work happened to surface and quantify for the first time.

`feature_engineering` (`dags/fraud_pipeline_dag.py`, built in Session 4)
calls `compute_engineered_features(df, accounts)` on **only whatever has
landed in the lake so far** -- a few thousand rows from streaming-replay
testing across Sessions 2-4 and this session's drift demo -- not on each
account's true, full transaction history from the original 5M-row
dataset. Six of the 15 engineered features are history-dependent
(`account_txn_seq_num`, `cumulative_distinct_devices`,
`cumulative_distinct_merchant_categories`, `is_new_device`,
`is_new_merchant_category`, `is_first_txn`), and computing them from an
isolated small batch makes every account look brand-new, regardless of
its real history:

```
Real training data (data/processed/reference_features.parquet):
    account_txn_seq_num mean=53.1, median=41, only 1.4% "first transaction"

This DAG's batch output (data/processed/features_latest.parquet):
    account_txn_seq_num mean=1.07, median=1,  93.7% "first transaction"
```

That gap -- not calendar effects, not category-mix noise -- is what
mainly drives the 14/29 (48.3%) baseline drifted-column share described
below. It also means **this bug has consequences beyond the drift
report**: `materialize_to_feast` pushes these artificially-reset values
straight into the live Feast/Redis online store on every DAG run,
potentially overwriting whatever the correct real-time
`OnlineFeatureEngine` (Session 3) had already computed for those same
accounts. Session 4's docs describe this DAG as "a correctness backstop"
for the online store; for these 6 columns, as currently written, it's the
opposite.

Confirmed with the user: leaving this as a documented known issue for a
dedicated future session to fix properly (most likely by having
`feature_engineering` read each account's prior state from the offline
dataset or from Feast's own online store before recomputing, rather than
computing from the batch in isolation), not patching it inside this
session's scope.

### The baseline drift number itself (still real, still useful for this demo)

Comparing a narrow batch (whatever's landed in the lake, subject to the
bug above) against a reference sampled from the entire training period
lands at 14/29 columns "drifted" (48.3% share) even with *no* injected
shift -- confirmed reproducible across isolated tests at
n=2,000/5,000/10,000 rows sliced from different points in the raw dataset.
The category one-hot columns are all near-zero drift score; the six
history-dependent columns above account for most of the 14. This does
mean the aggregate `DriftedColumnsCount` test (default 0.5 share
threshold) starts out close to the line for reasons unrelated to genuine
distribution shift -- worth fixing alongside the bug above, since a
cleaner baseline would make the drift report more trustworthy as a signal,
not just a demo. For *this* session's demo specifically, it had one
practical upside: the injected `amount` shift had to and reliably did tip
an already-nonzero, already-realistic baseline over the threshold, rather
than moving off a suspiciously clean 0%.

### Live-verified demo run

1. **Baseline** (before injecting anything): ran the drift comparison
   against the DAG's own `features_latest.parquet` as it stood before any
   drift injection (2,000 rows from earlier session testing) -- **14/29
   columns drifted, share 0.483, `drift_detected: False`**. Matches the
   isolated-slice tests above exactly.
2. **Injection**: `python producer/produce.py --limit 5000 --rate 1000
   --amount-multiplier 5.0` replayed 5,000 transactions with `amount`
   scaled 5x through the real Kafka topic; the consumer landed all 5,000
   (0 malformed, 0 dead-lettered) into the lake, bringing it to 13,000
   accumulated rows (`ingest_batch` re-reads and dedupes the *entire* lake
   every run, per Session 4's design -- the drifted batch is diluted into
   the full accumulated history, not compared in isolation).
3. **Triggered `fraud_pipeline` manually** (`airflow dags trigger`) against
   that 13,000-row lake. Real, live `drift_report` task output:

   ```
   DRIFT DETECTED: 15/29 columns drifted (51.7%) vs. the training reference
   distribution. Report: monitoring/drift_reports/drift_20260805T072646.html
   ```

   The `amount` column's own drift score jumped from ~0.02 (baseline,
   threshold 0.1) to ~1.8-1.9 in isolated pre-checks of the same
   multiplier at multiple batch sizes -- an ~90x increase, the single
   column that newly crossed its threshold and tipped the aggregate share
   from 14/29 (0.483, pass) to 15/29 (0.517, fail). Full HTML report
   committed as evidence is not checked into git (regenerated artifact,
   gitignored) but reproducible via the steps above; the run's JSON
   snapshot is embedded in the Airflow task log referenced above.

### Bugs found only by actually running this, not by reading the code

- **Airflow's per-service image build**: `x-airflow-common`'s `build:`
  block is shared via a YAML anchor across `airflow-init`,
  `airflow-webserver`, and `airflow-scheduler`, but Compose still builds
  **three separate images** (`fraud-detection-airflow-init`,
  `-airflow-webserver`, `-airflow-scheduler`) since none of them set an
  explicit shared `image:` name. Rebuilding only `airflow-init` (to pick up
  `evidently` in `requirements-airflow.txt`) left the scheduler -- where
  `LocalExecutor` actually runs task subprocesses -- on the old image,
  producing `ModuleNotFoundError: No module named 'evidently'` the first
  time `drift_report` actually ran. All three services need
  `docker compose build` together whenever `requirements-airflow.txt`
  changes; worth remembering for Session 10's compose finalization.
- **`evidently` transitively pulling an unconstrained `cryptography`**:
  installing `evidently==0.7.21` silently upgraded `cryptography` to
  50.0.0 (evidently declares no upper bound of its own), which breaks
  `pyOpenSSL` (`AttributeError: module 'lib' has no attribute
  'GEN_EMAIL'`) -- surfaced not at install time but the first time
  `validate_batch` ran, because Great Expectations' `compatibility.aws`
  module imports `boto3` unconditionally (not only when S3 is actually
  used), and `boto3` -> `botocore` -> `urllib3.contrib.pyopenssl` ->
  `OpenSSL`. `pip install` only printed a non-blocking resolver warning
  ("pyopenssl 24.3.0 requires cryptography<45,>=41.0.5, but you have
  cryptography 50.0.0") -- caught by running the DAG, not by reading pip's
  output. Fixed by pinning `cryptography==44.0.3` explicitly in
  `requirements-airflow.txt`.
- **`monitoring/` wasn't mounted into the Airflow container**: every other
  directory the DAG touches (`data_generation`, `feature_store`,
  `validation`, `data`, and the new `training`) has its own bind mount in
  `docker-compose.yml`'s `airflow-common` volumes block; `monitoring/` was
  missed, so `PROJECT_ROOT / "monitoring" / "drift_reports"` wasn't a real
  path inside the container and `Path.mkdir()` failed with
  `PermissionError` (not a clearer "not found") the first time
  `drift_report` tried to write its report. Added the missing mount.

## Grafana dashboard: live-updating, verified

Ran the Session 8 Locust load test again (`serving/locustfile.py`,
50 users, 90s) purely to generate live traffic and confirm the dashboard's
exact PromQL queries return real, moving numbers against Prometheus's
`/api/v1/query` during the run (rather than trusting the JSON is wired up
correctly and hoping):

| Query (mid-load-test) | Result |
|---|---|
| `sum(rate(http_requests_total{path="/predict"}[30s]))` | 106-113 req/s |
| `histogram_quantile(0.95, ... duration_seconds_bucket{path="/predict"}[30s])` | ~441 ms |
| `sum(rate(fraud_predictions_total[30s]))` | non-zero, moving |

Full load test result (50 users, 90s, 0 failures, 10,565 requests):

| Metric | Value |
|---|---|
| Requests/s (final) | 118.4 |
| p50 | 370 ms |
| p95 | 900 ms |
| p99 | 1200 ms |
| Max | 2216 ms |

(Not a re-measurement of Session 8's own dedicated load-test numbers --
that stands as-is in `docs/serving.md`. This run's purpose was solely to
generate live traffic and prove the Grafana dashboard reflects it in real
time, which it does.) Raw CSVs: `docs/eda/load_test_session9_*.csv`.

## What's left running

Kafka, Redis, Postgres, Airflow (webserver/scheduler), MLflow, Triton,
Prometheus, and Grafana are all up via `docker compose up -d`. The FastAPI
gateway (`serving/run_server.py`) and a Kafka consumer
(`consumer/consume.py`) were left running as local processes for this
session's verification; both are safe to stop and restart at any time (no
persistent state depends on them staying up between sessions, same as
Session 8).

## What I'd try with more time

- **Fix the known issue above**: give `feature_engineering` access to each
  account's true prior state (offline dataset or Feast's own online store)
  before recomputing the 6 history-dependent features, so
  `materialize_to_feast` stops writing artificially-reset values into the
  online store and the drift report's baseline reflects genuine
  distribution shift rather than a batch-recomputation artifact. This is
  the highest-priority item on this list -- it affects serving correctness,
  not just the drift demo.
- A drift-injection mode that doesn't dilute into the full accumulated
  lake -- e.g. giving `ingest_batch` an incremental read mode -- so the
  demo compares a single batch in isolation rather than the batch mixed
  with everything previously landed.
- Alerting (Prometheus Alertmanager, or an Airflow callback) on
  `drift_detected=True` instead of a log line -- deliberately out of scope
  for a "report, not a gate" first pass.
