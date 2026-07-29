# Claude Code Session Plan — Fraud Detection Pipeline

This file is the source of truth for multi-session work on this project. Read `PROJECT.md` first for the full requirements — this file only sequences the work.

## How to use this file

- Work through sessions in order. Each session assumes the previous ones are done.
- At the **start** of a session: read this file, find the first unchecked session, read its "Prerequisites" and confirm they're actually true (don't assume — check the repo state).
- At the **end** of every session, without exception:
  1. Check off completed tasks below (`- [x]`).
  2. Add a dated one-paragraph note under that session's "Session log" listing what was actually done, what's left, and any deviations from the plan.
  3. Commit the work with a clear message (not "wip").
  4. If AWS resources were touched this session: run `aws sagemaker list-endpoints` and `aws sagemaker list-notebook-instances` and confirm nothing billable is left running. Note the result in the session log.
  5. If a session can't be finished in one pass, stop at a clean boundary (a working, committed state) rather than leaving something half-edited — the next session should never start by fixing broken code from an interrupted one.
- Do not skip ahead to a later session's tasks even if it seems faster — later sessions assume earlier infrastructure exists and is tested, not just written.

---

## Session 0 — Environment & repo scaffold

**Goal:** A clean repo skeleton and local dev environment, nothing functional yet.

**Prerequisites:** None.

**Tasks:**
- [x] Initialize git repo with `.gitignore` (Python, Docker, IDE, AWS credentials, data files).
- [x] Create top-level structure: `producer/`, `consumer/`, `feature_store/`, `dags/`, `training/`, `serving/`, `monitoring/`, `infra/` (Terraform), `k8s/`, `docs/`.
- [x] Add `requirements.txt` / `pyproject.toml` with pinned versions.
- [x] Add a root `docker-compose.yml` stub (services added incrementally in later sessions, not all at once).
- [x] Add `README.md` stub with the architecture diagram from PROJECT.md and a "status: in progress" note.
- [x] Confirm Docker and docker-compose work locally (`docker compose version`).

**Definition of done:** `git log` shows an initial commit, folder structure exists, empty containers can be started without errors.

**Cost/safety check:** None — no cloud resources touched.

**Session log:**
- 2026-07-28: Repo scaffolded as its own standalone git repo directly inside `fraud-detection/` (deliberately not part of the parent Documents mega-repo, which has an unrelated GitHub remote and unrelated projects). Created the 10 top-level directories with `.gitkeep` placeholders, `.gitignore` (Python/Docker/IDE/AWS/data), `requirements.txt` (dev tooling only — pytest, python-dotenv, ruff; domain libs like pandas/kafka/feast/mlflow/torch/fastapi will be pinned in the session that first introduces them, per the file's own header note), and a `README.md` stub with the architecture diagram and repo layout table. `docker-compose.yml` was initially a bare `services: {}` stub, but `docker compose up` requires at least one service to exercise, so it was changed to a single `hello-world` "scaffold-check" service; Docker Desktop had to be started (daemon wasn't running), then `docker compose up` / `down` were run successfully end-to-end and torn down cleanly — this placeholder service will be replaced by Kafka/Zookeeper in Session 2. No deviations from the plan otherwise. Dependency manager choice (requirements.txt+venv over Poetry/uv) and repo-scope decision (standalone vs. monorepo) were confirmed with the user before starting, per architectural-decision guidance in global instructions.

---

## Session 1 — Synthetic data generation

**Goal:** A reproducible synthetic transaction dataset with realistic fraud patterns.

**Prerequisites:** Session 0 complete.

**Tasks:**
- [x] Write a data generator producing transaction records per the schema in PROJECT.md §5.1.
- [x] Inject realistic fraud patterns, not pure randomness (e.g., bursts of high-velocity transactions, geo-impossible sequences, new-device + high-amount combos) — a model trained on i.i.d. random noise won't produce meaningful precision/recall numbers.
- [x] Target 5M+ rows; make row count configurable so smaller runs are possible during development.
- [x] Write the raw dataset to local disk (parquet), documented in `docs/dataset.md` with the actual fraud rate achieved.
- [x] Note explicitly: this dataset is synthetic — say so in the README, don't imply it's real financial data.

**Definition of done:** Running the generator script produces a parquet file with the documented schema and fraud rate; a notebook or script shows basic EDA (class balance, feature distributions).

**Nuance to watch for:** avoid label leakage — don't derive a feature directly from the fraud label (e.g., don't accidentally make "is_flagged" a copy of "is_fraud" under a different name).

**Cost/safety check:** None.

**Session log:**
- 2026-07-28: Built `data_generation/generate_transactions.py` — fully vectorized (numpy/pandas, no per-row Python loops over the 5M-row population; only the ~10k injected fraud rows and account-level setup use small Python loops, which is negligible). Generates accounts with a home city, per-account spend multiplier, and overdispersed activity level, then base transactions via a repeat/uniform-timestamp approach followed by a sort per account (avoids needing an explicit Poisson-process loop). Injected 3 fraud patterns (velocity burst, geo-impossible sequence, new-device+high-amount) split ~40/30/30 of the fraud budget, merged into the base population *before* computing the 15 engineered features, so features emerge naturally rather than being hand-set. Ran the full default config (`--n-rows 5000000 --n-accounts 50000 --seed 42`): 5,010,000 rows, 0.1996% fraud rate, ~88s runtime, written to `data/raw/transactions.parquet` (gitignored, not committed — regenerate via the documented command). `data_generation/eda.py` prints class balance / per-feature fraud-vs-legitimate means / fraud rate by category, and saves 3 plots to `docs/eda/` (committed). Full methodology, schema, sentinel-value conventions (`-1` for "no prior transaction"), and measured signal strength are in `docs/dataset.md`. Verified no label leakage: no engineered feature reads `is_fraud`, all derived purely from timestamp/amount/device/category/location. One deviation from a literal read of the "10-15 features" spec: settled on 15 exactly, replacing two originally-planned rolling-time-window "distinct count in trailing Nd" features with cheaper *cumulative* distinct-count features (`cumulative_distinct_devices`, `cumulative_distinct_merchant_categories`) — same modeling intent (device/category novelty over time), avoids a much slower rolling-distinct-in-time-window pandas operation at 5M-row scale.

---

## Session 2 — Kafka ingestion layer

**Goal:** A working producer/consumer pair streaming synthetic transactions through Kafka.

**Prerequisites:** Session 1 complete (need data to stream).

**Tasks:**
- [ ] Add Kafka + Zookeeper (or KRaft mode) to `docker-compose.yml`.
- [ ] Producer: replays the synthetic dataset as a live stream at a configurable rate (transactions/sec), serialized as Avro or JSON with a documented schema.
- [ ] Consumer: reads from the topic, writes raw events to a local "data lake" path, and (stub for now) forwards records toward the feature engineering step.
- [ ] Add basic error handling: what happens on a malformed message, a consumer restart, an out-of-order timestamp.

**Definition of done:** `docker compose up` starts Kafka; running the producer and consumer scripts shows messages flowing end-to-end, verified by counting records landing in the data lake path.

**Nuance to watch for:** decide and document the offset/replay strategy (earliest vs latest) — this matters for reproducible demos.

**Cost/safety check:** None — local only.

**Session log:**
- _(fill in after running this session)_

---

## Session 3 — Feature store (Feast + Redis)

**Goal:** Feature definitions shared between an offline (training) store and an online (serving) store.

**Prerequisites:** Session 2 complete.

**Tasks:**
- [ ] Add Redis to `docker-compose.yml`.
- [ ] Define Feast feature views for the engineered features (velocity, average ticket size, geo-distance, etc.) against the offline parquet data.
- [ ] Materialize features into the Redis online store.
- [ ] Wire the Kafka consumer (from Session 2) to also push freshly computed features into the online store as records arrive.
- [ ] Write a small script that fetches features for a given `account_id` from the online store and confirms they match what training will see offline — this consistency check is the point of the whole layer, don't skip it.

**Definition of done:** Same feature values are retrievable both from the offline store (for training) and the online store (for serving), for a given entity key.

**Cost/safety check:** None.

**Session log:**
- _(fill in after running this session)_

---

## Session 4 — Airflow orchestration + data validation

**Goal:** A DAG that ties ingestion, validation, and feature engineering together on a schedule.

**Prerequisites:** Sessions 1–3 complete.

**Tasks:**
- [ ] Add Airflow to `docker-compose.yml` (webserver, scheduler, Postgres metadata DB).
- [ ] Write a DAG: ingest batch → Great Expectations validation → feature engineering → materialize to Feast.
- [ ] Add at least 3 meaningful Great Expectations checks (e.g., no nulls in required fields, amount within plausible bounds, fraud rate within an expected range) — not just boilerplate "column exists" checks.
- [ ] Configure the DAG schedule and confirm a manual trigger runs end-to-end.

**Definition of done:** Airflow UI shows a green DAG run touching every stage; a deliberately broken input (inject a bad row) causes the validation step to fail visibly rather than silently pass through.

**Cost/safety check:** None.

**Session log:**
- _(fill in after running this session)_

---

## Session 5 — Local model training + MLflow tracking

**Goal:** All 4 models trained locally, tracked and comparable in MLflow.

**Prerequisites:** Sessions 1–4 complete (need validated features to train on).

**Tasks:**
- [ ] Add MLflow tracking server to `docker-compose.yml`.
- [ ] Implement training scripts for Logistic Regression, Random Forest, Isolation Forest, and the LSTM (PyTorch).
- [ ] Handle class imbalance explicitly (SMOTE or class weighting) — log which approach per model and why.
- [ ] Log parameters, metrics (precision, recall, F1, AUC-PR — **not** plain accuracy), and model artifacts to MLflow for every run.
- [ ] Produce a comparison table/plot across the 4 models.

**Definition of done:** MLflow UI shows 4+ tracked runs with real metrics; a written note in `docs/model_comparison.md` states which model is the recommended production candidate and why.

**Nuance to watch for:** make sure the train/validation/test split respects time order (no future transactions leaking into training) — this is a common and easy-to-miss fraud-detection-specific bug.

**Cost/safety check:** None — all local.

**Session log:**
- _(fill in after running this session)_

---

## Session 6 — SageMaker training job (learning objective)

**Goal:** At least one model retrained as a managed SageMaker Training Job, with the artifact registered back into MLflow.

**Prerequisites:** Session 5 complete. **Read PROJECT.md §7 (cost management rules) again before starting.**

**Tasks:**
- [ ] Set an AWS Budget alert (e.g., $10) before touching any AWS resource this session.
- [ ] Confirm AWS CLI is configured with a scoped IAM role (not root credentials).
- [ ] Package the training script (recommend Random Forest or the LSTM) for SageMaker's script-mode training.
- [ ] Upload training data to S3 (a dedicated bucket, lifecycle-ruled to expire objects after N days to avoid storage creep).
- [ ] Launch a SageMaker Training Job on a free-tier-eligible instance (`ml.m5.large`), using a spot instance if feasible.
- [ ] On completion, pull the model artifact and register it in the local MLflow registry alongside the local-trained models, tagged as "trained via SageMaker."
- [ ] Do **not** deploy a persistent SageMaker endpoint in this session unless immediately followed by deletion in the same session (see next task).
- [ ] If an endpoint was created for testing, delete it before ending the session.

**Definition of done:** MLflow registry shows a SageMaker-trained model alongside the local ones; `aws sagemaker list-endpoints` returns empty at session end.

**Cost/safety check (mandatory, do not skip):**
- [ ] Ran `aws sagemaker list-endpoints` — confirmed empty.
- [ ] Ran `aws sagemaker list-notebook-instances` — confirmed none left running.
- [ ] Confirmed the S3 bucket has a lifecycle rule or was manually cleaned up.
- [ ] Noted actual AWS spend incurred this session in the session log below.

**Session log:**
- _(fill in after running this session — include actual dollar cost incurred)_

---

## Session 7 — Model registry + staged rollout

**Goal:** Formal staged promotion logic: Staging → Shadow → Canary → Production.

**Prerequisites:** Session 6 complete.

**Tasks:**
- [ ] Define the 4 registry stages in MLflow and the promotion criteria between them (e.g., shadow must match or beat current production recall before canary).
- [ ] Implement a simple shadow-mode harness: new model scores traffic alongside the production model, predictions logged but not acted on.
- [ ] Implement a canary split (e.g., 10% of traffic) with a rollback path if metrics degrade.
- [ ] Document the promotion criteria in `docs/model_registry.md`.

**Definition of done:** A model can be walked through all 4 stages manually, with the criteria and rollback path documented and demonstrable.

**Cost/safety check:** None if working locally against the MLflow registry only.

**Session log:**
- _(fill in after running this session)_

---

## Session 8 — Serving layer (Triton + FastAPI)

**Goal:** A real-time `/predict` endpoint backed by Triton, pulling features from the online store.

**Prerequisites:** Sessions 3, 5 (and ideally 7) complete.

**Tasks:**
- [ ] Add Triton Inference Server to `docker-compose.yml` (CPU mode is fine if no local GPU).
- [ ] Export the production-candidate model(s) to a Triton-compatible format (ONNX recommended for portability across the 4 model types).
- [ ] Build a FastAPI gateway: given a transaction, fetch account features from the Feast/Redis online store (not from the request payload — this is the point of Session 3), call Triton, return a fraud score.
- [ ] Load-test the endpoint (e.g., with `locust` or `k6`) at a sustained rate and record actual p50/p95/p99 latency.

**Definition of done:** `/predict` returns a real score for a real transaction; a load test report exists with actual measured latency numbers (target: p99 < 50ms at 500 TPS — report the real number even if it misses the target).

**Cost/safety check:** None — local serving.

**Session log:**
- _(fill in after running this session)_

---

## Session 9 — Monitoring & drift detection

**Goal:** Live observability of the serving layer, plus demonstrable drift detection.

**Prerequisites:** Session 8 complete.

**Tasks:**
- [ ] Add Prometheus + Grafana to `docker-compose.yml`.
- [ ] Instrument the FastAPI gateway to expose request rate, latency histogram, and fraud-flag rate as Prometheus metrics.
- [ ] Build a Grafana dashboard visualizing these in real time.
- [ ] Add an Evidently AI report comparing live feature distributions to the training distribution, run on a schedule via the Airflow DAG.
- [ ] Deliberately inject a synthetic distribution shift (e.g., replay data with an altered amount distribution) and confirm Evidently flags it — this is the demo moment, make sure it actually works, don't just wire it up and hope.

**Definition of done:** Grafana dashboard updates live while the load test from Session 8 runs; a drift report exists showing a flagged shift from the injected test.

**Cost/safety check:** None.

**Session log:**
- _(fill in after running this session)_

---

## Session 10 — Deployment (Docker Compose, Kubernetes, Terraform)

**Goal:** The full stack runs with one command; a Kubernetes path exists for the serving layer; Terraform scopes the SageMaker pieces only.

**Prerequisites:** Sessions 0–9 complete.

**Tasks:**
- [ ] Finalize `docker-compose.yml` so `docker compose up` brings up every service from Sessions 2–9 in the correct dependency order (use `depends_on` and healthchecks, not just ordering).
- [ ] Write Kubernetes manifests (Deployment, Service) for the serving layer; test against a local cluster (kind or minikube).
- [ ] Write a narrow Terraform module covering only the SageMaker training job resources and the S3 bucket from Session 6 — not the whole stack. Include the lifecycle rule from Session 6.
- [ ] Add a `terraform destroy` reminder/script so the AWS footprint can be torn down completely after a demo.

**Definition of done:** A fresh clone of the repo, on a clean machine, can run `docker compose up` and reach a working `/predict` endpoint within a few minutes, no manual steps beyond documented prerequisites (Docker installed, AWS creds configured if using the Terraform piece).

**Cost/safety check:** If Terraform was applied, confirm `terraform destroy` was run and AWS resources are gone before ending the session.

**Session log:**
- _(fill in after running this session)_

---

## Session 11 — Metrics, README, and portfolio polish

**Goal:** The repo is presentable: real numbers, clear README, working demo, resume bullets finalized.

**Prerequisites:** Sessions 0–10 complete.

**Tasks:**
- [ ] Replace every placeholder metric in the README with the real, measured numbers from Sessions 5, 6, 8, and 9.
- [ ] Add the architecture diagram (from PROJECT.md) to the README.
- [ ] Record a short demo (screen recording or GIF): a transaction flowing through the system, the Grafana dashboard updating, and a drift alert firing.
- [ ] Write a "what I'd do with more time" section — this reads as maturity, not incompleteness.
- [ ] Finalize resume bullets using the real numbers and update the master YAML resume.
- [ ] Do a final pass confirming no AWS resources are left running anywhere (`aws sagemaker list-endpoints`, `list-notebook-instances`, S3 bucket check).

**Definition of done:** The repo, README, and demo are ready to link from the portfolio site and resume, with no placeholder numbers remaining anywhere.

**Cost/safety check:** Final full AWS resource audit — confirm zero ongoing spend.

**Session log:**
- _(fill in after running this session)_
