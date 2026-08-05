# Fraud Detection Pipeline

**Status: in progress — Sessions 0–10 complete (repo scaffold through deployment: dataset generation, Kafka ingestion, Feast/Redis feature store, Airflow orchestration, local + SageMaker model training, MLflow staged registry, Triton/FastAPI serving, Prometheus/Grafana/Evidently monitoring, and a finalized Docker Compose stack + Kubernetes manifests + narrow Terraform module). Session 11 (metrics/README/demo polish) remains.**

An end-to-end, portfolio-grade fraud detection system: streaming ingestion, a
feature store, orchestrated training, cloud-based training (SageMaker), a
model registry, real-time serving, monitoring/drift detection, and
containerized deployment.

Full requirements and rationale: [`PROJECT.md`](PROJECT.md).
Session-by-session build plan and logs: [`SESSIONS.md`](SESSIONS.md).

This is a synthetic dataset and system built for demonstration purposes —
it does not use or represent real financial data.

## Architecture

```
Transaction stream (Kafka)  →  Feature store (Feast + Redis)
                                        ↓
      Orchestration (Airflow)  →  Model training  →  Model registry (MLflow)
      [local: scikit-learn/PyTorch]  [cloud: SageMaker]
                                        ↓
Live serving (Triton/FastAPI)  →  Monitoring (Prometheus/Grafana/Evidently)  →  Deployment (Docker/K8s/Terraform)
```

Four models are trained and compared: Logistic Regression (baseline), Random
Forest, Isolation Forest (unsupervised), and an LSTM (sequence-based).

## Repo layout

| Path | Purpose |
|---|---|
| `producer/` | Kafka producer replaying synthetic transactions as a live stream |
| `consumer/` | Kafka consumer writing to the data lake and feature store |
| `feature_store/` | Feast feature definitions (offline + online/Redis) |
| `dags/` | Airflow DAGs (ingest → validate → feature engineer → materialize) |
| `validation/` | Great Expectations checks run against each ingested batch |
| `training/` | Local and SageMaker model training code |
| `serving/` | Triton model repo + FastAPI gateway |
| `monitoring/` | Prometheus config, Grafana dashboards, Evidently reports |
| `infra/` | Terraform module (SageMaker training job resources only) |
| `k8s/` | Kubernetes manifests for the serving layer |
| `docs/` | Dataset docs, model comparison notes, registry promotion criteria |

## Quickstart

`docker-compose.yml` is now finalized (Session 10): a single command brings
up the entire stack (Kafka, Redis, Postgres, Airflow, MLflow, Triton, the
FastAPI gateway, Prometheus, Grafana), wired with `depends_on` +
healthchecks so services start in the correct order automatically.

Copy `.env.example` to `.env` first (fill in a generated Fernet key and
webserver secret key for Airflow -- see the comments in that file), then:

```bash
docker compose up -d
```

This assumes the dataset/model artifacts from earlier sessions already
exist on disk (`data/raw/*.parquet`, `feature_store/feature_repo/registry.db`,
`serving/model_metadata.json`, `serving/triton_model_repo/fraud_rf/1/model.onnx`)
-- see `docs/dataset.md`, `docs/feature_store.md`, and `docs/serving.md` for
how to (re)generate each from scratch on a truly clean checkout; these are
gitignored on purpose (large, regenerable, session-specific) rather than
committed.

Once healthy:

```bash
curl -X POST http://localhost:8090/predict -H "Content-Type: application/json" \
  -d '{"account_id":"acct_000000","amount":123.45,"merchant_category":"electronics"}'
```

Airflow UI: http://localhost:8080 (`admin` / whatever you set in `.env`).
Grafana: http://localhost:3000. MLflow: http://localhost:5000.

To replay synthetic transactions through the live Kafka pipeline:

```bash
python consumer/consume.py          # writes valid records to data/lake/, pushes features live
python producer/produce.py --limit 5000 --rate 200   # replays the dataset
```

See [`docs/kafka.md`](docs/kafka.md) for the message schema, offset/replay
strategy, and error-handling behavior; [`docs/feature_store.md`](docs/feature_store.md)
for how the Feast/Redis feature store keeps training and serving consistent;
[`docs/orchestration.md`](docs/orchestration.md) for the Airflow DAG;
[`docs/serving.md`](docs/serving.md) for the gateway and load-test results;
and [`docs/deployment.md`](docs/deployment.md) for the Session 10
Docker Compose / Kubernetes / Terraform writeup, including
[`k8s/README.md`](k8s/README.md) and [`infra/terraform/README.md`](infra/terraform/README.md).

## Metrics

Real numbers (dataset scale, precision/recall/latency) will be filled in
here as they're measured — no placeholders. See `docs/` once training and
serving sessions are complete.

## What I'd do with more time

_(filled in during the final polish session — see SESSIONS.md Session 11)_
