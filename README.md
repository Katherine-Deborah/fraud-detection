# Fraud Detection Pipeline

**Status: in progress — Sessions 0–4 complete (repo scaffold, synthetic dataset generation, Kafka ingestion, Feast/Redis feature store, Airflow orchestration + Great Expectations validation).**

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

Services are added to `docker-compose.yml` incrementally as each session
implements them.

```bash
docker compose up -d kafka redis    # single-node Kafka (KRaft mode) + Redis
python consumer/consume.py          # writes valid records to data/lake/, pushes features live
python producer/produce.py --limit 5000 --rate 200   # replays the dataset
```

See [`docs/kafka.md`](docs/kafka.md) for the message schema, offset/replay
strategy, and error-handling behavior (malformed messages, out-of-order
timestamps, consumer restarts), and [`docs/feature_store.md`](docs/feature_store.md)
for how the Feast/Redis feature store keeps training and serving consistent.

Copy `.env.example` to `.env` (fill in a generated Fernet key and webserver
secret key -- see the comments in that file) before bringing up Airflow:

```bash
docker compose up -d postgres
docker compose up airflow-init          # one-time: migrates the metadata DB, creates the admin user
docker compose up -d airflow-webserver airflow-scheduler
```

Airflow UI: http://localhost:8080 (`admin` / whatever you set in `.env`).
See [`docs/orchestration.md`](docs/orchestration.md) for the DAG's stages,
the Great Expectations checks, and how this batch job relates to the
real-time Kafka path. The rest of the stack comes online in later sessions:

```bash
docker compose up
```

## Metrics

Real numbers (dataset scale, precision/recall/latency) will be filled in
here as they're measured — no placeholders. See `docs/` once training and
serving sessions are complete.

## What I'd do with more time

_(filled in during the final polish session — see SESSIONS.md Session 11)_
