# Fraud Detection Pipeline

**Status: in progress — Session 0 (environment & repo scaffold) complete.**

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
| `dags/` | Airflow DAGs (ingest → validate → feature engineer → train) |
| `training/` | Local and SageMaker model training code |
| `serving/` | Triton model repo + FastAPI gateway |
| `monitoring/` | Prometheus config, Grafana dashboards, Evidently reports |
| `infra/` | Terraform module (SageMaker training job resources only) |
| `k8s/` | Kubernetes manifests for the serving layer |
| `docs/` | Dataset docs, model comparison notes, registry promotion criteria |

## Quickstart

Services are added to `docker-compose.yml` incrementally as each session
implements them — there is nothing to run yet. Once the stack is complete:

```bash
docker compose up
```

## Metrics

Real numbers (dataset scale, precision/recall/latency) will be filled in
here as they're measured — no placeholders. See `docs/` once training and
serving sessions are complete.

## What I'd do with more time

_(filled in during the final polish session — see SESSIONS.md Session 11)_
