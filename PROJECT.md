# Fraud Detection Pipeline — Project Description & Requirements

## 1. Purpose

A portfolio-grade, end-to-end fraud detection system that demonstrates the full breadth of a modern ML/data engineering stack: streaming ingestion, a feature store, orchestrated training, cloud-based training (SageMaker), a model registry, real-time serving, monitoring/drift detection, and containerized deployment.

This is not a notebook. It is a runnable system with a GitHub repo, a working demo, and real numbers (dataset scale, precision/recall/latency) that survive a "walk me through this project" interview question.

## 2. Why this project, and who it's for

Fraud detection is chosen deliberately over alternatives because:

- It is instantly legible to any interviewer — no domain explanation needed, unlike niche academic projects.
- It naturally requires both streaming (Kafka) and batch (Airflow) — the combination most Data Engineer JDs ask for verbatim.
- It maps directly onto real teams at several target companies:
  - **Meta / TikTok** — Trust & Safety / integrity engineering (fraud, spam, abuse at streaming scale)
  - **Cloudflare** — security-adjacent real-time ML infra
  - **NVIDIA** — GPU-served inference (Triton) is a legitimate differentiator here
  - **Delta** — transaction/booking fraud is a real business problem
  - **Tesla, Annapurna, Nokia** — infra-general roles care more about the pipeline shape than the domain
- It is weaker fit for **Cohere** (LLM-first) and **PathAI** (healthcare-first) — that's fine, those are covered by other portfolio projects (Provider Match Dashboard, AIVC neuron work).

## 3. Architecture overview

```
Transaction stream (Kafka)  →  Feature store (Feast + Redis)
                                        ↓
      Orchestration (Airflow)  →  Model training  →  Model registry (MLflow)
      [local: scikit-learn/PyTorch]  [cloud: SageMaker]
                                        ↓
Live serving (Triton/FastAPI)  →  Monitoring (Prometheus/Grafana/Evidently)  →  Deployment (Docker/K8s/Terraform)
```

Four models are trained and compared: **Logistic Regression** (baseline), **Random Forest**, **Isolation Forest** (unsupervised anomaly detection), and an **LSTM** (sequence-based, transaction history per account).

## 4. Tech stack & rationale

| Layer | Tool | Why it's here | Cost |
|---|---|---|---|
| Streaming ingestion | Kafka (via Docker) | Real-time transaction events; the "message queue" DE JDs ask for | Free, self-hosted |
| Feature store | Feast + Redis | Consistent features between training and serving — the #1 ML infra keyword missing from most portfolios | Free, self-hosted |
| Orchestration | Airflow | DAG-based retraining and validation scheduling | Free, self-hosted |
| Data validation | Great Expectations | Automated schema/quality checks in the DAG | Free, open source |
| Local training | scikit-learn, PyTorch (LSTM) | Fast iteration, no cloud cost during development | Free |
| Cloud training | **AWS SageMaker** | Learning objective — cloud-native training job, hyperparameter tuning, managed infra | **Paid** — use AWS Free Tier + strict guardrails (see §7) |
| Experiment tracking / registry | MLflow | Versioned models, staged rollout (shadow → canary → full) | Free, self-hosted |
| Serving | Triton Inference Server + FastAPI gateway | GPU-capable model serving; speaks directly to NVIDIA-style roles | Free, self-hosted (CPU mode works fine without a GPU) |
| Monitoring | Prometheus, Grafana, Evidently | Latency/throughput dashboards + data/prediction drift detection | Free, self-hosted |
| Deployment | Docker Compose (full local stack), Kubernetes manifests (kind/minikube), Terraform (for the SageMaker pieces only) | Reproducible environment, IaC story for infra-heavy roles | Free, self-hosted |

**Explicitly excluded:** no LLM explainer layer, no other paid API calls, no always-on cloud hosting unless the person later decides to add a free-tier public demo link.

## 5. Functional requirements

### 5.1 Synthetic dataset
- Generate a synthetic transaction dataset (target: 5M+ rows) with realistic class imbalance (~0.1–0.3% fraud rate).
- Fields: transaction_id, account_id, timestamp, amount, merchant_category, location, device_id, is_fraud (label), plus 10–15 engineered behavioral features (velocity, average ticket size, geo-distance from last transaction, etc.).
- Class imbalance handled via SMOTE or class weighting — document which, and why, per model.

### 5.2 Streaming ingestion (Kafka)
- A producer simulates real-time transaction events at a configurable rate (transactions/sec).
- A consumer writes raw events into both a data lake (local disk/S3) and pushes engineered features into the feature store.

### 5.3 Feature store (Feast + Redis)
- Offline store (for training) and online store (Redis, for low-latency serving) share the same feature definitions — this consistency is the whole point.

### 5.4 Orchestration (Airflow)
- DAG stages: ingest → validate (Great Expectations) → feature engineering → training trigger → registry promotion check.
- Runs on a schedule and can be triggered manually for demo purposes.

### 5.5 Model training
- **Local:** all 4 models trained with scikit-learn/PyTorch, tracked in MLflow.
- **Cloud (SageMaker):** at least one model (recommend Random Forest or the LSTM) retrained as a SageMaker Training Job, using a free-tier-eligible instance type, with the trained artifact registered back into MLflow. This is the "I've used a managed cloud training service" credential — see §7 for cost rules.

### 5.6 Model registry
- MLflow Model Registry with staged rollout: Staging → Shadow (log predictions, don't act on them) → Canary (% of traffic) → Production.

### 5.7 Serving
- FastAPI gateway exposing a `/predict` endpoint, backed by Triton for actual model inference.
- Pulls online features from the Feast/Redis store by account_id, not from the request payload — this is what makes the feature store real rather than decorative.

### 5.8 Monitoring
- Prometheus scrapes latency/throughput metrics from the serving layer.
- Grafana dashboard: request rate, p50/p95/p99 latency, fraud-flag rate over time.
- Evidently AI report comparing live feature distributions against the training distribution (drift detection), run on a schedule via Airflow.

### 5.9 Deployment
- `docker-compose.yml` runs the entire stack (Kafka, Redis, Feast, Airflow, MLflow, Triton/FastAPI, Prometheus, Grafana) locally with one command.
- Kubernetes manifests (tested against kind or minikube) for the serving layer, as the "I can operate this beyond docker-compose" story.
- Terraform module scoped narrowly to the SageMaker training job resources only (not the whole stack) — this keeps AWS spend contained to one well-understood piece.

## 6. Non-functional requirements (targets to hit and report)

- Dataset scale: 5M+ synthetic transactions.
- Model quality: recall ≥ 90% at false-positive rate ≤ 2% (report precision/recall/F1/AUC-PR per model — never plain accuracy, given the class imbalance).
- Serving latency: p99 < 50ms per prediction at a sustained load of 500 transactions/sec (load-test this, report the real number).
- Drift detection: demonstrable — inject a synthetic distribution shift and show Evidently flag it.

## 7. Cost management rules (read before touching SageMaker)

1. Set an AWS Budget alert (e.g., $10 threshold) before starting any SageMaker work.
2. Use free-tier-eligible instance types only: `ml.t3.medium` for notebooks, `ml.m5.large` for training jobs (AWS Free Tier covers limited monthly hours for the first 2 months — check current limits, they change).
3. Never leave a SageMaker **endpoint** running — endpoints bill hourly whether or not they're serving traffic. Deploy only for the duration of a test, then delete it immediately (`aws sagemaker delete-endpoint`).
4. Prefer **SageMaker Training Jobs** (billed only for job duration) over persistent notebook instances left open.
5. Use spot instances for training where possible to cut cost further.
6. At the end of every session touching AWS: run `aws sagemaker list-endpoints` and `aws sagemaker list-notebook-instances` and confirm nothing is left running.
7. Everything else in the stack (Kafka, Feast, Redis, Airflow, MLflow, Triton, Prometheus, Grafana, Docker, Kubernetes) runs entirely self-hosted at zero cost — SageMaker is the one deliberate, contained, guardrailed exception.

## 8. Deliverables

- Public GitHub repo with clean commit history (not one giant commit).
- README with: architecture diagram, quickstart (`docker-compose up`), the real metrics from an actual run (not placeholders), and an explicit "what I'd do with more time" section.
- A short demo (screen recording or GIF) showing a transaction flowing through the system and the Grafana dashboard updating live.
- Finalized resume bullets (see below), inserted into the master YAML resume.

**Resume bullets (finalized 2026-08-10, real measured numbers):**
- Built an end-to-end fraud detection system processing streaming transactions via Kafka into a Feast/Redis feature store, training and comparing 4 models (LSTM, Isolation Forest, Random Forest, Logistic Regression) on 5M+ synthetic transactions with a 0.2% fraud rate; Random Forest reached 0.655 AUC-PR and 86.3% recall in production, LSTM reached 88.1% recall
- Trained models locally and via an AWS SageMaker training job, with MLflow-managed experiment tracking and a staged model registry (staging → shadow → canary → production)
- Deployed models via Triton Inference Server behind a FastAPI gateway pulling live features from a Feast/Redis online store; load-tested with Locust (0 failures across 5-100 concurrent users), root-caused a Feast-client concurrency bottleneck via isolated component benchmarks, and built Airflow-orchestrated retraining with Prometheus/Grafana monitoring and an Evidently drift detector that caught a live-injected 5x transaction-amount shift (48.3% → 51.7% drifted columns)
- Containerized the full stack with Docker Compose (8 services, health-check-ordered) and Kubernetes manifests (tested on a local `kind` cluster), with a Terraform module scoping the SageMaker+S3 training infrastructure separately from the always-on stack

_(Insert into the master YAML resume manually — this repo doesn't have
visibility into where that file lives.)_

## 9. Explicitly out of scope

- No LLM/GenAI explanation layer.
- No paid always-on cloud hosting for the demo (unless later added deliberately, on a free tier).
- No AWS services beyond SageMaker Training Jobs (and, narrowly, S3 for data/artifacts if needed) — no RDS, no always-on EC2, no managed Kafka (MSK), etc. Keep the paid surface area to exactly one thing.

## 10. Definition of done

- `docker-compose up` brings up the full stack and a transaction can be traced end-to-end from producer to Grafana dashboard.
- At least one model has been trained via SageMaker and the artifact is visible in the MLflow registry.
- Real metrics (not placeholders) are in the README.
- Repo is public, documented, and linked from the portfolio site and resume.
