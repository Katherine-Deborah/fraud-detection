# Session 6 — SageMaker training job

**Status as of this write-up: code written and locally smoke-tested; the
live AWS steps (account confirmed, everything after it) have not been run
yet.** This doc is both the design record and the runbook to finish the
session. Read PROJECT.md §7 (cost rules) before doing anything below.

## What's already done (no AWS touched)

- `training/sagemaker_entry.py` — the actual training script that runs
  *inside* the SageMaker container. Reuses `training/data_prep.py`'s
  chronological split, feature engineering, and threshold-selection logic
  verbatim (same `RandomForestClassifier` hyperparameters as Session 5's
  `train_random_forest.py`) so this run is comparable to the local one,
  not a reimplementation. Writes `model.joblib` + `metrics.json` into
  `SM_MODEL_DIR` — it does **not** talk to MLflow directly, because the
  training container has no network route back to the local
  docker-compose MLflow server.
  - **Smoke-tested locally** (`python training/sagemaker_entry.py --train
    data/raw --model-dir <scratch> --n-estimators 10 --max-depth 5
    --min-samples-leaf 50`): loads the real 3.5M-row training split,
    trains, evaluates, writes both output files correctly. Real launches
    use the full defaults (`n-estimators=200`, `max-depth=20`,
    `min-samples-leaf=20`) via `sagemaker_launch.py`, matching Session 5.
- `training/sagemaker_upload_data.py` — boto3 script: creates the S3
  bucket if it doesn't exist, attaches a 14-day lifecycle rule scoped to
  the `training-data/` prefix (PROJECT.md §7's "expire objects" rule), and
  uploads `data/raw/transactions.parquet`.
- `training/sagemaker_launch.py` — uses the `sagemaker` Python SDK's
  `SKLearn` estimator to launch the training job: `ml.m5.large`, spot
  instances by default, `source_dir=training/` +
  `dependencies=[data_generation/]` (the container flattens these into
  `/opt/ml/code`, which is why `sagemaker_entry.py` imports
  `data_prep` bare instead of `training.data_prep`).
- `training/register_sagemaker_model.py` — downloads the resulting
  `model.tar.gz` from S3, extracts `model.joblib` + `metrics.json`, and
  logs a new MLflow run (`random_forest_sagemaker`) tagged
  `trained_via=sagemaker`, alongside the 4 local Session 5 runs.
- `infra/aws_iam/` — policy JSON for two separate IAM principals (see that
  folder's README): your CLI user (scoped, not root/admin) and the
  SageMaker execution role (assumed by the *service*, not you).
- `infra/aws_budget/setup_budget.py` — boto3 Budgets script, idempotent,
  creates a $10/month budget with email alerts at 80% actual spend / 100%
  forecasted spend.

### A real dependency conflict, found and fixed

`mlflow==2.12.2` (already pinned for Session 5) requires `numpy<2` and
`pyarrow<16`. `sagemaker==2.232.2` requires `numpy<2.0`. Both directly
conflict with `feast==0.65.0`'s `numpy>=2.0,<3` / `pyarrow>=16.1.0`
requirement (Session 3). Installing `boto3`+`sagemaker` into the main
project venv silently downgraded numpy/pyarrow and broke feast — caught by
pip's own resolver warning, not silently. Fixed the same way Session 4
handled Airflow's conflicting deps (`Dockerfile.airflow` /
`requirements-airflow.txt`): a second, isolated venv.

- **`.venv`** (main, gitignored) — unchanged, restored to
  `numpy==2.4.6`/`pyarrow==25.0.0`. Runs Kafka/Feast/Airflow-local/Session
  1-5 code. Does **not** have boto3/sagemaker/mlflow installed — never did
  reliably (see caveat below).
- **`.venv-sagemaker`** (new, gitignored) — installs
  `requirements-sagemaker.txt` (boto3, sagemaker SDK, mlflow, scikit-learn,
  joblib, pandas, pyarrow==15.0.2). Used only for the four
  `training/sagemaker_*.py` / `register_sagemaker_model.py` scripts, which
  never import Feast/Kafka code. Verified all three non-entry scripts
  import cleanly here (`sagemaker_entry.py` was run end-to-end, see above).

**Caveat surfaced while debugging this:** the main venv, before this
session, had *no* scikit-learn/mlflow/torch installed at all despite all
three being pinned in `requirements.txt` since Session 5 — meaning
Session 5's own scripts (`training/train_*.py`) will not currently run
against `.venv` as-is. Not fixed here (out of scope for what this session
needs — the isolated `.venv-sagemaker` happens to also satisfy
scikit-learn/mlflow, but not torch/feast together). If you hit
`ModuleNotFoundError` running a Session 5 script, `pip install -r
requirements.txt` in `.venv` first and expect the same
numpy/pyarrow-vs-mlflow tension described above if you ever add
`sagemaker`/`boto3` there directly — don't.

There's also a harmless `mlflow` vs `mlflow-skinny` version-mismatch
warning in `.venv-sagemaker` (a transitive pin from `sagemaker-mlflow`,
which this project doesn't use) — cosmetic only, imports and the smoke
test both work.

## Runbook — what's left, in order

### 1. AWS account
Confirmed done — account created.

### 2. IAM setup
Follow `infra/aws_iam/README.md`: create the scoped CLI user (attach
`cli_user_permissions_policy.json`) and the SageMaker execution role
(trust + permissions policies in the same folder). Save the CLI user's
access key ID/secret and the execution role's ARN — you'll need both next.

### 3. Install & configure AWS CLI
```
# Windows: winget install Amazon.AWSCLI  (or the MSI from AWS's site)
aws configure
# paste the CLI user's Access Key ID / Secret Access Key, region e.g. us-east-1
aws sts get-caller-identity   # confirms it's the scoped user, not root
```

### 4. Budget alert (mandatory before step 5, per PROJECT.md §7 rule 1)
```
.venv-sagemaker/Scripts/python infra/aws_budget/setup_budget.py --email <your-email> --limit 10
```
Verify in the console: Billing → Budgets.

### 5. Upload training data
```
.venv-sagemaker/Scripts/python training/sagemaker_upload_data.py --bucket fraud-detection-sagemaker-<your-account-id>
```
(Bucket name must start with `fraud-detection-sagemaker-` — the IAM
policies in step 2 are scoped to that prefix.) Note the printed S3 URI.

### 6. Launch the training job
```
.venv-sagemaker/Scripts/python training/sagemaker_launch.py \
  --role-arn arn:aws:iam::<account-id>:role/fraud-detection-sagemaker-execution \
  --s3-input-uri s3://fraud-detection-sagemaker-<account-id>/training-data/rf-session6/
```
Runs in the foreground (`wait=True` by default) and prints the job name +
model S3 URI when done. Expect several minutes (container pull + ~10 min
of training at full `n_estimators=200` on the full 3.5M-row training
split, similar to Session 5's local 614.6s run) plus spot-instance
queueing time, which can vary.

### 7. Register the artifact in MLflow
Bring up the local stack first if it's not already running
(`docker compose up -d mlflow`), then:
```
.venv-sagemaker/Scripts/python training/register_sagemaker_model.py \
  --model-data <printed by step 6> --job-name <printed by step 6>
```
Confirm in the MLflow UI (`http://localhost:5000`): a
`random_forest_sagemaker` run should appear alongside the 4 Session 5
runs, tagged `trained_via=sagemaker`.

### 8. Mandatory cost/safety check (do not skip)
```
aws sagemaker list-endpoints            # must be empty -- no endpoint was created this session
aws sagemaker list-notebook-instances   # must be empty -- no notebook was created this session
aws s3api get-bucket-lifecycle-configuration --bucket fraud-detection-sagemaker-<account-id>
```
Record the actual dollar cost incurred (visible in Billing → Cost
Explorer or the training job's billing details, typically well under $1
for one ~10-minute `ml.m5.large` spot job) in `SESSIONS.md`'s Session 6
log, then check off the remaining Session 6 tasks there.

## Why Random Forest, not the LSTM, for this session

PROJECT.md §5.5 says "recommend Random Forest or the LSTM." Random Forest
was already the stated production candidate in `docs/model_comparison.md`
(highest AUC-PR), and SageMaker's script-mode `SKLearn` container is a
much smaller lift than packaging a custom PyTorch container/entry point
for the LSTM's sequence-windowing logic. The LSTM's full-5M-row-scale
retrain (flagged as future work in `docs/model_comparison.md`) is a
reasonable Session 6 stretch goal but not required for the session's
"at least one model" definition of done.

## Framework version note

`sagemaker_launch.py` pins `framework_version="1.2-1"` for the managed
SKLearn container (AWS's prebuilt SageMaker SKLearn images don't ship
every PyPI scikit-learn version) vs. `scikit-learn==1.4.2` pinned locally.
`RandomForestClassifier`'s relevant API (`class_weight`, `n_estimators`,
`max_depth`, `min_samples_leaf`, `predict_proba`) is stable across that
gap — no behavior-relevant difference expected for this model.
