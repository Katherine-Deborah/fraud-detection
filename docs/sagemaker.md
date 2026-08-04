# Session 6 — SageMaker training job

**Status as of this write-up: account created, IAM configured, budget
alert live, training data uploaded to S3. The training job launch itself
is blocked on an AWS Service Quota increase for `ml.m5.large` for training
job usage** (new accounts default to a quota of 0 for SageMaker training
instances) **— request submitted 2026-08-03, pending AWS approval.**
Nothing else in the runbook can proceed until that's granted. This doc is
both the design record and the runbook to finish the session. Read
PROJECT.md §7 (cost rules) before doing anything below.

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

## Bugs found during live testing (2026-08-03)

Two more real bugs surfaced once actual AWS calls were being made, not
just syntax/import checks — both fixed in the same live-testing pass:

- **IAM policy had non-existent action names.** The original
  `cli_user_permissions_policy.json` granted
  `s3:PutBucketLifecycleConfiguration` / `s3:GetBucketLifecycleConfiguration`
  — these aren't real IAM actions. S3's lifecycle-config API operations map
  to `s3:PutLifecycleConfiguration` / `s3:GetLifecycleConfiguration` (no
  "Bucket" in the name), caught via a real `AccessDenied` error while
  `sagemaker_upload_data.py` tried to attach the 14-day lifecycle rule.
  Also dropped `s3:HeadBucket`, which isn't a distinct IAM action —
  `HeadBucket` calls are actually authorized by `s3:ListBucket`, already
  granted separately. Fixed; re-ran the upload successfully.
- **`setup_budget.py`'s rerun path silently didn't update the alert
  email.** `budgets:UpdateBudget` can't change notification subscribers,
  so rerunning the script with a corrected email left alerts pointed at
  the old address with no error. Changed the idempotent path to delete and
  recreate the budget on rerun instead of updating in place.

## Runbook — what's left, in order

### 1. AWS account — done
Account created.

### 2. IAM setup — done
Followed `infra/aws_iam/README.md`: scoped CLI user + SageMaker execution
role both created, access key ID/secret and the execution role's ARN
saved.

### 3. Install & configure AWS CLI — done
```
# Windows: winget install Amazon.AWSCLI  (or the MSI from AWS's site)
aws configure
# paste the CLI user's Access Key ID / Secret Access Key, region e.g. us-east-1
aws sts get-caller-identity   # confirms it's the scoped user, not root
```

### 4. Budget alert — done (mandatory before step 5, per PROJECT.md §7 rule 1)
```
.venv-sagemaker/Scripts/python infra/aws_budget/setup_budget.py --email <your-email> --limit 10
```
Verified in the console: Billing → Budgets. Hit and fixed a real bug here
during live testing — see "Bugs found during live testing" below.

### 5. Upload training data — done
```
.venv-sagemaker/Scripts/python training/sagemaker_upload_data.py --bucket fraud-detection-sagemaker-<your-account-id>
```
(Bucket name must start with `fraud-detection-sagemaker-` — the IAM
policies in step 2 are scoped to that prefix.) Note the printed S3 URI.
Also hit and fixed a real IAM bug here — see below.

### 6. Launch the training job — BLOCKED on AWS quota approval
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

**Currently blocked:** launching returns `ResourceLimitExceeded` — new AWS
accounts default to a **`ml.m5.large` for training job usage** quota of 0.
A Service Quota increase request was submitted 2026-08-03 via the Service
Quotas console (SageMaker → training job usage) and is pending AWS
approval; there's no way to train on managed SageMaker until it's granted.
Nothing to do here but wait and re-run step 6 once the request is
approved — the launch script itself (after the bucket/region fix below) is
believed ready to go.

**Uncommitted fix in `training/sagemaker_launch.py`, made while chasing
this down, not yet the actual quota problem but two real bugs surfaced on
the way to it:** the SageMaker SDK defaults to staging code and model
output in an auto-named `sagemaker-<region>-<account-id>` bucket that our
IAM policy correctly doesn't grant access to (scoped to
`fraud-detection-sagemaker-*` only) — fixed by deriving `code_location`
and `output_path` from the same bucket passed via `--s3-input-uri`. Also
found the boto3 session was resolving to `us-west-2` (the CLI's configured
default region) while the data bucket actually lives in `us-east-1` (from
`sagemaker_upload_data.py`'s own default) — the `SKLearn` estimator has no
standalone `--region` flag, so this now threads a `boto3.Session` with the
explicit `--region` argument through a `sagemaker.session.Session`. Worth
committing alongside this doc update.

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
