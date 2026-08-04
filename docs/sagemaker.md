# Session 6 — SageMaker training job

**Status: complete.** A Random Forest model was trained as a managed
SageMaker Training Job and registered into the local MLflow registry
tagged `trained_via=sagemaker`, matching Session 5's local Random Forest
result closely (AUC-PR 0.655 vs. 0.655, recall 86.3% vs. 86.3% — see
"Final result" below). This doc is the design record and the record of
everything hit getting there; read PROJECT.md §7 (cost rules) if revisiting
this pattern for a future session.

## Final result (2026-08-04)

- **Job:** `fraud-rf-sagemaker-2026-08-04-18-33-43-040`, on-demand
  `ml.m5.large`, region **`ap-southeast-2` (Sydney)** — not `us-east-1`,
  see "Region mismatches" below for why.
- **Billable time:** 1,684 seconds (~28 min, vs. Session 5's local 614.6s
  — `ml.m5.large` has only 2 vCPUs, materially less parallelism than the
  local dev machine for `RandomForestClassifier`'s `n_jobs`).
- **Metrics:** AUC-PR 0.6554, recall 0.8628 @ threshold 0.373 (val FPR
  0.0199 against the 2% target) — consistent with Session 5's local run
  (0.655 / 86.3%) to 3 decimal places, a good end-to-end correctness
  signal that script-mode training reproduces the local result.
- **Registered:** MLflow run `random_forest_sagemaker`
  (`http://localhost:5000`, experiment id 1), tagged
  `trained_via=sagemaker` + `sagemaker_job_name`, alongside the 4 Session 5
  local runs.
- **AWS spend:** ~$0.07 for the training job (1,684 billable seconds ×
  an estimated ~$0.145/hr on-demand `ml.m5.large` rate in `ap-southeast-2`
  — not independently verified against Cost Explorer, since the scoped CLI
  user intentionally has no billing-read permissions; check the console
  for the authoritative number). Plus negligible S3 storage for the
  326 MB training parquet, uploaded to **two** buckets due to the region
  mismatch below — both already have the 14-day lifecycle rule and expire
  automatically (`fraud-detection-sagemaker-183079729790` in `us-east-1`
  on 2026-08-17; `fraud-detection-sagemaker-183079729790-syd` in
  `ap-southeast-2` on 2026-08-18).

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

There's also an `mlflow` vs `mlflow-skinny` version-mismatch warning in
`.venv-sagemaker` (a transitive pin from `sagemaker-mlflow`, which this
project doesn't use) — **this was originally logged here as "cosmetic
only," which turned out to be wrong; see "Bugs found registering the
artifact" below.**

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

## Bugs found launching the job (2026-08-04)

- **Spot and on-demand training quotas are separate.** The Service Quota
  increase approved ahead of this session was for `ml.m5.large for
  training job usage` (on-demand). `sagemaker_launch.py` defaults to
  `use_spot_instances=True`, which draws against a *different* quota —
  `ml.m5.large for spot training job usage` — still 0. First launch
  attempt failed with `ResourceLimitExceeded` on the spot quota; retried
  with `--no-spot` and got a second, more surprising `ResourceLimitExceeded`
  on the on-demand quota that was supposedly just approved — see next bug.
- **The quota was approved in the wrong region.** AWS Service Quotas are
  per-region. The approval email specified **Asia Pacific (Sydney) /
  `ap-southeast-2`** — not `us-east-1`, where the training bucket and all
  prior testing lived, and not `us-west-2` (the CLI's configured default
  region either, for that matter). The scoped CLI user has no
  `servicequotas:*` permissions (by design — not needed for day-to-day
  runs), so this couldn't be queried directly; confirmed by reading the
  approval notification text. Fix: re-ran `sagemaker_upload_data.py`
  against a **second** bucket, `fraud-detection-sagemaker-183079729790-syd`,
  in `ap-southeast-2`, and launched there instead — see "Region
  mismatches" note. **Takeaway for next time:** always request quota
  increases in the same region you intend to actually use, and confirm the
  approval email's region line before assuming a quota applies where you
  think it does.
- **`framework_version="1.2-1"` was not actually behavior-compatible with
  local `scikit-learn==1.4.2`, despite this doc previously saying it was
  (see the now-corrected "Framework version note" below).** `joblib.load`
  of the SageMaker-trained `model.joblib` failed with `ValueError: node
  array from the pickle has an incompatible dtype` — scikit-learn 1.3
  added a `missing_go_to_left` field to the internal tree-node C struct
  (to support missing-value splits), which changes the pickle layout for
  every tree-based estimator, `RandomForestClassifier` included. Fixed by
  pinning `.venv-sagemaker`'s scikit-learn to `1.2.1` (exact match to the
  container's `1.2-1` framework version) — **`requirements-sagemaker.txt`
  needs the same pin, not yet updated in that file as of this write-up.**
- **The "cosmetic" `mlflow`/`mlflow-skinny` mismatch from Session 6 part 1
  was not actually cosmetic.** `.venv-sagemaker` had `mlflow==2.12.2` but
  `mlflow-skinny==3.15.1` (a stray transitive pull, likely via
  `sagemaker`'s own `sagemaker-mlflow` extra). The *effective* runtime
  behavior followed the newer skinny package, so `mlflow.sklearn.log_model`
  called MLflow 3.x's `/api/2.0/mlflow/logged-models` endpoint, which the
  `ghcr.io/mlflow/mlflow:v2.12.2` tracking server (docker-compose, Session
  5) doesn't have — `404 Not Found`, only surfaced once an actual
  `log_model` call was made against a live server, not on import. Fixed by
  `pip install --force-reinstall mlflow==2.12.2 mlflow-skinny==2.12.2`
  together. That reinstall's dependency resolution then silently pulled
  scikit-learn back up to `1.9.0` (no version pin on scikit-learn from
  either mlflow package), re-breaking the pickle-compatibility fix above —
  had to re-pin `scikit-learn==1.2.1` a second time, *after* the mlflow
  reinstall, for both fixes to hold simultaneously. **Order matters here:
  install mlflow/mlflow-skinny first, scikit-learn last**, if rebuilding
  this venv from scratch. `requirements-sagemaker.txt` should pin
  `mlflow-skinny==2.12.2` explicitly alongside `mlflow==2.12.2` and
  `scikit-learn==1.2.1` to make this reproducible — **not yet done as of
  this write-up, do this before the next SageMaker session.**

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

### 6. Launch the training job — done
```
.venv-sagemaker/Scripts/python training/sagemaker_launch.py \
  --role-arn arn:aws:iam::183079729790:role/fraud-detection-sagemaker-execution \
  --s3-input-uri s3://fraud-detection-sagemaker-183079729790-syd/training-data/rf-session6/ \
  --region ap-southeast-2 --no-spot
```
Note this is **not** the command shown in the script's own docstring —
`--region ap-southeast-2` (to match where the quota was actually approved)
and `--no-spot` (the spot quota is separate and still 0) are both required
overrides. See "Bugs found launching the job" above for the full story;
`training/sagemaker_launch.py`'s `code_location`/`output_path`/region
fix from the earlier blocked attempt is committed and part of why this
worked. Took ~28 minutes billable (1,684s) — longer than the ~10 min
estimate in this doc's earlier draft, since `ml.m5.large`'s 2 vCPUs give
`RandomForestClassifier` much less parallelism than the local dev machine.

### 7. Register the artifact in MLflow — done
Bring up the local stack first if it's not already running
(`docker compose up -d mlflow`), then:
```
.venv-sagemaker/Scripts/python training/register_sagemaker_model.py \
  --model-data <printed by step 6> --job-name <printed by step 6>
```
Confirmed in the MLflow UI (`http://localhost:5000`): the
`random_forest_sagemaker` run appears alongside the 4 Session 5 runs,
tagged `trained_via=sagemaker`. Required two dependency fixes first — see
"Bugs found launching the job" above (scikit-learn version pin, then the
mlflow/mlflow-skinny mismatch, in that order).

### 8. Mandatory cost/safety check — done
```
aws sagemaker list-endpoints --region ap-southeast-2            # empty
aws sagemaker list-endpoints --region us-east-1                 # empty
aws sagemaker list-notebook-instances --region ap-southeast-2   # empty
aws sagemaker list-notebook-instances --region us-east-1        # empty
aws s3api get-bucket-lifecycle-configuration --bucket fraud-detection-sagemaker-183079729790-syd --region ap-southeast-2
aws s3api get-bucket-lifecycle-configuration --bucket fraud-detection-sagemaker-183079729790 --region us-east-1
```
All confirmed clean in both regions touched this session. Actual spend:
see "Final result" above (~$0.07 training compute + negligible storage).

**Known gap, not fixed:** the 14-day lifecycle rule only covers the
`training-data/` prefix. `sagemaker_launch.py`'s `code_location`/
`output_path` fix writes to `code/` and `output/` in the same bucket,
which have **no** expiration rule — the ~326 MB training data will
self-clean, but the (much smaller — low MB) code bundle and model
artifact under those prefixes will persist indefinitely unless manually
deleted. Low cost impact at this size, but worth widening the lifecycle
rule to the whole bucket next time this is touched, now that the model
artifact is safely inside MLflow's own artifact store and the S3 copy is
redundant.

## Why Random Forest, not the LSTM, for this session

PROJECT.md §5.5 says "recommend Random Forest or the LSTM." Random Forest
was already the stated production candidate in `docs/model_comparison.md`
(highest AUC-PR), and SageMaker's script-mode `SKLearn` container is a
much smaller lift than packaging a custom PyTorch container/entry point
for the LSTM's sequence-windowing logic. The LSTM's full-5M-row-scale
retrain (flagged as future work in `docs/model_comparison.md`) is a
reasonable Session 6 stretch goal but not required for the session's
"at least one model" definition of done.

## Framework version note (corrected 2026-08-04)

`sagemaker_launch.py` pins `framework_version="1.2-1"` for the managed
SKLearn container. **This section originally claimed the gap to
`scikit-learn==1.4.2` locally had "no behavior-relevant difference
expected" — that was wrong.** `RandomForestClassifier`'s fit/predict API
is stable across the gap, but the *pickled model's binary layout* is not:
scikit-learn 1.3 changed the internal tree-node struct, so a model
trained under 1.2.1 cannot be `joblib.load`ed under 1.4.2 or later. The
container version and the loading environment's scikit-learn version must
match exactly for deserialization, even though the training/inference API
itself didn't change. `.venv-sagemaker` now pins `scikit-learn==1.2.1` for
this reason (see "Bugs found launching the job" above).
