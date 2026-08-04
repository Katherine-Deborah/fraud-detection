# IAM setup for Session 6 (SageMaker)

Two distinct IAM principals are needed — don't conflate them:

1. **Your CLI user** — the credentials you put into `aws configure` on this
   machine. Needs `cli_user_permissions_policy.json`: enough to create the
   training bucket, launch/monitor training jobs, pass the execution role
   to SageMaker, set the budget alert, and check for leftover endpoints —
   nothing else. Not root, not `AdministratorAccess`.
2. **The SageMaker execution role** — a role (not a user) that the
   SageMaker *service* assumes while your training job runs. It never has
   credentials of its own; your CLI user grants it permission via
   `iam:PassRole` (already scoped into policy 1, to this one role name
   only). Needs `sagemaker_execution_trust_policy.json` (who can assume
   it) and `sagemaker_execution_permissions_policy.json` (what it can do:
   read/write the training bucket, write CloudWatch Logs, pull the
   prebuilt scikit-learn training image from ECR).

Both policies scope S3 access to bucket names starting with
`fraud-detection-sagemaker-` — name your bucket with that prefix (see
`training/sagemaker_upload_data.py`), or edit the `Resource` ARNs to match
whatever name you actually pick.

## Console steps (first time, no AWS CLI needed yet)

1. **CLI user**: IAM → Users → Create user → no console access needed →
   attach policy → "Create policy" → JSON tab → paste
   `cli_user_permissions_policy.json` → name it
   `fraud-detection-cli-policy` → attach it to the new user → Security
   credentials tab → Create access key → "Command Line Interface (CLI)" →
   save the Access Key ID / Secret Access Key (shown once).
2. **Execution role**: IAM → Roles → Create role → Custom trust policy →
   paste `sagemaker_execution_trust_policy.json` → next → "Create policy"
   → JSON tab → paste `sagemaker_execution_permissions_policy.json` → name
   it `fraud-detection-sagemaker-permissions` → attach it → name the role
   exactly `fraud-detection-sagemaker-execution` (the CLI user's
   `iam:PassRole` statement is scoped to this exact name) → copy the
   role's ARN, you'll pass it as `--role-arn` to
   `training/sagemaker_launch.py`.

Full runbook with the AWS CLI install and `aws configure` steps that come
after this: `docs/sagemaker.md`.
