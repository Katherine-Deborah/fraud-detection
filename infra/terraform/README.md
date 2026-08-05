# Terraform — SageMaker training resources (Session 10)

Narrow by design, per PROJECT.md §9 ("keep the paid surface area to exactly
one thing") and SESSIONS.md's Session 10 task: this module manages **only**
the two durable, Session-6-created resources —

- the S3 training-data bucket (`s3.tf`), now with the lifecycle rule
  widened to the whole bucket (Session 6 only covered `training-data/`,
  leaving `code/`/`output/` to accumulate forever — see
  `docs/sagemaker.md`'s "Known gap, not fixed") and a public-access block
  added (Session 6 never set one explicitly)
- the SageMaker execution role (`iam.tf`), reusing the exact trust/
  permissions policy JSON already reviewed and tested live in Session 6
  (`infra/aws_iam/*.json`), not a re-authored copy

**Not** in this module, deliberately:
- **The CLI user.** A separate IAM principal (see
  `infra/aws_iam/README.md`) created once, by hand, holding long-lived
  access keys — managing IAM Users/access keys in Terraform state is a
  real secret-handling liability this portfolio project doesn't need to
  take on.
- **The training job itself.** There is no `aws_sagemaker_training_job`
  resource in the AWS provider — Terraform models durable infrastructure
  with a desired steady state; a training job is a one-shot, imperative
  action with a start/end time. `training/sagemaker_launch.py` (Session 6)
  remains how a job actually gets launched, using the role/bucket this
  module creates.
- **The AWS Budget alert** (`infra/aws_budget/setup_budget.py`, Session 6)
  — already idempotent Python, no reason to duplicate it here.

## Prerequisites

- Terraform >= 1.5. This session used a locally-downloaded binary (no
  admin rights available) at `%USERPROFILE%\.local\bin\terraform.exe` —
  add that to your `PATH`, or call it by full path.
- AWS credentials configured (`~/.aws/credentials`) for the scoped CLI
  user described in `infra/aws_iam/README.md` — the same credentials
  `training/sagemaker_launch.py` already uses. Read-only actions
  (`init`/`validate`/`plan`) work with the existing scoped policy;
  `apply`/`destroy` need `s3:CreateBucket`/`s3:PutLifecycleConfiguration`/
  `s3:PutBucketPublicAccessBlock`/`iam:CreateRole`/`iam:PutRolePolicy`/
  `iam:DeleteRole` — audit `cli_user_permissions_policy.json` before
  applying live if you're re-running this with the exact Session 6 policy,
  since it was scoped to what `sagemaker_upload_data.py`/manual console
  steps needed, not what Terraform needs.

## ⚠️ Known collision if you actually `apply` this against the existing account

**Verified this session, not theoretical:** `terraform plan -var="region=us-east-1"`
computed `bucket_name = "fraud-detection-sagemaker-183079729790"` — which
is the **exact bucket Session 6 already created** by hand via
`training/sagemaker_upload_data.py` (see `docs/sagemaker.md`). This module
was written, `init`'d, `validate`'d, and `plan`'d live against real AWS
credentials this session (5 resources to add, 0 changed/destroyed) —
**but never `apply`'d**, specifically to avoid this collision without a
deliberate decision first. Before ever running `apply` for real:

1. **Import the existing resources instead of recreating them:**
   ```bash
   terraform import -var="region=us-east-1" aws_s3_bucket.training_data fraud-detection-sagemaker-183079729790
   terraform import -var="region=<role's actual region>" aws_iam_role.sagemaker_execution fraud-detection-sagemaker-execution
   ```
   then `terraform plan` again and reconcile any drift (e.g. the
   lifecycle-rule-scope difference this module intentionally introduces),
   **or**
2. Pick a different `bucket_name` (and matching `sagemaker_execution_role_name`
   if you also want a separate role) so this module manages a fresh,
   parallel set of resources instead of colliding with the hand-created
   ones.

## Usage

```bash
cd infra/terraform
terraform init

# Read-only, no resources touched -- confirms the module is well-formed
# against real AWS auth. This is as far as this session went.
terraform plan -var="region=<region with your ml.m5.large training quota>"

# Only after resolving the collision note above:
terraform apply -var="region=..."
```

Outputs (`bucket_name`, `sagemaker_execution_role_arn`) feed directly into
`training/sagemaker_upload_data.py --bucket` and
`training/sagemaker_launch.py --role-arn`.

## Tearing down

**Never leave `apply`'d resources around after a demo** — PROJECT.md §7's
cost rules exist for exactly this. `destroy_reminder.py` in this directory
runs `terraform destroy` with an explicit confirmation prompt, then re-runs
the same `list-endpoints`/`list-notebook-instances` safety checks every
prior AWS-touching session in `SESSIONS.md` has run by hand:

```bash
.venv-sagemaker/Scripts/python infra/terraform/destroy_reminder.py --region us-east-1
```
