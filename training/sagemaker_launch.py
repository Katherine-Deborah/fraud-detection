"""Launch the Session 6 SageMaker Training Job: Random Forest, script-mode,
same hyperparameters as the local Session 5 run (training/train_random_forest.py),
on a spot ml.m5.large instance.

Prerequisites (full runbook in docs/sagemaker.md):
  - AWS CLI configured (`aws configure`) with a scoped IAM user, not root.
  - An AWS Budget alert already set (infra/aws_budget/setup_budget.py).
  - Training data already uploaded: python training/sagemaker_upload_data.py --bucket <bucket>
  - A SageMaker execution role ARN (see infra/aws_iam/) that the SageMaker
    *service* assumes to run the job -- distinct from your own CLI user's
    credentials.

Usage:
    python training/sagemaker_launch.py \\
        --role-arn arn:aws:iam::<account-id>:role/fraud-detection-sagemaker-execution \\
        --s3-input-uri s3://fraud-detection-sagemaker-<account-id>/training-data/rf-session6/

Cost/safety: launches one billable ml.m5.large training job (spot pricing
by default -- typically well under $1 for a run this size, a few minutes
of training) -- see PROJECT.md §7 and docs/sagemaker.md. Does NOT create
a persistent endpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sagemaker.sklearn.estimator import SKLearn

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--s3-input-uri", required=True, help="S3 prefix from sagemaker_upload_data.py")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--use-spot", dest="use_spot", action="store_true", default=True)
    parser.add_argument("--no-spot", dest="use_spot", action="store_false")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--no-wait", dest="wait", action="store_false", default=True)
    args = parser.parse_args()

    estimator = SKLearn(
        entry_point="sagemaker_entry.py",
        source_dir=str(REPO_ROOT / "training"),
        dependencies=[str(REPO_ROOT / "data_generation")],
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        base_job_name="fraud-rf-sagemaker",
        hyperparameters={
            "n-estimators": args.n_estimators,
            "max-depth": args.max_depth,
            "min-samples-leaf": args.min_samples_leaf,
        },
        use_spot_instances=args.use_spot,
        max_run=1800,
        max_wait=3600 if args.use_spot else None,
    )

    print(f"launching training job on {args.instance_type} (spot={args.use_spot}) ...")
    estimator.fit({"train": args.s3_input_uri}, wait=args.wait)

    if args.wait:
        print(f"\njob name: {estimator.latest_training_job.name}")
        print(f"model artifact S3 URI: {estimator.model_data}")
        print("\nnext step -- register the artifact into the local MLflow registry:")
        print(
            f"  python training/register_sagemaker_model.py "
            f"--model-data {estimator.model_data} "
            f"--job-name {estimator.latest_training_job.name}"
        )
    else:
        print("launched without waiting -- check status with:")
        print("  aws sagemaker list-training-jobs --max-results 5")


if __name__ == "__main__":
    main()
